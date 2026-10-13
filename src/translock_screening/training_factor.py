"""固定训练预算内的同架构因子对照；不选择测试集最优配置。"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from translock_screening.benchmark import _score_assay, source_split, training_weights

ARMS = {
    "random_frozen": (False, False),
    "pretrained_frozen": (True, False),
    "random_trained": (False, True),
    "pretrained_trained": (True, True),
}


def make_splits(units):
    """沿用整篇留出，记录每条训练侧连接图排除，防止立体异构跨侧泄漏。"""
    if units["unit_id"].duplicated().any() or set(units["label"].unique()) != {0, 1}:
        raise ValueError("单位身份或二分类标签异常")
    splits, members = {}, []
    for document in sorted(units["document_id"].unique()):
        train, test = source_split(units, document)
        if len(set(units.iloc[train]["label"])) != 2:
            raise ValueError(f"训练侧仅一类：{document}")
        excluded = sorted(set(range(len(units))) - set(train) - set(test))
        splits[document] = (train, test)
        for role, indices in [("fit", train), ("held_out", test), ("excluded_overlap", excluded)]:
            for index in indices:
                row = units.iloc[index]
                members.append({"outer_document": document, "role": role, **row.to_dict()})
    return splits, pd.DataFrame(members)


def epoch_orders(n, seed, epochs, offset):
    """单独的随机数发生器使四组顺序不受模型构造/梯度的随机数消耗影响。"""
    generator = np.random.default_rng(seed + offset)
    return [generator.permutation(n) for _ in range(epochs)]


def configure_torch(config):
    import torch

    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("预定CUDA设备不可用")
    return torch


def build_model(payload, pretrained, train_encoder, seed, config, device):
    import torch
    from chemprop import nn

    class FactorModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.encoder = nn.BondMessagePassing(**payload["hyper_parameters"])
            if pretrained:
                self.encoder.load_state_dict(payload["state_dict"], strict=True)
            self.encoder.requires_grad_(train_encoder)
            self.aggregation = nn.MeanAggregation()
            self.normalization = torch.nn.LayerNorm(
                self.encoder.output_dim,
                eps=config["layer_norm_eps"],
                elementwise_affine=False,
            )
            # 头使用独立种子，预训练加载和随机编码器构造不能改变头初值。
            torch.manual_seed(seed + config["head_seed_offset"])
            self.head = torch.nn.Linear(self.encoder.output_dim, 1)

        def encode(self, graph):
            return self.aggregation(self.encoder(graph), graph.batch)

        def from_vectors(self, vectors):
            return self.head(self.normalization(vectors)).squeeze(-1)

        def forward(self, graph):
            return self.from_vectors(self.encode(graph))

    return FactorModel().to(device)


class GraphCache:
    """只缓存无标签的原子键特征，训练图依照固定单位行序组批。"""

    def __init__(self, smiles, device):
        from chemprop import featurizers
        from rdkit import Chem

        self.device = device
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        if tuple(featurizer.shape) != (72, 14):
            raise ValueError("原子键维度与协议不符")
        lookup = {}
        for value in dict.fromkeys(smiles):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError("准入图解析失败")
            lookup[value] = featurizer(molecule)
        self.graphs = [lookup[value] for value in smiles]

    def batch(self, indices):
        from chemprop.data import BatchMolGraph

        batch = BatchMolGraph([self.graphs[int(i)] for i in indices])
        batch.to(self.device)
        return batch


def tensors_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def vector_cache(model, graphs, batch_size):
    import torch

    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                model.encode(
                    graphs.batch(range(start, min(start + batch_size, len(graphs.graphs))))
                )
                for start in range(0, len(graphs.graphs), batch_size)
            ]
        ).detach()


def predict(model, graphs, indices, batch_size, vectors=None):
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            score = (
                model(graphs.batch(part)) if vectors is None else model.from_vectors(vectors[part])
            )
            scores.append(score.detach().cpu().numpy())
    values = np.concatenate(scores).astype(float)
    if not np.isfinite(values).all():
        raise ValueError("模型分数非有限")
    return values


def loss_value(labels, scores, weights):
    """数值稳定的加权BCE；输入logit而非已饱和概率。"""
    return float(np.average(np.logaddexp(0.0, scores) - labels * scores, weights=weights))


def fit_model(
    model, graphs, units, train, seed, learning_rate, config, vectors=None, *, optimizer=None
):
    import torch
    from torch.nn import functional as functional

    before = perf_counter()
    weights_np = training_weights(units.iloc[train].reset_index(drop=True), False)
    labels_np = units.iloc[train]["label"].to_numpy(dtype=float)
    labels = torch.tensor(labels_np, dtype=torch.float32, device=graphs.device)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=graphs.device)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=config["weight_decay"],
        )
    batch_size = config["batch_size"]
    initial = tensors_cpu(model)
    history = []
    clipped_steps = {"encoder": 0, "head": 0}
    schedules = epoch_orders(len(train), seed, config["epochs"], config["order_seed_offset"])
    for epoch in range(config["epochs"] + 1):
        if epoch in {0, 10, config["epochs"]}:
            scores = predict(model, graphs, train, batch_size, vectors)
            history.append(
                {"epoch": epoch, "train_weighted_bce": loss_value(labels_np, scores, weights_np)}
            )
        if epoch == config["epochs"]:
            break
        model.train()
        if vectors is not None:
            model.encoder.eval()
        for start in range(0, len(train), batch_size):
            local = schedules[epoch][start : start + batch_size]
            indices = train[local]
            optimizer.zero_grad(set_to_none=True)
            scores = (
                model(graphs.batch(indices))
                if vectors is None
                else model.from_vectors(vectors[indices])
            )
            loss = (
                functional.binary_cross_entropy_with_logits(scores, labels[local], reduction="none")
                * weights[local]
            ).sum() / batch_size
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限")
            loss.backward()
            for name in ["encoder", "head"]:
                parameters = [p for p in getattr(model, name).parameters() if p.requires_grad]
                if parameters:
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters,
                        config["gradient_clip_norm"],
                        error_if_nonfinite=True,
                    )
                    clipped_steps[name] += int(norm.item() > config["gradient_clip_norm"])
            optimizer.step()
    final = tensors_cpu(model)
    if any(not torch.isfinite(value).all() for value in final.values()):
        raise ValueError("模型权重非有限")
    delta = {
        prefix: max(
            (final[key] - value).abs().max().item()
            for key, value in initial.items()
            if key.startswith(prefix + ".")
        )
        for prefix in ["encoder", "head"]
    }
    if vectors is not None and delta["encoder"] != 0:
        raise ValueError("冻结编码器发生更新")
    if vectors is None and delta["encoder"] == 0:
        raise ValueError("训练编码器没有更新")
    if delta["head"] == 0:
        raise ValueError("分类头没有更新")
    return history, {
        "seconds": perf_counter() - before,
        "encoder_max_abs_change": delta["encoder"],
        "head_max_abs_change": delta["head"],
        "epochs": config["epochs"],
        "optimizer_steps": config["epochs"] * int(np.ceil(len(train) / batch_size)),
        "gradient_clip_steps": clipped_steps,
    }


def score_frame(frame, config):
    """按assay独立计分，不能把跨平台数值合并排序。"""
    rows = []
    for (method, seed, rate, document, assay), group in frame.groupby(
        ["method", "seed", "learning_rate", "document_id", "assay_id"], sort=True
    ):
        labels = group["label"].to_numpy(dtype=int)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "learning_rate": rate,
                "document_id": document,
                "assay_id": assay,
                "primary_eligible": len(group) >= config["minimum_primary_assay_units"]
                and len(set(labels)) == 2,
                **_score_assay(
                    labels, group["score"].to_numpy(), config["primary_budget_fraction"]
                ),
            }
        )
    return pd.DataFrame(rows)


def contrasts(metrics):
    """每来源、种子、步长先配对，再做描述性汇总。"""
    primary = metrics[metrics["primary_eligible"]]
    pivot = primary.pivot(
        index=["document_id", "assay_id", "seed", "learning_rate"],
        columns="method",
        values="enrichment",
    )
    if pivot[list(ARMS)].isna().any().any():
        raise ValueError("四组未齐，不能输出完整因素差")
    result = pivot.copy()
    result["pretraining_when_frozen"] = pivot.pretrained_frozen - pivot.random_frozen
    result["pretraining_when_trained"] = pivot.pretrained_trained - pivot.random_trained
    result["training_from_pretrained"] = pivot.pretrained_trained - pivot.pretrained_frozen
    result["training_from_random"] = pivot.random_trained - pivot.random_frozen
    result["interaction"] = result.training_from_pretrained - result.training_from_random
    return result.reset_index()
