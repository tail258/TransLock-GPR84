"""固定官方消息传递权重的CPU表示；不训练编码器，不使用随机预测头。"""

from pathlib import Path

import numpy as np
from rdkit import Chem


class FrozenCheMeleon:
    """沿用官方指纹接口的消息传递与平均聚合，并显式限制设备和权重位置。"""

    def __init__(self, checkpoint: Path, config: dict):
        import torch
        from chemprop import featurizers, nn

        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.batch_size = config["batch_size"]
        self.dimensions = config["output_dimensions"]
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        if tuple(self.featurizer.shape) != (72, 14):
            raise ValueError("Chemprop默认原子/键特征与冻结定义不一致")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.hyper_parameters = payload["hyper_parameters"]
        self.message_passing = nn.BondMessagePassing(**self.hyper_parameters)
        self.message_passing.load_state_dict(payload["state_dict"], strict=True)
        self.message_passing.cpu()
        self.message_passing.requires_grad_(False)
        self.message_passing.eval()
        self.aggregation = nn.MeanAggregation()
        self.aggregation.cpu()
        self.aggregation.eval()
        if any(parameter.device.type != "cpu" for parameter in self.message_passing.parameters()):
            raise ValueError("编码器未处于冻结的CPU设备")
        if self.message_passing.output_dim != self.dimensions:
            raise ValueError("权重输出维度与冻结配置不符")

    def graph(self, smiles):
        from chemprop.data import BatchMolGraph

        molecules = [Chem.MolFromSmiles(value) for value in smiles]
        if any(molecule is None for molecule in molecules):
            raise ValueError("固定输入包含不可解析结构")
        graphs = BatchMolGraph([self.featurizer(molecule) for molecule in molecules])
        graphs.to("cpu")
        return graphs

    def encode(self, smiles):
        if not smiles:
            return np.empty((0, self.dimensions), dtype=np.float32)
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(smiles), self.batch_size):
                graph = self.graph(smiles[start : start + self.batch_size])
                hidden = self.message_passing(graph)
                batches.append(self.aggregation(hidden, graph.batch).cpu().numpy())
        result = np.concatenate(batches)
        if result.shape != (len(smiles), self.dimensions) or not np.isfinite(result).all():
            raise ValueError("预训练表示维度/有限性检查失败")
        return result

    def official_fingerprint(self, smiles):
        """只用于验证与官方MPNN.fingerprint默认路径等价；FFN从不参与表示。"""
        from chemprop.models import MPNN
        from chemprop.nn import RegressionFFN

        model = MPNN(
            message_passing=self.message_passing,
            agg=self.aggregation,
            predictor=RegressionFFN(input_dim=self.dimensions),
        )
        model.eval()
        with self.torch.inference_mode():
            return model.fingerprint(self.graph(smiles)).cpu().numpy()

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.message_passing.parameters())
