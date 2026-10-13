"""按assay评估的固定来源留出检索。"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_top_k_hits(y: np.ndarray, scores: np.ndarray, budget: int) -> float:
    """边界并列采用均匀抽取期望，不按真实标签挑选并列成员。"""
    if budget < 1 or budget > len(y) or len(y) != len(scores):
        raise ValueError("候选预算或输入维度不合法")
    rounded = np.round(np.asarray(scores, dtype=float), 12)
    if not np.isfinite(rounded).all():
        raise ValueError("分数必须有限")
    boundary = np.sort(rounded)[-budget]
    above = rounded > boundary
    tied = rounded == boundary
    remainder = budget - int(above.sum())
    return float(y[above].sum() + remainder * y[tied].mean())


def source_split(units: pd.DataFrame, document: str) -> tuple[np.ndarray, np.ndarray]:
    """按整篇来源留出，再移除训练侧的所有测试连接图。"""
    test = units["document_id"].eq(document).to_numpy()
    held_connections = set(units.loc[test, "connection_id"])
    train = (~test) & ~units["connection_id"].isin(held_connections).to_numpy()
    return np.flatnonzero(train), np.flatnonzero(test)


def training_weights(units: pd.DataFrame, assay_balanced: bool) -> np.ndarray:
    """先按图重复次数降权，再可选assay归一；后者不同时保证图总权重相同。"""
    occurrences = units.groupby("smiles")["smiles"].transform("size").to_numpy()
    weights = 1.0 / occurrences
    if assay_balanced:
        totals = pd.Series(weights).groupby(units["assay_id"].to_numpy()).transform("sum")
        weights = weights / totals.to_numpy()
    return weights * len(weights) / weights.sum()


def _features(units: pd.DataFrame, radius: int, include_chirality: bool):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, includeChirality=include_chirality
    )
    cache = {}
    for smiles in units["smiles"].unique():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("准入后出现不可解析结构")
        cache[smiles] = generator.GetSparseCountFingerprint(mol)
    fps = [cache[s] for s in units["smiles"]]
    dictionaries = [
        {str(key): float(np.log1p(count)) for key, count in fp.GetNonzeroElements().items()}
        for fp in fps
    ]
    return fps, dictionaries


def _score_assay(y, scores, fraction):
    n = len(y)
    budget = max(1, math.ceil(n * fraction))
    positives = int(y.sum())
    hits = expected_top_k_hits(y, scores, budget)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": n,
        "positives": positives,
        "weak": n - positives,
        "budget": budget,
        "expected_hits_with_ties": hits,
        "random_expected_hits": budget * positives / n,
        "precision": hits / budget,
        "recall": hits / positives if positives else None,
        "enrichment": hits / (budget * positives / n) if positives else None,
        "roc_auc": float(roc_auc_score(y, scores)) if two_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if positives else None,
    }


def run_source_benchmark(units: pd.DataFrame, config: dict[str, Any]):
    """所有固定方法在同一来源留出运行一次，不按外层结果选择并调参。"""
    start = perf_counter()
    if units.empty:
        return {"status": "no_eligible_units"}, pd.DataFrame(), pd.DataFrame()
    fps, dictionaries = _features(units, config["radius"], config["include_chirality"])
    feature_seconds = perf_counter() - start
    predictions, metrics, fold_records = [], [], []
    labels = units["label"].to_numpy(dtype=int)
    for document in sorted(units["document_id"].unique()):
        train, test = source_split(units, document)
        train_frame = units.iloc[train].reset_index(drop=True)
        fold = {
            "document_id": document,
            "train_units": len(train),
            "test_units": len(test),
            "excluded_train_connection_overlap": int(
                units["document_id"].ne(document).sum() - len(train)
            ),
        }
        if len(np.unique(labels[train])) != 2:
            fold_records.append({**fold, "status": "training_single_class"})
            continue
        vectorizer = DictVectorizer(dtype=np.float64, sparse=True)
        x_train = vectorizer.fit_transform([dictionaries[i] for i in train])
        x_test = vectorizer.transform([dictionaries[i] for i in test])
        # 新版SciPy产生int64索引，liblinear要求int32；仅转换索引类型，不改特征。
        for matrix in (x_train, x_test):
            if max(matrix.shape) >= np.iinfo(np.int32).max or matrix.nnz >= np.iinfo(np.int32).max:
                raise ValueError("稀疏矩阵超出当前liblinear索引容量")
            matrix.indices = matrix.indices.astype(np.int32)
            matrix.indptr = matrix.indptr.astype(np.int32)
        fold_scores = {}
        fit_times = {}
        for name, balanced in [("lr_graph_weighted", False), ("lr_assay_balanced", True)]:
            before = perf_counter()
            model = LogisticRegression(
                C=config["logistic_c"],
                solver="liblinear",
                max_iter=config["max_iter"],
                random_state=config["seed"],
                class_weight=None,
            )
            model.fit(x_train, labels[train], sample_weight=training_weights(train_frame, balanced))
            fold_scores[name] = model.predict_proba(x_test)[:, 1]
            fit_times[name] = perf_counter() - before
            if np.any(model.n_iter_ >= config["max_iter"]):
                raise RuntimeError("固定迭代预算未收敛；不得默默使用失败模型")
        before = perf_counter()
        positive_fps = [fps[i] for i in train if labels[i] == 1]
        fold_scores["nearest_measured_strong"] = np.array(
            [max(DataStructs.BulkTanimotoSimilarity(fps[i], positive_fps)) for i in test]
        )
        fit_times["nearest_measured_strong"] = perf_counter() - before
        fold_scores["random_expectation"] = np.zeros(len(test))
        nearest = np.array(
            [max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in train])) for i in test]
        )
        for name, scores in fold_scores.items():
            for position, index in enumerate(test):
                row = units.iloc[index]
                predictions.append(
                    {
                        "unit_id": row["unit_id"],
                        "document_id": document,
                        "assay_id": row["assay_id"],
                        "smiles": row["smiles"],
                        "label": int(labels[index]),
                        "method": name,
                        "score": float(scores[position]),
                        "nearest_train_tanimoto": float(nearest[position]),
                    }
                )
            test_frame = units.iloc[test]
            for assay in sorted(test_frame["assay_id"].unique()):
                within = test_frame["assay_id"].eq(assay).to_numpy()
                y = labels[test][within]
                primary = len(y) >= config["minimum_primary_assay_units"] and len(set(y)) == 2
                for fraction in config["budget_fractions"]:
                    values = _score_assay(y, scores[within], fraction)
                    if name == "random_expectation":
                        values["average_precision"] = None
                    metrics.append(
                        {
                            "document_id": document,
                            "assay_id": assay,
                            "method": name,
                            "budget_fraction": fraction,
                            "primary_eligible": primary,
                            **values,
                        }
                    )
        fold_records.append(
            {
                **fold,
                "status": "complete",
                "model_seconds": fit_times,
                "training_feature_columns": x_train.shape[1],
                "nearest_train_tanimoto_median": float(np.median(nearest)),
            }
        )
    metrics_frame = pd.DataFrame(metrics)
    if metrics_frame.empty:
        return (
            {"status": "no_trainable_folds", "folds": fold_records},
            pd.DataFrame(),
            metrics_frame,
        )
    primary = metrics_frame[
        metrics_frame["primary_eligible"]
        & metrics_frame["budget_fraction"].eq(config["primary_budget_fraction"])
    ]
    macro = (
        primary.groupby("method")[
            ["precision", "recall", "enrichment", "roc_auc", "average_precision"]
        ]
        .mean()
        .reset_index()
    )
    macro = macro.astype(object).where(pd.notna(macro), None)
    summary = {
        "status": "exploratory_retrospective_source_holdout_not_blind_test",
        "feature_seconds": feature_seconds,
        "total_seconds": perf_counter() - start,
        "folds": fold_records,
        "primary_assay_macro": macro.to_dict("records"),
        "primary_assay_count": int(primary["assay_id"].nunique()),
        "primary_document_count": int(primary["document_id"].nunique()),
        "scope": "assay_graph_units_known_antagonist_context_potency_not_function_direction",
        "no_model_chosen_from_outer_test": True,
        "uncertainty": "每来源逐项公开；来源少且化学系列相关，不作显著性或全库召回保证",
    }
    return summary, pd.DataFrame(predictions), metrics_frame
