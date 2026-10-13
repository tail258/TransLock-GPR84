"""冻结表征与同尺度 r3 对照的来源留出回顾评价。"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from translock_screening.benchmark import (
    _features,
    _score_assay,
    source_split,
    training_weights,
)

METHODS = ("chemeleon_scaled_lr", "r3_scaled_lr")
MEMBER_COLUMNS = ("document_id", "role", "unit_id", "assay_id", "connection_id", "label")
PREDICTION_COLUMNS = (
    "unit_id",
    "document_id",
    "assay_id",
    "smiles",
    "label",
    "method",
    "score",
)


def _liblinear_sparse_indices(matrix: Any) -> Any:
    """新版 SciPy 的 int64 稀疏索引需转换为 liblinear 接受的 int32。"""
    if not hasattr(matrix, "indices"):
        return matrix
    if max(matrix.shape) >= np.iinfo(np.int32).max or matrix.nnz >= np.iinfo(np.int32).max:
        raise ValueError("稀疏矩阵超出 liblinear 索引容量")
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix


def _fit_scaled_lr(
    x_train: Any,
    x_test: Any,
    labels: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, StandardScaler, LogisticRegression]:
    # 相同的图降权同时用于缩放和分类头，只在本折训练侧拟合。
    scaler = StandardScaler(with_mean=False)
    scaler.fit(x_train, sample_weight=weights)
    scaled_train = _liblinear_sparse_indices(scaler.transform(x_train))
    scaled_test = _liblinear_sparse_indices(scaler.transform(x_test))
    model = LogisticRegression(
        C=config["logistic_c"],
        solver="liblinear",
        max_iter=config["max_iter"],
        random_state=config["seed"],
        class_weight=None,
    )
    model.fit(scaled_train, labels, sample_weight=weights)
    if np.any(model.n_iter_ >= config["max_iter"]):
        raise RuntimeError("固定迭代预算未收敛；不得使用失败模型")
    scores = model.predict_proba(scaled_test)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("模型给出非有限分数")
    return scores, scaler, model


def _empty_result(status: str):
    return (
        {"status": status, "folds": [], "scaler": {"with_mean": False, "weighted": True}},
        pd.DataFrame(columns=PREDICTION_COLUMNS),
        pd.DataFrame(),
        pd.DataFrame(columns=MEMBER_COLUMNS),
        {},
    )


def run_transfer_benchmark(
    units: pd.DataFrame,
    embeddings: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """同一冻结折上比较预训练表示与缩放后的原 r3 logcount 表示。"""
    if units.empty:
        return _empty_result("no_eligible_units")
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(units) or vectors.shape[1] < 1:
        raise ValueError("表征矩阵必须按单位行对齐，且有至少一列")
    if not np.isfinite(vectors).all():
        raise ValueError("表征矩阵包含非有限数")
    labels = units["label"].to_numpy(dtype=int)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("单位标签必须为 0/1")
    if units["unit_id"].duplicated().any():
        raise ValueError("unit_id 不能重复")

    started = perf_counter()
    _, r3_dictionaries = _features(units, config["radius"], config["include_chirality"])
    feature_seconds = perf_counter() - started
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    saved_models: dict[str, dict[str, dict[str, Any]]] = {}

    for document in sorted(units["document_id"].unique()):
        train_idx, test_idx = source_split(units, document)
        test_connections = set(units.iloc[test_idx]["connection_id"])
        excluded_idx = np.flatnonzero(
            units["document_id"].ne(document).to_numpy()
            & units["connection_id"].isin(test_connections).to_numpy()
        )
        for role, indices in (
            ("fit", train_idx),
            ("held_out", test_idx),
            ("excluded_overlap", excluded_idx),
        ):
            for index in indices:
                row = units.iloc[index]
                members.append(
                    {
                        "document_id": document,
                        "role": role,
                        "unit_id": row["unit_id"],
                        "assay_id": row["assay_id"],
                        "connection_id": row["connection_id"],
                        "label": int(row["label"]),
                    }
                )
        fold: dict[str, Any] = {
            "document_id": document,
            "train_units": len(train_idx),
            "test_units": len(test_idx),
            "excluded_train_connection_overlap": len(excluded_idx),
            "method_records": {},
        }
        if len(np.unique(labels[train_idx])) != 2:
            folds.append({**fold, "status": "training_single_class"})
            continue
        train = units.iloc[train_idx].reset_index(drop=True)
        weights = training_weights(train, False)
        vectorizer = DictVectorizer(dtype=np.float64, sparse=True)
        r3_train = vectorizer.fit_transform([r3_dictionaries[i] for i in train_idx])
        r3_test = vectorizer.transform([r3_dictionaries[i] for i in test_idx])
        datasets = {
            "chemeleon_scaled_lr": (vectors[train_idx], vectors[test_idx], None),
            "r3_scaled_lr": (r3_train, r3_test, vectorizer),
        }
        saved_models[document] = {}
        test_frame = units.iloc[test_idx]
        for method, (x_train, x_test, fitted_vectorizer) in datasets.items():
            before = perf_counter()
            scores, scaler, model = _fit_scaled_lr(
                x_train, x_test, labels[train_idx], weights, config
            )
            fold["method_records"][method] = {
                "fit_seconds": perf_counter() - before,
                "feature_columns": x_train.shape[1],
                "zero_variance_columns": int(np.count_nonzero(scaler.var_ == 0)),
                "iterations": int(model.n_iter_[0]),
            }
            saved_models[document][method] = {
                "scaler": scaler,
                "vectorizer": fitted_vectorizer,
                "model": model,
                "fit_unit_ids": train["unit_id"].tolist(),
            }
            for position, index in enumerate(test_idx):
                row = units.iloc[index]
                predictions.append(
                    {
                        "unit_id": row["unit_id"],
                        "document_id": document,
                        "assay_id": row["assay_id"],
                        "smiles": row["smiles"],
                        "label": int(labels[index]),
                        "method": method,
                        "score": float(scores[position]),
                    }
                )
            for assay in sorted(test_frame["assay_id"].unique()):
                within = test_frame["assay_id"].eq(assay).to_numpy()
                assay_labels = labels[test_idx][within]
                primary = (
                    len(assay_labels) >= config["minimum_primary_assay_units"]
                    and len(set(assay_labels)) == 2
                )
                for fraction in config["budget_fractions"]:
                    metrics.append(
                        {
                            "document_id": document,
                            "assay_id": assay,
                            "method": method,
                            "budget_fraction": fraction,
                            "primary_eligible": primary,
                            **_score_assay(assay_labels, scores[within], fraction),
                        }
                    )
        folds.append({**fold, "status": "complete"})

    metric_frame = pd.DataFrame(metrics)
    member_frame = pd.DataFrame(members, columns=MEMBER_COLUMNS)
    prediction_frame = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    if metric_frame.empty:
        return (
            {
                "status": "no_trainable_folds",
                "folds": folds,
                "feature_seconds": feature_seconds,
                "scaler": {"with_mean": False, "weighted": True, "zero_variance": "sklearn_safe"},
            },
            prediction_frame,
            metric_frame,
            member_frame,
            saved_models,
        )
    primary = metric_frame[
        metric_frame["primary_eligible"]
        & metric_frame["budget_fraction"].eq(config["primary_budget_fraction"])
    ]
    macro_columns = ["precision", "recall", "enrichment", "roc_auc", "average_precision"]
    macro = primary.groupby("method")[macro_columns].mean().reset_index()
    macro = macro.astype(object).where(pd.notna(macro), None)
    summary = {
        "status": "exploratory_retrospective_source_holdout_not_blind_test",
        "methods": list(METHODS),
        "feature_seconds": feature_seconds,
        "total_seconds": perf_counter() - started,
        "folds": folds,
        "scaler": {
            "with_mean": False,
            "weighted": True,
            "zero_variance": "sklearn_safe",
            "fit_on": "source_fold_training_units_only",
        },
        "primary_assay_macro": macro.to_dict("records"),
        "primary_assay_count": int(primary["assay_id"].nunique()),
        "primary_document_count": int(primary["document_id"].nunique()),
        "no_model_chosen_from_outer_test": True,
    }
    return summary, prediction_frame, metric_frame, member_frame, saved_models
