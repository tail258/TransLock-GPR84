"""基于 ECFP 的 GPR84 简单回归基线。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


def _murcko_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"无法解析建模 SMILES：{smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return scaffold or f"ACYCLIC::{smiles}"


def build_first_observation_dataset(measurements: pd.DataFrame) -> pd.DataFrame:
    """每个结构仅使用最早公开年份的测量，避免未来标签进入早期数据。"""
    required = {
        "standardized_inchikey",
        "standardized_smiles",
        "document_year",
        "document_chembl_id",
        "assay_chembl_id",
        "standard_value",
    }
    missing = required - set(measurements.columns)
    if missing:
        raise ValueError(f"建模输入缺少字段：{sorted(missing)}")

    first_year = measurements.groupby("standardized_inchikey")["document_year"].transform(
        "min"
    )
    earliest = measurements.loc[measurements["document_year"] == first_year].copy()
    earliest["pIC50_measurement"] = 9.0 - np.log10(earliest["standard_value"].astype(float))

    records: list[dict[str, Any]] = []
    for inchikey, group in earliest.groupby("standardized_inchikey", sort=True):
        smiles_values = sorted(set(group["standardized_smiles"]))
        if len(smiles_values) != 1:
            raise ValueError(f"同一 InChIKey 对应多个标准化结构：{inchikey}")
        document_ids = sorted(set(group["document_chembl_id"]))
        assay_ids = sorted(set(group["assay_chembl_id"]))
        records.append(
            {
                "standardized_inchikey": inchikey,
                "standardized_smiles": smiles_values[0],
                "first_year": int(group["document_year"].iloc[0]),
                "document_group": ";".join(document_ids),
                "assay_ids": ";".join(assay_ids),
                "measurement_count": len(group),
                "median_ic50_nm": float(group["standard_value"].median()),
                "pIC50": float(group["pIC50_measurement"].median()),
                "pIC50_min": float(group["pIC50_measurement"].min()),
                "pIC50_max": float(group["pIC50_measurement"].max()),
                "murcko_scaffold": _murcko_scaffold(smiles_values[0]),
            }
        )
    return pd.DataFrame(records)


def make_split_indices(
    dataset: pd.DataFrame, config: dict[str, Any], random_seed: int
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """生成时间外、骨架外、文献组和随机对照拆分。"""
    indices = np.arange(len(dataset))
    years = dataset["first_year"].to_numpy()
    time_train = indices[years <= config["time_train_through"]]
    time_test = indices[years > config["time_train_through"]]
    if not len(time_train) or not len(time_test):
        raise ValueError("时间切分的训练集或测试集为空")

    scaffold_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["scaffold_test_fraction"],
        random_state=random_seed,
    )
    scaffold_train, scaffold_test = next(
        scaffold_splitter.split(indices, groups=dataset["murcko_scaffold"])
    )
    random_train, random_test = train_test_split(
        indices,
        test_size=config["random_test_fraction"],
        random_state=random_seed,
    )

    document_groups = dataset["document_group"]
    fold_count = min(config["document_cv_folds"], document_groups.nunique())
    if fold_count < 2:
        raise ValueError("文献组数量不足以进行交叉验证")
    document_cv = list(GroupKFold(n_splits=fold_count).split(indices, groups=document_groups))
    return {
        "time": [(time_train, time_test)],
        "scaffold": [(scaffold_train, scaffold_test)],
        "document_cv": document_cv,
        "random": [(np.asarray(random_train), np.asarray(random_test))],
    }


def build_fingerprints(
    smiles_values: list[str], *, radius: int, size: int
) -> tuple[np.ndarray, list[DataStructs.ExplicitBitVect]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    vectors: list[DataStructs.ExplicitBitVect] = []
    matrix = np.zeros((len(smiles_values), size), dtype=np.float32)
    for index, smiles in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"ECFP 输入无法解析：{smiles}")
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
        vectors.append(fingerprint)
    return matrix, vectors


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    residual = observed - predicted
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator else None
    if np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        spearman = None
    else:
        correlation = spearmanr(observed, predicted).statistic
        spearman = float(correlation) if math.isfinite(correlation) else None
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


def _models(config: dict[str, Any], random_seed: int) -> dict[str, Any]:
    return {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=config["ridge_alpha"]),
        "random_forest": RandomForestRegressor(
            n_estimators=config["random_forest_trees"],
            min_samples_leaf=config["random_forest_min_samples_leaf"],
            max_features=config["random_forest_max_features"],
            n_jobs=config["n_jobs"],
            random_state=random_seed,
        ),
    }


def _nearest_train_similarity(
    fingerprints: list[DataStructs.ExplicitBitVect],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> list[float]:
    train_fingerprints = [fingerprints[index] for index in train_indices]
    return [
        max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fingerprints))
        for index in test_indices
    ]


def run_baselines(
    dataset: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """运行所有预注册拆分和模型，并返回摘要及逐条预测。"""
    random_seed = config["random_seed"]
    matrix, fingerprints = build_fingerprints(
        dataset["standardized_smiles"].tolist(),
        radius=config["ecfp_radius"],
        size=config["ecfp_size"],
    )
    observed = dataset["pIC50"].to_numpy(dtype=float)
    splits = make_split_indices(dataset, config["splits"], random_seed)
    result_splits: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []

    for split_name, folds in splits.items():
        fold_results: list[dict[str, Any]] = []
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            similarities = _nearest_train_similarity(fingerprints, train_indices, test_indices)
            model_results: dict[str, Any] = {}
            for model_name, model in _models(config["models"], random_seed).items():
                model.fit(matrix[train_indices], observed[train_indices])
                predicted = model.predict(matrix[test_indices])
                model_results[model_name] = _metrics(observed[test_indices], predicted)
                for row_index, prediction, similarity in zip(
                    test_indices, predicted, similarities, strict=True
                ):
                    record = dataset.iloc[row_index]
                    prediction_rows.append(
                        {
                            "split": split_name,
                            "fold": fold_index,
                            "model": model_name,
                            "standardized_inchikey": record["standardized_inchikey"],
                            "first_year": int(record["first_year"]),
                            "document_group": record["document_group"],
                            "murcko_scaffold": record["murcko_scaffold"],
                            "observed_pIC50": float(observed[row_index]),
                            "predicted_pIC50": float(prediction),
                            "nearest_train_tanimoto": float(similarity),
                        }
                    )
            train_structures = set(dataset.iloc[train_indices]["standardized_inchikey"])
            test_structures = set(dataset.iloc[test_indices]["standardized_inchikey"])
            train_scaffolds = set(dataset.iloc[train_indices]["murcko_scaffold"])
            test_scaffolds = set(dataset.iloc[test_indices]["murcko_scaffold"])
            fold_results.append(
                {
                    "fold": fold_index,
                    "train_size": len(train_indices),
                    "test_size": len(test_indices),
                    "shared_structures": len(train_structures & test_structures),
                    "shared_scaffolds": len(train_scaffolds & test_scaffolds),
                    "test_nearest_train_tanimoto_median": float(np.median(similarities)),
                    "models": model_results,
                }
            )
        result_splits[split_name] = fold_results

    summary = {
        "schema_version": 1,
        "random_seed": random_seed,
        "dataset": {
            "entities": len(dataset),
            "scaffolds": dataset["murcko_scaffold"].nunique(),
            "documents": dataset["document_group"].nunique(),
            "years": sorted(int(value) for value in dataset["first_year"].unique()),
        },
        "ecfp": {"radius": config["ecfp_radius"], "size": config["ecfp_size"]},
        "splits": result_splits,
        "failure_samples": [],
    }
    summary["gate2"] = evaluate_gate(summary, config["gate2_thresholds"])
    return summary, pd.DataFrame(prediction_rows)


def evaluate_gate(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """用预注册阈值判断时间外和骨架外基线是否提供真实增益。"""
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for split_name in ("time", "scaffold"):
        fold = summary["splits"][split_name][0]
        dummy_mae = fold["models"]["dummy_median"]["mae"]
        candidates = {
            name: metrics
            for name, metrics in fold["models"].items()
            if name != "dummy_median"
        }
        best_name, best = min(candidates.items(), key=lambda item: item[1]["mae"])
        improvement = (dummy_mae - best["mae"]) / dummy_mae if dummy_mae else 0.0
        split_checks = {
            "minimum_test_size": fold["test_size"] >= thresholds["min_test_size"],
            "no_structure_overlap": fold["shared_structures"] == 0,
            "mae_improvement": improvement >= thresholds["min_mae_improvement_fraction"],
            "minimum_spearman": (best["spearman"] or -1.0)
            >= thresholds["min_spearman"],
        }
        if split_name == "scaffold":
            split_checks["no_scaffold_overlap"] = fold["shared_scaffolds"] == 0
        checks[split_name] = all(split_checks.values())
        details[split_name] = {
            "best_model": best_name,
            "dummy_mae": dummy_mae,
            "best_mae": best["mae"],
            "mae_improvement_fraction": improvement,
            "best_spearman": best["spearman"],
            "checks": split_checks,
        }
    return {"passed": all(checks.values()), "split_pass": checks, "details": details}


def write_baseline_outputs(
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: dict[str, Any],
    dataset_path: Path,
    predictions_path: Path,
    summary_path: Path,
) -> None:
    """保存实体表、逐条预测和带配置的评估摘要。"""
    for path in (dataset_path, predictions_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(dataset_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
