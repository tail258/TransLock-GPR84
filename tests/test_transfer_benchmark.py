"""冻结迁移评价的小样、训练侧拟合和连接图留出核查。"""

import numpy as np
import pandas as pd

from translock_screening.benchmark import _features, source_split, training_weights
from translock_screening.measurements import connection_smiles
from translock_screening.transfer_benchmark import run_transfer_benchmark


def fixture_units():
    structures = {
        "A": [("C[C@H](F)O", 1), ("CCO", 0), ("CCN", 1), ("c1ccccc1Cl", 0)],
        "B": [("C[C@@H](F)O", 1), ("CCC", 0), ("CCBr", 1), ("COC", 0)],
        "C": [("CC(=O)O", 1), ("CCO", 0), ("CN", 1), ("ClCCl", 0)],
    }
    rows = []
    for document, examples in structures.items():
        for index, (smiles, label) in enumerate(examples):
            rows.append(
                {
                    "document_id": document,
                    "assay_id": f"assay_{document}",
                    "unit_id": f"{document}_{index}",
                    "connection_id": connection_smiles(smiles),
                    "smiles": smiles,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def fixture_config():
    return {
        "radius": 3,
        "include_chirality": True,
        "logistic_c": 1.0,
        "max_iter": 3000,
        "seed": 20261001,
        "minimum_primary_assay_units": 2,
        "primary_budget_fraction": 0.2,
        "budget_fractions": [0.2, 0.5],
    }


def test_real_lr_keeps_scaler_vocabulary_and_members_inside_training_fold():
    units = fixture_units()
    embeddings = np.arange(len(units) * 3, dtype=float).reshape(len(units), 3) / 10
    embeddings[0, 0] = 1_000_000.0  # A 来源的极端测试值不能影响 A 留出折缩放。
    summary, predictions, metrics, members, models = run_transfer_benchmark(
        units, embeddings, fixture_config()
    )

    assert summary["status"] == "exploratory_retrospective_source_holdout_not_blind_test"
    assert summary["primary_assay_count"] == 3
    assert len(predictions) == len(units) * 2
    assert set(predictions["method"]) == {"chemeleon_scaled_lr", "r3_scaled_lr"}
    assert len(metrics) == 3 * 2 * 2
    assert set(models) == {"A", "B", "C"}
    assert len(members) == len(units) * 3

    for document in models:
        fit_idx, held_idx = source_split(units, document)
        fold_members = members[members["document_id"].eq(document)]
        roles = fold_members.set_index("unit_id")["role"].to_dict()
        fit_ids = set(units.iloc[fit_idx]["unit_id"])
        held_ids = set(units.iloc[held_idx]["unit_id"])
        assert {key for key, role in roles.items() if role == "fit"} == fit_ids
        assert {key for key, role in roles.items() if role == "held_out"} == held_ids
        assert not set(units.iloc[fit_idx]["connection_id"]) & set(
            units.iloc[held_idx]["connection_id"]
        )
        for method in ("chemeleon_scaled_lr", "r3_scaled_lr"):
            fitted = models[document][method]
            assert set(fitted["fit_unit_ids"]) == fit_ids
            assert fitted["model"].n_iter_[0] < fixture_config()["max_iter"]
            assert fitted["scaler"].with_mean is False
        assert len(fold_members) == len(units)

    train_idx, test_idx = source_split(units, "A")
    weights = training_weights(units.iloc[train_idx].reset_index(drop=True), False)
    expected_mean = np.average(embeddings[train_idx], axis=0, weights=weights)
    np.testing.assert_allclose(models["A"]["chemeleon_scaled_lr"]["scaler"].mean_, expected_mean)
    assert expected_mean[0] < 100
    assert units.iloc[0]["unit_id"] in set(units.iloc[test_idx]["unit_id"])
    b_train, _ = source_split(units, "B")
    b_weights = training_weights(units.iloc[b_train].reset_index(drop=True), False)
    assert not np.allclose(b_weights, np.ones(len(b_weights)))
    np.testing.assert_allclose(
        models["B"]["chemeleon_scaled_lr"]["scaler"].mean_,
        np.average(embeddings[b_train], axis=0, weights=b_weights),
    )

    _, feature_dicts = _features(units, radius=3, include_chirality=True)
    train_features = set().union(*(feature_dicts[i] for i in train_idx))
    test_features = set().union(*(feature_dicts[i] for i in test_idx))
    unique_test_features = test_features - train_features
    assert unique_test_features
    vocabulary = models["A"]["r3_scaled_lr"]["vectorizer"].vocabulary_
    assert not unique_test_features & set(vocabulary)


def test_empty_and_single_class_folds_are_explicit():
    empty = pd.DataFrame()
    summary, predictions, metrics, members, models = run_transfer_benchmark(
        empty, np.empty((0, 3)), fixture_config()
    )
    assert summary["status"] == "no_eligible_units"
    assert predictions.empty and metrics.empty and members.empty and not models

    units = fixture_units().iloc[:8].copy()
    units["label"] = 1
    embeddings = np.ones((len(units), 3))
    summary, predictions, metrics, members, models = run_transfer_benchmark(
        units, embeddings, fixture_config()
    )
    assert summary["status"] == "no_trainable_folds"
    assert len(summary["folds"]) == 2
    assert all(fold["status"] == "training_single_class" for fold in summary["folds"])
    assert predictions.empty and metrics.empty and not models
    assert len(members) == len(units) * 2
