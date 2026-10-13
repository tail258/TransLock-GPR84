"""预算并列、连接图泄漏与加权总量的关键行为检查。"""

import numpy as np
import pandas as pd
import pytest

from translock_screening.benchmark import (
    _features,
    expected_top_k_hits,
    run_source_benchmark,
    source_split,
    training_weights,
)


def test_tied_boundary_does_not_choose_known_active():
    assert expected_top_k_hits(np.array([0, 1, 0]), np.array([3, 2, 2]), 2) == 0.5
    assert expected_top_k_hits(np.array([1, 0, 1]), np.zeros(3), 1) == pytest.approx(2 / 3)


def test_source_split_excludes_other_source_stereoisomer_connection():
    units = pd.DataFrame(
        {"document_id": ["a", "b", "b"], "connection_id": ["shared", "shared", "new"]}
    )
    train, test = source_split(units, "a")
    assert train.tolist() == [2]
    assert test.tolist() == [0]


def test_assay_balance_preserves_regularization_scale():
    units = pd.DataFrame({"assay_id": ["a", "a", "b"], "smiles": ["C", "CC", "CCC"]})
    weights = training_weights(units, True)
    assert weights.sum() == pytest.approx(3)
    assert weights[:2].sum() == pytest.approx(weights[2])


def test_repeated_graphs_across_assays_do_not_dominate_unbalanced_fit():
    units = pd.DataFrame({"assay_id": ["a", "b", "b"], "smiles": ["C", "C", "CC"]})
    weights = training_weights(units, False)
    assert weights[:2].sum() == pytest.approx(weights[2])


def test_explicit_stereochemistry_is_preserved_when_enabled():
    units = pd.DataFrame({"smiles": ["C[C@H](O)F", "C[C@@H](O)F"]})
    _, features = _features(units, 3, True)
    assert features[0] != features[1]


def test_empty_input_has_status_instead_of_fitted_result():
    summary, predictions, metrics = run_source_benchmark(pd.DataFrame(), {})
    assert summary["status"] == "no_eligible_units"
    assert predictions.empty and metrics.empty


def test_real_small_source_holdout_runs_with_sparse_integer_indices():
    rows = []
    for doc, samples in [
        ("a", [("CCO", 1), ("CCN", 0), ("CCCO", 1), ("CCCN", 0)]),
        ("b", [("c1ccccc1O", 1), ("c1ccccc1N", 0), ("Cc1ccccc1O", 1), ("Cc1ccccc1N", 0)]),
    ]:
        for smiles, label in samples:
            rows.append(
                {
                    "document_id": doc,
                    "assay_id": doc,
                    "unit_id": smiles,
                    "smiles": smiles,
                    "connection_id": smiles,
                    "label": label,
                }
            )
    summary, predictions, metrics = run_source_benchmark(
        pd.DataFrame(rows),
        {
            "radius": 3,
            "include_chirality": True,
            "logistic_c": 1.0,
            "max_iter": 3000,
            "seed": 1,
            "minimum_primary_assay_units": 2,
            "primary_budget_fraction": 0.5,
            "budget_fractions": [0.5],
        },
    )
    assert summary["primary_assay_count"] == 2
    assert len(predictions) == 32
    assert metrics[metrics["method"].eq("random_expectation")]["average_precision"].isna().all()
