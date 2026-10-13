"""因子对照的分割、防泄漏、稳定损失与配对差检查。"""

import numpy as np
import pandas as pd
import pytest

from translock_screening.training_factor import (
    ARMS,
    contrasts,
    epoch_orders,
    loss_value,
    make_splits,
)


def test_splits_preserve_all_roles_and_exclude_shared_connections():
    units = pd.DataFrame(
        {
            "unit_id": list("abcdef"),
            "document_id": ["A", "A", "B", "B", "C", "C"],
            "connection_id": ["same", "a", "same", "b", "c", "d"],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )
    splits, members = make_splits(units)
    for document, (train, test) in splits.items():
        assert not set(units.iloc[train].connection_id) & set(units.iloc[test].connection_id)
        assert len(members[members.outer_document.eq(document)]) == 6
    assert splits["A"][0].tolist() == [3, 4, 5]


def test_orders_are_paired_independent_of_global_random_state_and_exhaustive():
    first = epoch_orders(19, 24, 3, 2000)
    np.random.seed(1)
    np.random.random(500)
    second = epoch_orders(19, 24, 3, 2000)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
        assert sorted(a.tolist()) == list(range(19))
    assert not np.array_equal(first[0], epoch_orders(19, 25, 3, 2000)[0])


def test_logit_bce_stays_finite_without_probability_clipping():
    y = np.array([1, 0, 1, 0])
    s = np.array([-1000.0, 1000.0, 1000.0, -1000.0])
    assert loss_value(y, s, np.ones(4)) == pytest.approx(500.0)


def test_factor_contrasts_pair_by_source_seed_and_rate_before_average():
    rows = []
    for source, offset in [("A", 0), ("B", 1)]:
        for arm, value in zip(ARMS, [0.8, 1.0, 0.9, 1.3], strict=True):
            rows.append(
                {
                    "document_id": source,
                    "assay_id": source,
                    "seed": 1,
                    "learning_rate": 0.001,
                    "method": arm,
                    "enrichment": value + offset,
                    "primary_eligible": True,
                }
            )
    frame = pd.DataFrame(rows)
    result = contrasts(frame)
    np.testing.assert_allclose(result.pretraining_when_frozen, 0.2)
    np.testing.assert_allclose(result.pretraining_when_trained, 0.4)
    np.testing.assert_allclose(result.training_from_pretrained, 0.3)
    np.testing.assert_allclose(result.training_from_random, 0.1)
    np.testing.assert_allclose(result.interaction, 0.2)
    with pytest.raises(ValueError, match="四组未齐"):
        contrasts(frame.iloc[:-1])
