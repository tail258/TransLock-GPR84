import numpy as np
import pandas as pd
import pytest

from translock_screening.external_validation import (
    collapse_connections,
    fixed_metrics,
    identity_keys,
    patent_label,
)


def test_identity_retains_samples_but_blocks_stereo_isotope_salt_leakage():
    a = identity_keys("C[C@H](O)F")
    b = identity_keys("C[C@@H](O)F.Cl")
    d = identity_keys("[2H]C([2H])([2H])[C@H](O)F")
    assert a["smiles"] != b["smiles"] != d["smiles"]
    assert a["broad_identity"] == b["broad_identity"] == d["broad_identity"]


def test_interval_labels_and_uninformative_ranking():
    assert [patent_label(x) for x in ["A", "B", "C"]] == [1, 1, 0]
    with pytest.raises(KeyError):
        patent_label("D")
    y = np.array([1] * 53 + [0] * 8)
    result = fixed_metrics(y, np.zeros(61))
    assert result["budget"] == 13
    assert result["ef"] == pytest.approx(1)
    assert result["roc_auc"] == pytest.approx(0.5)
    assert result["ef_ceiling"] == pytest.approx(61 / 53)


def test_collapse_does_not_use_prediction_and_removes_conflicts():
    frame = pd.DataFrame(
        {
            "sample_id": ["I-10", "I-2", "I-3", "I-4"],
            "isotope_free": ["a", "a", "b", "b"],
            "label": [1, 1, 1, 0],
            "score": [100, -100, 100, -100],
        }
    )
    assert collapse_connections(frame).sample_id.tolist() == ["I-2"]
