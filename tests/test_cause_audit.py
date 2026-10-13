"""保证身份、删失和重复记录不制造桥接证据。"""

import math

import pandas as pd
import pytest

from translock_screening.cause_audit import build_cause_audit, exact_mean_sem, exact_single_value
from translock_screening.measurements import assay_graph_units


def record(activity, assay, document, smiles="Cc1ccccc1", value=10, relation="="):
    return {
        "activity_id": activity,
        "assay_chembl_id": assay,
        "document_chembl_id": document,
        "standardized_smiles": smiles,
        "standard_relation": relation,
        "standard_value": value,
        "document_doi": document,
        "molecule_chembl_id": "mol",
        "connection_id": smiles,
        "document_year": 2020,
        "label": int(value <= 1000),
    }


def audit(rows):
    measurements = pd.DataFrame(rows)
    units, _ = assay_graph_units(measurements)
    return build_cause_audit(measurements, units)


def test_stereoisomers_are_not_cross_assay_bridges():
    tables = audit([record(1, "a", "d1", "C[C@H](O)F"), record(2, "b", "d2", "C[C@@H](O)F")])
    assert tables["bridge_candidates"].empty


def test_bridge_preserves_duplicates_and_does_not_average_or_count_twice():
    tables = audit(
        [record(1, "a", "d1"), record(2, "a", "d1", value=20), record(3, "b", "d2", value=100)]
    )
    assert len(tables["bridge_candidates"]) == 1
    assert tables["assay_edges"].iloc[0]["shared_graphs"] == 1
    assert pd.isna(tables["bridge_candidates"].iloc[0]["log10_ic50_right_over_left"])
    assert tables["bridge_candidates"].iloc[0]["activity_ids_left"] == "1;2"


def test_censored_values_never_become_exact_differences():
    tables = audit([record(1, "a", "d1", relation=">"), record(2, "b", "d2", value=100)])
    assert pd.isna(tables["bridge_candidates"].iloc[0]["log10_ic50_right_over_left"])
    assert exact_single_value([record(1, "a", "d", value=math.inf)]) is None


def test_exact_bridge_difference_has_explicit_direction():
    tables = audit([record(1, "a", "d1", value=10), record(2, "b", "d2", value=100)])
    assert tables["bridge_candidates"].iloc[0]["log10_ic50_right_over_left"] == 1


def test_shared_scaffold_does_not_merge_distinct_units_or_empty_scaffolds():
    tables = audit(
        [
            record(1, "a", "d", "Cc1ccccc1"),
            record(2, "a", "d", "CCc1ccccc1", value=2000),
            record(3, "a", "d", "CCC"),
            record(4, "a", "d", "CCCC"),
        ]
    )
    groups = tables["scaffold_groups"].set_index("scaffold")
    assert groups.loc["c1ccccc1", "strong_units"] == 1
    assert groups.loc["c1ccccc1", "weak_units"] == 1
    assert not groups.loc["", "usable_as_scaffold_group"]
    assert len(tables["scaffold_members"]) == 4


def test_duplicate_activity_id_is_rejected():
    with pytest.raises(ValueError, match="activity_id重复"):
        audit([record(1, "a", "d"), record(1, "b", "d")])


@pytest.mark.parametrize("text", ["<5", ">8 ± 0.1", "inactive", "", "6.0"])
def test_author_censored_or_missing_sem_is_not_silently_a_point(text):
    assert exact_mean_sem(text) is None


def test_author_precision_and_sem_are_preserved():
    assert exact_mean_sem(" 6.97 ± 0.11 ") == (6.97, 0.11)
