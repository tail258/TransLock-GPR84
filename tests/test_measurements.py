"""检验删失边界、身份与冲突处理，避免重构重新制造标签。"""

import pandas as pd
import pytest

from translock_screening.measurements import (
    assay_graph_units,
    connection_smiles,
    threshold_label,
)


@pytest.mark.parametrize(
    "value,relation,expected",
    [
        (1000, "=", 1),
        (1000, ">", 0),
        (1000, ">=", None),
        (1001, ">=", 0),
        (1000, "<=", 1),
        (1001, "<", None),
        (1, ">", None),
        (0, "=", None),
        (float("nan"), "=", None),
    ],
)
def test_censored_threshold_boundaries(value, relation, expected):
    assert threshold_label(value, relation, 1000) == expected


def test_stereoisomers_share_leakage_group_but_not_input_identity():
    assert connection_smiles("C[C@H](O)F") == connection_smiles("C[C@@H](O)F")


def test_conflicting_labels_are_not_median_aggregated():
    rows = []
    for activity, label in [(1, 1), (2, 0)]:
        rows.append(
            {
                "document_chembl_id": "doc",
                "assay_chembl_id": "assay",
                "standardized_smiles": "CCO",
                "connection_id": "CCO",
                "activity_id": activity,
                "molecule_chembl_id": "mol",
                "standard_relation": "=",
                "document_year": 2024,
                "document_doi": "doi",
                "label": label,
            }
        )
    units, conflicts = assay_graph_units(pd.DataFrame(rows))
    assert units.empty
    assert conflicts.iloc[0]["activity_ids"] == "1;2"


def test_same_label_repeats_remain_linked_but_not_recounted():
    row = {
        "document_chembl_id": "doc",
        "assay_chembl_id": "assay",
        "standardized_smiles": "CCO",
        "connection_id": "CCO",
        "molecule_chembl_id": "mol",
        "standard_relation": "=",
        "document_year": 2024,
        "document_doi": "doi",
        "label": 1,
    }
    units, conflicts = assay_graph_units(
        pd.DataFrame([{**row, "activity_id": 1}, {**row, "activity_id": 2}])
    )
    assert len(units) == 1
    assert units.iloc[0]["measurement_count"] == 2
    assert conflicts.empty


def test_no_admissions_returns_explicit_empty_schema():
    units, conflicts = assay_graph_units(pd.DataFrame())
    assert units.empty and "label" in units.columns
    assert conflicts.empty and "reason" in conflicts.columns
