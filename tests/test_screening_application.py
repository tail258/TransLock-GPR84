"""筛选身份保护、范围限制及不使用其他种子挑名单的行为检验。"""

import numpy as np
import pandas as pd
import pytest

from translock_screening.screening_application import (
    add_rankings,
    connection,
    nearest_support,
    nominate,
    normalize,
    prepare_catalogue,
    validate_source,
    validate_training_scope,
)


def test_identity_scope_and_deduplication_preserve_every_input():
    raw = pd.DataFrame(
        {
            "molecule_chembl_id": [f"ID{i}" for i in range(7)],
            "canonical_smiles": [
                "C[C@@H](O)F",
                "CCO.Cl",
                "OCC",
                None,
                "C[Si](C)(C)C",
                "CCCCCCCC",
                "CCN",
            ],
        }
    )
    scope = {
        "min_heavy_atoms": 2,
        "max_heavy_atoms": 5,
        "allowed_elements": ["C", "N", "O", "F", "Cl"],
    }
    result = prepare_catalogue(raw, {"ID6"}, {connection(normalize("C[C@H](O)F"))}, scope)
    assert result.candidate_id.tolist() == raw.molecule_chembl_id.tolist()
    assert result.status.tolist() == [
        "known_target_excluded",
        "ready_for_scoring",
        "duplicate_excluded",
        "structure_or_scope_excluded",
        "structure_or_scope_excluded",
        "structure_or_scope_excluded",
        "known_target_excluded",
    ]
    assert result.iloc[2].duplicate_of == "ID1"
    assert result.iloc[1].smiles == "CCO"
    assert result.iloc[5].reasons == "outside_training_size_scope"
    with pytest.raises(ValueError, match="ID重复"):
        prepare_catalogue(pd.concat([raw, raw.iloc[:1]]), set(), set(), scope)


def test_primary_ranking_and_nomination_ignore_sensitivity_and_retain_cap():
    frame = pd.DataFrame(
        {
            "candidate_id": ["B", "A", "C", "D"],
            "score_1": [2.0, 2.0, 1.0, 0.0],
            "score_2": [-9.0, 0.0, 100.0, 900.0],
            "nearest_strong_similarity": [0.9, 0.1, 0.2, 0.8],
            "scaffold": ["ring1", "ring1", "ring2", "ring3"],
        }
    )
    rules = {"primary_seed": 1, "pool_cap": 3, "pool_fraction": 1, "maximum_candidates": 10}
    ranked = add_rankings(frame, [1, 2])
    assert ranked.set_index("candidate_id").loc["A", "rank_1"] == 1
    result, budget = nominate(ranked, rules)
    assert budget == 3 and result.candidate_id.tolist() == ["A", "C"]
    frame["score_2"] *= -1
    assert nominate(add_rankings(frame, [1, 2]), rules)[0].candidate_id.tolist() == ["A", "C"]
    frame.loc[0, "score_1"] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        add_rankings(frame, [1, 2])


def test_nearest_support_keeps_same_graph_opposite_assay_classes():
    units = pd.DataFrame(
        {
            "smiles": ["CCO", "CCO", "CCN"],
            "unit_id": ["strong", "weak", "other"],
            "label": [1, 0, 0],
        }
    )
    result = nearest_support(["CCO"], units).iloc[0]
    assert result.nearest_strong_similarity == result.nearest_weak_similarity == 1.0
    assert result.nearest_strong_unit_id == "strong"
    assert result.nearest_weak_unit_id == "weak"


def test_source_and_training_scope_guards_reject_drift():
    source = {
        "status": "complete",
        "database_version": "ChEMBL_37",
        "selection": {
            "molecule_type": "Small molecule",
            "order_by": "molecule_chembl_id",
            "requested_limit": 2,
        },
        "statistics": {"downloaded_records": 2},
        "known_target_identity": {"target_chembl_id": "CHEMBL3714079"},
    }
    validate_source(source, 2)
    source["known_target_identity"]["target_chembl_id"] = "other_target"
    with pytest.raises(ValueError, match="来源"):
        validate_source(source, 2)
    units = pd.DataFrame({"smiles": ["CCO", "CCCCO"]})
    scope = {"min_heavy_atoms": 3, "max_heavy_atoms": 5, "allowed_elements": ["H", "C", "O"]}
    validate_training_scope(units, scope)
    scope["max_heavy_atoms"] = 6
    with pytest.raises(ValueError, match="大小范围"):
        validate_training_scope(units, scope)
