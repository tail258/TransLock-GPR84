"""骨架匹配不得与完整身份、杂原子差别或药理标签混淆。"""

import pandas as pd

from translock_screening.scaffold_audit import candidate_pairs, compare, describe, prepare_reference


def test_ring_sidechains_and_generic_framework_are_distinct_questions():
    a, b, c = [describe(s) for s in ("Cc1ccccc1", "CCc1ccccc1", "Cc1ccncc1")]
    assert a["connection"] != b["connection"]
    assert a["scaffold"] == b["scaffold"] != c["scaffold"]
    assert a["generic_framework"] == c["generic_framework"]
    assert describe("CCO")["scaffold"] == ""


def test_scope_duplicates_and_invalid_rows_remain_traceable():
    frame = pd.DataFrame(
        {"id": ["a", "b", "c", "bad"], "smi": ["Cc1ccccc1", "Cc1ccccc1", "Cc1ccncc1", "bad"]}
    )
    reference, failures = prepare_reference(frame, "test", "id", "smi")
    assert len(reference) == 3 and failures[0]["reference_id"] == "bad"
    candidates = pd.DataFrame({"candidate_id": ["x"], "smiles": ["CCc1ccccc1"]})
    result = compare(candidates, reference).iloc[0]
    assert result.reference_graphs == 2 and result.connection_match_rows == 0
    assert result.scaffold_match_rows == 2 and result.generic_framework_match_rows == 3
    assert result.scaffold_match_ids == "a||b"
    assert result.nearest_scaffold_similarity == 1


def test_stereo_is_ignored_and_pairwise_framework_redundancy_exposed():
    assert describe("C[C@H](O)c1ccccc1") == describe("C[C@@H](O)c1ccccc1")
    pairs = candidate_pairs(
        pd.DataFrame(
            {"candidate_id": ["a", "b", "c"], "smiles": ["Cc1ccccc1", "Cc1ccncc1", "C1CCCCC1"]}
        )
    )
    assert len(pairs) == 3 and pairs.same_generic_framework.all()
    assert not pairs.same_scaffold.any() and not pairs.same_connection.any()
