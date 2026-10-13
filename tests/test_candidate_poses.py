"""结构假设保留原已指定构型、双键与候选身份，选姿势不使用活性标签。"""

from rdkit import Chem

from translock_screening.candidate_poses import (
    conditional_stereoisomers,
    first_acceptable_pose,
)


def test_only_unspecified_stereo_changes():
    original = "C[C@H](O)C(F)Cl"
    states = conditional_stereoisomers(original)
    assert len(states) == 2
    original_mol = Chem.MolFromSmiles(original)
    original_cip = {i: c for i, c in Chem.FindMolChiralCenters(original_mol) if c != "?"}
    for state in states:
        mol = Chem.MolFromSmiles(state)
        mapping = mol.GetSubstructMatch(original_mol)
        cip = dict(Chem.FindMolChiralCenters(mol))
        assert all(cip[mapping[i]] == value for i, value in original_cip.items())
    assert conditional_stereoisomers("C/C=C/Cl") == ["C/C=C/Cl"]
    assert len(conditional_stereoisomers("CC=CCl")) == 2


def test_engine_order_is_retained_when_higher_pose_fails():
    audits = [
        {"rank": 1, "valid": False},
        {"rank": 2, "valid": True, "minimum_receptor_distance_A": 1.0},
        {"rank": 3, "valid": True, "minimum_receptor_distance_A": 1.8},
        {"rank": 4, "valid": True, "minimum_receptor_distance_A": 2.0},
    ]
    assert first_acceptable_pose(audits, 1.5) == 3
    assert first_acceptable_pose(audits, 3) is None
