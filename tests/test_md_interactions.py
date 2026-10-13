import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from translock_screening.md_interactions import (
    circular_statistics,
    dihedrals,
    gromos_clusters,
    hbond_geometry,
    ligand_features,
    pairwise_pose_distances,
    rmsf,
    torsion_definitions,
)


def test_hbond_geometry_uses_actual_hydrogen_and_periodic_image():
    xyz = np.array([[9.5, 0, 0], [0.5, 0, 0], [2.2, 0, 0], [0.5, 2, 0]])
    distance, angle = hbond_geometry(xyz, [[0, 1, 2], [0, 1, 3]], [10, 10, 10])
    assert distance[0] == pytest.approx(2.7)
    assert angle[0] == pytest.approx(180)
    assert angle[1] == pytest.approx(90)


def test_amide_nitrogen_is_not_acceptor_and_partial_polarity_is_not_salt():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NCS(=O)(=O)C"))
    f = ligand_features(mol)
    assert 3 in f["Donor"] and 3 not in f["Acceptor"]
    assert 2 in f["Acceptor"]
    assert f["charged_atoms"] == []
    assert 1 not in f["nonpolar_carbons"] and 4 not in f["nonpolar_carbons"]


def test_zwitterion_is_not_misclassified_as_no_charged_groups():
    features = ligand_features(Chem.AddHs(Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")))
    assert len(features["charged_atoms"]) == 2


def test_rmsf_uses_fluctuation_about_mean_not_input_displacement():
    xyz = np.array([[[8.0, 0, 0]], [[10.0, 0, 0]], [[12.0, 0, 0]]])
    assert rmsf(xyz)[0] == pytest.approx(np.sqrt(8 / 3))
    assert np.allclose(rmsf(xyz + 40), rmsf(xyz))


def test_clustering_retains_translation_and_has_deterministic_centers():
    xyz = np.array([[[x, 0, 0], [x, 1, 0]] for x in [0, 0.2, 3, 3.3]])
    d = pairwise_pose_distances(xyz)
    assert d[0, 2] == pytest.approx(3)
    labels, centers = gromos_clusters(d, 0.5)
    assert labels.tolist() == [1, 1, 2, 2]
    assert centers == [0, 2]


def test_dihedral_matches_independent_rdkit_and_wrap_statistics():
    xyz = np.array([[0.0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 1]])
    conf = Chem.Conformer(4)
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, p)
    value = dihedrals(xyz[None], [[0, 1, 2, 3]])[0, 0]
    assert value == pytest.approx(rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3))
    stats = circular_statistics([179, -179])
    assert abs(stats["circular_mean_deg"]) == pytest.approx(180)
    assert stats["resultant_length"] > 0.99


def test_torsion_scope_is_topological_not_outcome_selected():
    mol = Chem.MolFromSmiles("CCOc1ccccc1")
    defs = torsion_definitions(mol, list(range(3, 9)))
    assert {tuple(d["atoms"][1:3]) for d in defs} == {(1, 2), (2, 3)}
    assert defs[0]["core_boundary"]


def test_sp_adjacent_angles_are_excluded_as_near_linear():
    mol = Chem.MolFromSmiles("c1ccccc1C#CC1CC1")
    assert torsion_definitions(mol, range(mol.GetNumAtoms())) == []
