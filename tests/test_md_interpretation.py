"""区分整体平移、刚体转向、真实内部形变及代表帧选择。"""

import numpy as np
import pytest
from rdkit import Chem

from translock_screening.md_interpretation import (
    actual_medoid_index,
    coordinate_only_molecule,
    displacement_metrics,
    murcko_atom_indices,
)


@pytest.fixture
def coordinates():
    return np.array([[0.0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3], [2, 2, 2]])


def test_translation_is_not_internal_deformation(coordinates):
    result = displacement_metrics(coordinates + [3, 4, 0], coordinates, [0, 1, 2, 3])
    assert result["total_A"] == pytest.approx(5)
    assert result["centroid_shift_A"] == pytest.approx(5)
    assert result["centered_A"] < 1e-10
    assert result["self_fit_internal_A"] < 1e-10


def test_rotation_remains_pose_change_but_not_internal_deformation(coordinates):
    q = coordinates - coordinates.mean(axis=0)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = displacement_metrics(q @ rotation, q, [0, 1, 2, 3])
    assert result["total_A"] > 1
    assert result["centroid_shift_A"] < 1e-10
    assert result["self_fit_internal_A"] < 1e-10


def test_terminal_movement_is_distinguished_from_core(coordinates):
    moved = coordinates.copy()
    moved[-1] += [0, 0, 5]
    result = displacement_metrics(moved, coordinates, [0, 1, 2, 3])
    assert result["core_A"] == 0
    assert result["periphery_A"] == 5
    assert result["self_fit_internal_A"] > 0.2


def test_reflection_is_not_hidden_as_proper_rotation(coordinates):
    result = displacement_metrics(coordinates * [-1, 1, 1], coordinates, [0, 1, 2, 3])
    assert result["self_fit_internal_A"] > 0.1


def test_representative_is_actual_central_frame_not_favorable_start(coordinates):
    frames = np.array([coordinates + x for x in [0, 4, 5]])
    assert actual_medoid_index(frames) == 1


def test_murcko_core_excludes_terminal_substituent():
    mol = Chem.MolFromSmiles("CCc1ccccc1")
    assert murcko_atom_indices(mol).tolist() == list(range(2, 8))


def test_nonfinite_coordinates_fail(coordinates):
    coordinates[0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        displacement_metrics(coordinates, coordinates, [0, 1, 2, 3])


def test_heavy_only_export_does_not_inherit_explicit_h_charge_array():
    molecule = Chem.MolFromSmiles("CCO")
    molecule.AddConformer(Chem.Conformer(3))
    molecule.SetProp("atom.dprop.PartialCharge", "0 0 0 0 0 0 0 0 0")
    xyz = np.array([[0.0, 0, 0], [1.4, 0, 0], [2.4, 1, 0]])
    exported = coordinate_only_molecule(molecule, xyz, "example")
    assert not exported.HasProp("atom.dprop.PartialCharge")
    assert molecule.HasProp("atom.dprop.PartialCharge")
    assert np.array_equal(exported.GetConformer().GetPositions(), xyz)
