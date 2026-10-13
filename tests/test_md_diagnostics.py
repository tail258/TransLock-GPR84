"""真实周期边界和对齐陷阱的最小测试。"""

import numpy as np
import pytest

from translock_screening.md_diagnostics import fit_to_reference, minimum_image, pair_distances


def test_boundary_bond_is_short_not_box_length():
    xyz = np.array([[0.01, 0, 0], [4.99, 0, 0]])
    assert pair_distances(xyz, [(0, 1)], [5, 5, 5])[0] == pytest.approx(0.02)


def test_nan_and_invalid_box_stop_diagnostics():
    with pytest.raises(ValueError):
        minimum_image([np.nan, 0, 0], [5, 5, 5])
    with pytest.raises(ValueError):
        minimum_image([0, 0, 0], [5, 0, 5])


def test_alignment_removes_translation_but_not_internal_ligand_displacement():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    shift = np.array([5, 6, 7])
    rotation, a, b, rmsd = fit_to_reference(reference + shift, reference)
    assert rmsd < 1e-10
    ligand = np.array([[0.5, 0.5, 0.5]])
    displaced = ligand + shift + np.array([0.1, 0, 0])
    assert np.linalg.norm((displaced - a) @ rotation + b - ligand) == pytest.approx(0.1)
