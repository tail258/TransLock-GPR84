"""连接肽身份与坐标变换的风险测试。"""

import numpy as np
import pytest

from translock_screening.membrane_construct import (
    alanine_linker_map,
    alpha_hydrogen_from_heavy,
    chiral_minimization_barrier,
    l_alanine_cb,
    rotation_to_z,
)


def test_minimization_barrier_rejects_mirror_without_penalizing_l_geometry():
    import openmm
    from openmm import unit

    system = openmm.System()
    for _ in range(4):
        system.addParticle(12)
    system.addForce(chiral_minimization_barrier([(0, 1, 2, 3)]))
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    positions = np.array([[0.146, 0, 0], [0, 0, 0], [-0.052, 0.142, 0], [0, 0, 0.153]])
    context.setPositions(positions)
    assert (
        context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        == 0
    )
    positions[3, 2] *= -1
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    assert state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) > 1000
    assert np.isfinite(
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    ).all()


def test_linker_sidechain_has_l_geometry_before_hydrogens():
    n, ca, c = np.array([1.46, 0, 0]), np.zeros(3), np.array([-0.52, 1.42, 0])
    cb = l_alanine_cb(n, ca, c)
    assert np.linalg.norm(cb - ca) == pytest.approx(1.53)
    assert np.dot(np.cross(n - ca, c - ca), cb - ca) > 1
    r = rotation_to_z([1, 2, 3])
    assert np.allclose(l_alanine_cb(r @ n, r @ ca, r @ c), r @ cb)


def test_alpha_hydrogen_is_opposite_cb_and_rotation_invariant():
    n, ca, c = np.array([1.46, 0, 0]), np.zeros(3), np.array([-0.52, 1.42, 0])
    cb = l_alanine_cb(n, ca, c)
    h = alpha_hydrogen_from_heavy(n, ca, c, cb)
    assert np.linalg.norm(h - ca) == pytest.approx(1.09)
    assert np.dot(np.cross(n - ca, c - ca), h - ca) < -1
    angle = np.rad2deg(np.arccos(np.dot(cb - ca, h - ca) / (1.53 * 1.09)))
    assert 100 < angle < 120
    r = rotation_to_z([1, 2, 3])
    assert np.allclose(alpha_hydrogen_from_heavy(r @ n, r @ ca, r @ c, r @ cb), r @ h)
    with pytest.raises(ValueError, match="四面体"):
        alpha_hydrogen_from_heavy(n, ca, c, cb * [1, 1, -1])


def test_linker_never_inherits_native_identity():
    ids = list(range(9, 220)) + list(range(309, 388))
    records = alanine_linker_map(ids, 218, 318, 13)
    assert len(records) == 291
    synthetic = [r for r in records if r["native_id"] is None]
    assert len(synthetic) == 13
    assert [r["model_id"] for r in synthetic] == list(range(210, 223))
    assert records[208]["native_id"] == 217
    assert records[222]["native_id"] == 319


def test_missing_endpoint_stops_construct():
    with pytest.raises(ValueError, match="端点"):
        alanine_linker_map([9, 10, 319, 320], 218, 318, 13)


@pytest.mark.parametrize("axis", [[0, 0, 1], [0, 0, -1], [1, 2, 3]])
def test_orientation_is_proper_rotation(axis):
    r = rotation_to_z(axis)
    assert np.linalg.det(r) == pytest.approx(1)
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.allclose(r @ (np.array(axis) / np.linalg.norm(axis)), [0, 0, 1])
    xyz = np.array([[1, 0, 2], [-2, 3, 4], [5, 1, 0]])
    assert np.allclose(
        np.linalg.norm(xyz[1:] - xyz[0], axis=1),
        np.linalg.norm((xyz @ r.T)[1:] - (xyz @ r.T)[0], axis=1),
    )
