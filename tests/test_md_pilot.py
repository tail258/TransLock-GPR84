"""重复随机源与周期像的真实OpenMM最小检查。"""

import numpy as np
import openmm
import pytest
from openmm import unit

from translock_screening.md_pilot import fresh_replica, nearest_molecular_image


def test_molecular_periodic_translation_preserves_internal_distances():
    points = np.array([[5.1, 0, 0], [5.2, 0.1, 0]])
    whole = nearest_molecular_image(points, np.zeros(3), [5, 5, 5])
    assert np.allclose(whole, points - [5, 0, 0])
    assert np.linalg.norm(whole[1] - whole[0]) == pytest.approx(
        np.linalg.norm(points[1] - points[0])
    )


def test_new_seed_changes_velocities_without_moving_starting_positions():
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    xyz = np.eye(4, 3)
    context.setPositions(xyz)
    state = context.getState(getPositions=True, getParameters=True)
    velocities = []
    for seed in (20261011, 20261012):
        replica = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
        new_integrator, seeds = fresh_replica(replica, state, seed)
        new_context = openmm.Context(
            replica, new_integrator, openmm.Platform.getPlatformByName("Reference")
        )
        new_context.setPositions(state.getPositions())
        new_context.setVelocitiesToTemperature(310 * unit.kelvin, seeds["velocities"])
        result = new_context.getState(getPositions=True, getVelocities=True)
        assert np.allclose(result.getPositions(asNumpy=True).value_in_unit(unit.nanometer), xyz)
        assert new_integrator.getRandomNumberSeed() == seed
        velocities.append(
            result.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        )
        del new_context, new_integrator
    assert not np.allclose(*velocities)


def test_fresh_replica_barostat_seed_and_rejects_restrained_state():
    system = openmm.System()
    system.addParticle(12)
    restraint = openmm.CustomExternalForce("k*x*x")
    restraint.addGlobalParameter("k", 1000)
    restraint.addParticle(0, [])
    system.addForce(restraint)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0]])
    with pytest.raises(ValueError, match="位置约束"):
        fresh_replica(system, context.getState(getPositions=True, getParameters=True), 42)
    context.setParameter("k", 0)
    state = context.getState(getPositions=True, getParameters=True)
    replica = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    barostat = openmm.MonteCarloMembraneBarostat(
        1,
        0,
        310,
        openmm.MonteCarloMembraneBarostat.XYIsotropic,
        openmm.MonteCarloMembraneBarostat.ZFree,
        0,
    )
    replica.addForce(barostat)
    new_integrator, seeds = fresh_replica(replica, state, 42)
    assert seeds == {"thermostat": 42, "barostat": 142, "velocities": 242}
    assert barostat.getRandomNumberSeed() == 142 and barostat.getFrequency() == 25
    assert replica.getForce(0).getGlobalParameterDefaultValue(0) == 0
    assert new_integrator.getStepSize().value_in_unit(unit.femtosecond) == pytest.approx(2)
