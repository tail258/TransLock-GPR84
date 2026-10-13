"""检查只加载导出的System时，不会恢复成旧约束强度或旧盒尺寸。"""

import numpy as np
import openmm
from openmm import unit

from translock_screening.md_restart import synchronize_restart_defaults


def test_restart_defaults_follow_actual_state_without_state_parameter_override():
    system = openmm.System()
    system.addParticle(12)
    force = openmm.CustomExternalForce("0.5*k*x*x")
    force.addGlobalParameter("k", 1000)
    force.addParticle(0, [])
    system.addForce(force)
    context = openmm.Context(
        system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions([[1, 0, 0]])
    context.setParameter("k", 0)
    vectors = [openmm.Vec3(3, 0, 0), openmm.Vec3(0, 4, 0), openmm.Vec3(0, 0, 5)]
    context.setPeriodicBoxVectors(*vectors)
    state = context.getState(getPositions=True, getParameters=True)
    synchronize_restart_defaults(system, state)
    # 新Context只使用System默认值，未加载State来掩盖导出错误。
    restored = openmm.Context(
        system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    restored.setPositions([[1, 0, 0]])
    assert restored.getParameter("k") == 0
    assert (
        restored.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
        == 0
    )
    box = restored.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.allclose(box, np.diag([3, 4, 5]))
