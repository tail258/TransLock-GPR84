"""从真实State同步恢复用System默认参数与盒尺寸；不改变已完成轨迹。"""


def synchronize_restart_defaults(system, state):
    import openmm
    from openmm import unit

    parameters = dict(state.getParameters())
    changed = []
    for force in system.getForces():
        if hasattr(force, "getNumGlobalParameters"):
            for i in range(force.getNumGlobalParameters()):
                name = force.getGlobalParameterName(i)
                if name in parameters:
                    before = force.getGlobalParameterDefaultValue(i)
                    force.setGlobalParameterDefaultValue(i, parameters[name])
                    if before != parameters[name]:
                        changed.append({"name": name, "before": before, "after": parameters[name]})
        if isinstance(force, openmm.MonteCarloMembraneBarostat):
            force.setDefaultPressure(parameters[force.Pressure()] * unit.bar)
            force.setDefaultTemperature(parameters[force.Temperature()] * unit.kelvin)
            force.setDefaultSurfaceTension(
                parameters[force.SurfaceTension()] * unit.bar * unit.nanometer
            )
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    return changed
