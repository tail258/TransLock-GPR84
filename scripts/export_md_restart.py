"""从完整对照末态导出可恢复包，并实际加载checkpoint核查，不增加MD时长。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import openmm
from openmm import unit
from openmm.app import PDBFile, Simulation

from translock_screening.md_restart import synchronize_restart_defaults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["status"] == "complete_1ns_technical_sample"
    stage = manifest["stages"][-1]
    name = stage["name"]
    state = openmm.XmlSerializer.deserialize((args.run / f"{name}_state.xml").read_text())
    system = openmm.XmlSerializer.deserialize((args.run / f"{name}_system.xml").read_text())
    integrator = openmm.XmlSerializer.deserialize((args.run / f"{name}_integrator.xml").read_text())
    pdb = PDBFile(str(args.run / f"{name}.pdb"))
    records = json.loads((args.run / "atom_map.json").read_text())
    assert pdb.topology.getNumAtoms() == system.getNumParticles() == len(records)
    for atom, record in zip(pdb.topology.atoms(), records, strict=True):
        assert atom.element.symbol == record["element"]
        atom.name = record["name"]
        atom.residue.name = record["residue"]
        atom.residue.id = record["residue_id"]
        atom.residue.chain.id = record["chain"]
    changed = synchronize_restart_defaults(system, state)
    pdb.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/md_restart.py",
        args.output / "md_restart.py",
    )
    for suffix, target in [
        ("state.xml", "state.xml"),
        ("integrator.xml", "integrator.xml"),
        ("chk", "checkpoint.chk"),
    ]:
        shutil.copyfile(
            args.run / f"{name}_{suffix}" if suffix != "chk" else args.run / f"{name}.chk",
            args.output / target,
        )
    (args.output / "system.xml").write_text(
        openmm.XmlSerializer.serialize(system), encoding="utf-8"
    )
    with (args.output / "system.pdb").open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(pdb.topology, state.getPositions(), handle, keepIds=True)
    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("CUDA"),
        {"DeviceIndex": "0", "Precision": "mixed"},
    )
    assert simulation.context.getParameter("k") == stage["restraint_k"] == 0
    simulation.loadCheckpoint(str(args.output / "checkpoint.chk"))
    check = simulation.context.getState(
        getPositions=True, getEnergy=True, getForces=True, getParameters=True
    )
    xyz = check.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    delta = np.max(np.abs(xyz - state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))
    assert delta < 1e-6 and np.isfinite(xyz).all()
    forces = check.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    energy = check.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert np.isfinite(forces).all() and np.isfinite(energy)
    time_ps = check.getTime().value_in_unit(unit.picosecond)
    assert abs(time_ps - 1000) < 1e-5 and check.getParameters()["k"] == 0
    result = {
        "status": "restart_checkpoint_passed",
        "source_run": str(args.run),
        "source_stage": name,
        "simulation_time_ps": time_ps,
        "steps_executed": 0,
        "updated_system_defaults": changed,
        "maximum_position_difference_nm": float(delta),
        "restored_energy_kJ_mol": float(energy),
        "box_nm": check.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer).tolist(),
        "scope": "真实checkpoint恢复/导出核查；未增加轨迹，不是独立重复",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
