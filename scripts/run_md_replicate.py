"""从验收小样末态建立独立随机重复，固定时长、在线检查与检查点。"""

import argparse
import csv
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import openmm
from openmm import unit
from openmm.app import DCDReporter, PDBFile, Simulation
from run_membrane_control import resource_check, save, write_pdb

from translock_screening.md_pilot import PilotDiagnostics, fresh_replica
from translock_screening.md_restart import synchronize_restart_defaults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "warmup", "restart", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-ps", type=int, default=5000)
    args = parser.parse_args()
    assert args.time_ps > 0 and args.time_ps % 10 == 0
    warmup = json.loads((args.warmup / "manifest.json").read_text())
    readback = json.loads((args.warmup / "analysis_v1/independent_readback.json").read_text())
    restart = json.loads((args.restart / "manifest.json").read_text())
    assert warmup["status"] == "complete_1ns_technical_sample" and readback["status"] == "passed"
    assert restart["status"] == "restart_checkpoint_passed"
    cfg = warmup["config"]
    resource_check(args.root, cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    code = args.output / "code_snapshot"
    code.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("run_membrane_control.py")):
        shutil.copyfile(path, code / path.name)
    repo = Path(__file__).parents[1]
    for name in ("md_pilot.py", "md_diagnostics.py", "md_restart.py"):
        shutil.copyfile(repo / "src/translock_screening" / name, code / name)
    for name in ("uv.lock", "pyproject.toml"):
        shutil.copyfile(repo / "environments/membrane_md" / name, code / name)
    for name in ("state.xml", "system.xml"):
        shutil.copyfile(args.restart / name, args.output / ("source_" + name))
    shutil.copyfile(args.warmup / "atom_map.json", args.output / "atom_map.json")
    result = dict(
        status="initializing",
        ligand_id=cfg.get("ligand_id", "GLPG1205"),
        warmup=str(args.warmup),
        restart=str(args.restart),
        construct=warmup["construct"],
        parameters=warmup["parameters"],
        target_time_ps=args.time_ps,
        preparation_config=cfg,
        production_conditions={
            "temperature_K": 310,
            "pressure_bar": 1,
            "surface_tension_bar_nm": 0,
            "membrane_scaling": "XYIsotropic/ZFree",
            "timestep_fs": 2,
            "friction_per_ps": 1,
            "barostat_interval_steps": 25,
        },
        base_seed=args.seed,
        started_utc=datetime.now(UTC).isoformat(),
        checkpoint_rng_loaded=False,
        timestep_fs=2,
        report_interval_ps=10,
        checkpoint_interval_ps=500,
        position_restraint_k=0,
        time_ps=0,
        scope="共同准备末态条件下的新速度/恒温器/压强随机重复；短程姿势证据，非活性确证",
    )
    save(args.output / "manifest.json", result)
    sim = None
    try:
        state = openmm.XmlSerializer.deserialize((args.restart / "state.xml").read_text())
        system = openmm.XmlSerializer.deserialize((args.restart / "system.xml").read_text())
        assert not any(isinstance(f, openmm.CustomCompoundBondForce) for f in system.getForces())
        pdb = PDBFile(str(args.restart / "system.pdb"))
        atoms = json.loads((args.warmup / "atom_map.json").read_text())
        assert len(atoms) == pdb.topology.getNumAtoms() == system.getNumParticles()
        for atom, record in zip(pdb.topology.atoms(), atoms, strict=True):
            assert atom.element.symbol == record["element"]
            atom.name, atom.residue.name = record["name"], record["residue"]
            atom.residue.id, atom.residue.chain.id = record["residue_id"], record["chain"]
        integrator, seeds = fresh_replica(system, state, args.seed)
        result["random_seeds"] = seeds
        pdb.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        sim = Simulation(
            pdb.topology,
            system,
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),
            {"DeviceIndex": "0", "Precision": "mixed"},
        )
        sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        sim.context.setPositions(state.getPositions())
        sim.context.setVelocitiesToTemperature(310 * unit.kelvin, seeds["velocities"])
        sim.context.applyVelocityConstraints(1e-6)
        assert sim.context.getParameter("k") == 0
        sim.context.setTime(0)
        sim.currentStep = 0
        sim.saveState(str(args.output / "initial_state.xml"))
        (args.output / "system.xml").write_text(
            openmm.XmlSerializer.serialize(system), encoding="utf-8"
        )
        (args.output / "integrator.xml").write_text(
            openmm.XmlSerializer.serialize(integrator), encoding="utf-8"
        )
        initial_xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        diagnostics = PilotDiagnostics(args.warmup, initial_xyz)
        dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
        result["atom_count"] = system.getNumParticles()
        result["degrees_of_freedom"] = dof
        rows = []
        start = time.perf_counter()

        def diagnose():
            current = sim.context.getState(
                getPositions=True, getEnergy=True, getForces=True, getParameters=True
            )
            xyz = current.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            box = current.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            force = current.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            potential = current.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic = current.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            assert np.isfinite(force).all() and np.isfinite([potential, kinetic]).all()
            assert current.getParameters()["k"] == 0
            row = diagnostics.calculate(xyz, box)
            row.update(
                time_ps=float(current.getTime().value_in_unit(unit.picosecond)),
                potential_energy_kJ_mol=float(potential),
                temperature_K=float(2 * kinetic / (0.00831446261815324 * dof)),
                max_force_kJ_mol_nm=float(np.linalg.norm(force, axis=1).max()),
                wall_seconds=time.perf_counter() - start,
                **resource_check(args.root, cfg),
            )
            rows.append(row)
            with (args.output / "diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            result.update(
                status="running",
                time_ps=row["time_ps"],
                latest_diagnostics=row,
                updated_utc=datetime.now(UTC).isoformat(),
            )
            save(args.output / "manifest.json", result)
            assert row["geometry_passed"], f"固定几何检查未通过：{row}"
            return current

        diagnose()
        sim.reporters.append(
            DCDReporter(str(args.output / "trajectory.dcd"), 5000, enforcePeriodicBox=False)
        )
        for elapsed in range(10, args.time_ps + 1, 10):
            sim.step(5000)
            current = diagnose()
            if elapsed % 500 == 0 or elapsed == args.time_ps:
                sim.saveCheckpoint(str(args.output / f"checkpoint_{elapsed:05d}ps.chk"))
                print(
                    json.dumps(
                        {
                            "ligand": result["ligand_id"],
                            "seed": args.seed,
                            "time_ps": elapsed,
                            "wall_seconds": time.perf_counter() - start,
                            "ligand_RMSD_A": rows[-1]["ligand_RMSD_to_original_pose_A"],
                        }
                    ),
                    flush=True,
                )
        sim.saveState(str(args.output / "final_state.xml"))
        synchronize_restart_defaults(system, current)
        pdb.topology.setPeriodicBoxVectors(current.getPeriodicBoxVectors())
        write_pdb(args.output / "final.pdb", pdb.topology, current.getPositions())
        (args.output / "final_system.xml").write_text(
            openmm.XmlSerializer.serialize(system), encoding="utf-8"
        )
        (args.output / "final_integrator.xml").write_text(
            openmm.XmlSerializer.serialize(integrator), encoding="utf-8"
        )
        scalar_rows = [
            {k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows
        ]
        with (args.output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_rows[0])
            writer.writeheader()
            writer.writerows(scalar_rows)
        result.update(
            status="complete_fixed_length",
            wall_seconds=time.perf_counter() - start,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        save(args.output / "manifest.json", result)
    except Exception as error:
        result.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_utc=datetime.now(UTC).isoformat(),
        )
        if sim is not None:
            try:
                sim.saveState(str(args.output / "failure_state.xml"))
                sim.saveCheckpoint(str(args.output / "failure.chk"))
            except Exception as save_error:
                result["failure_save_error"] = str(save_error)
        save(args.output / "manifest.json", result)
        raise


if __name__ == "__main__":
    main()
