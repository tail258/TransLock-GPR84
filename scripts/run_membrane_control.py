"""构建GLPG1205/受体/POPC/水盐并运行固定分级膜小样。"""

import argparse
import csv
import importlib.metadata
import json
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import openmm
import yaml
from openmm import unit
from openmm.app import PME, DCDReporter, ForceField, HBonds, Modeller, PDBFile, Simulation

from translock_screening.md_diagnostics import fit_to_reference, minimum_image, pair_distances
from translock_screening.md_restart import synchronize_restart_defaults


def save(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def resource_check(root, cfg):
    free = shutil.disk_usage(root).free / 2**30
    temperature = float(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .splitlines()[0]
    )
    if free < cfg["minimum_free_GiB"] or temperature >= cfg["max_gpu_temperature_C"]:
        raise RuntimeError(f"资源停止线：{free:.1f}GiB，{temperature}°C")
    return {"free_GiB": free, "gpu_temperature_C": temperature}


def write_pdb(path, topology, positions):
    with path.open("w", encoding="utf-8") as h:
        PDBFile.writeFile(topology, positions, h, keepIds=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "construct", "parameters", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reuse-built-system", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    construct = json.loads((args.construct / "manifest.json").read_text())
    parameters = json.loads((args.parameters / "manifest.json").read_text())
    assert construct["status"] == "construct_geometry_passed"
    assert parameters["status"] == "ligand_parameters_passed"
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/md_diagnostics.py",
        args.output / "md_diagnostics.py",
    )
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/membrane_construct.py",
        args.output / "membrane_construct.py",
    )
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/md_restart.py",
        args.output / "md_restart.py",
    )
    shutil.copyfile(args.config, args.output / "config.yaml")
    envroot = Path(__file__).parents[1] / "environments/membrane_md"
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(envroot / name, args.output / name)
    result = {
        "status": "preparing_complete_system",
        "construct": str(args.construct),
        "parameters": str(args.parameters),
        "config": cfg,
        "stages": [],
        "resource_samples": [],
        "scope": "1ns完整膜技术/初步平衡小样；不代表充分平衡或未知候选活性",
    }
    save(args.output / "manifest.json", result)
    phase = "initialize"
    sim = None
    try:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        from rdkit import Chem

        result["resource_samples"].append(resource_check(args.root, cfg))
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        mol = Molecule.from_json((args.parameters / "openff_molecule.json").read_text())
        generator = SMIRNOFFTemplateGenerator(
            molecules=mol,
            forcefield=cfg["ligand_forcefield"],
            cache=str(args.output / "template_cache.json"),
        )
        ff = ForceField(*cfg["forcefields"])
        ff.registerTemplateGenerator(generator.generator)
        protein = PDBFile(str(args.construct / "receptor_minimized.pdb"))
        ligand = PDBFile(str(args.parameters / "ligand.pdb"))
        expected_atoms = json.loads((args.parameters / "atom_parameters.json").read_text())
        stereo_template = Chem.SDMolSupplier(str(args.parameters / "ligand.sdf"), removeHs=False)[0]
        assert stereo_template is not None and stereo_template.GetNumAtoms() == len(expected_atoms)
        ligand_atoms = list(ligand.topology.atoms())
        assert len(ligand_atoms) == len(expected_atoms)
        assert [a.element.atomic_number for a in ligand_atoms] == [
            a["atomic_number"] for a in expected_atoms
        ]
        xyz_read = np.array(ligand.positions.value_in_unit(unit.angstrom))
        assert (
            np.abs(xyz_read - np.array([a["position_A"] for a in expected_atoms])).max() < 0.00051
        )
        read_bonds = {frozenset((a.index, b.index)) for a, b in ligand.topology.bonds()}
        expected_bonds = {frozenset((b.atom1_index, b.atom2_index)) for b in mol.bonds}
        assert read_bonds == expected_bonds
        renamed = []
        for atom, record in zip(ligand_atoms, expected_atoms, strict=True):
            if atom.name != record["name"]:
                renamed.append(
                    {"index": atom.index, "parser_name": atom.name, "mapped_name": record["name"]}
                )
                atom.name = record["name"]
        result["ligand_pdb_atom_name_normalizations"] = renamed
        model = Modeller(protein.topology, protein.positions)
        model.add(ligand.topology, ligand.positions)
        protein_atom_count = protein.topology.getNumAtoms()
        ligand_indices = list(range(protein_atom_count, model.topology.getNumAtoms()))
        # 检查导入PDB后原子顺序；参数与坐标必须对应同一映射。
        assert [a.name for a in ligand.topology.atoms()] == [a["name"] for a in expected_atoms]
        ligand_heavy = [
            protein_atom_count + i for i, a in enumerate(expected_atoms) if a["atomic_number"] != 1
        ]
        mapping = json.loads((args.construct / "residue_map.json").read_text())
        native_by_model = {r["model_id"]: r["native_id"] for r in mapping}
        native_ca = [
            a.index
            for a in protein.topology.atoms()
            if a.name == "CA" and native_by_model[int(a.residue.id)] is not None
        ]
        restrained = [
            a.index
            for a in protein.topology.atoms()
            if a.name in {"N", "CA", "C", "O"} and native_by_model[int(a.residue.id)] is not None
        ] + ligand_heavy
        initial = np.array(model.positions.value_in_unit(unit.nanometer))
        phase = "membrane_build"
        result["status"] = "building_membrane"
        save(args.output / "manifest.json", result)
        build_start = time.perf_counter()
        if args.reuse_built_system:
            print("只读复用已建完整膜，核对所有原子顺序和元素", flush=True)
            old = json.loads((args.reuse_built_system / "manifest.json").read_text())
            assert old["construct"] == str(args.construct) and old["parameters"] == str(
                args.parameters
            )
            assert old["config"] == cfg
            built = PDBFile(str(args.reuse_built_system / "system_initial.pdb"))
            old_atoms = json.loads((args.reuse_built_system / "atom_map.json").read_text())
            assert built.topology.getNumAtoms() == len(old_atoms)
            for atom, record in zip(built.topology.atoms(), old_atoms, strict=True):
                assert atom.element.symbol == record["element"]
                atom.name = record["name"]
                atom.residue.name = record["residue"]
                atom.residue.id = record["residue_id"]
                atom.residue.chain.id = record["chain"]
            model = Modeller(built.topology, built.positions)
            result["reused_built_system"] = str(args.reuse_built_system)
            result["original_membrane_build_seconds"] = old["build_seconds"]
        else:
            print("开始POPC完整膜构建", flush=True)
            model.addMembrane(
                ff,
                lipidType="POPC",
                minimumPadding=cfg["minimum_padding_nm"] * unit.nanometer,
                ionicStrength=cfg["ionic_strength_molar"] * unit.molar,
                platform=openmm.Platform.getPlatformByName("CUDA"),
            )
        result["build_seconds"] = time.perf_counter() - build_start
        write_pdb(args.output / "system_initial.pdb", model.topology, model.positions)
        result["residue_counts"] = dict(Counter(r.name for r in model.topology.residues()))
        result["atom_count"] = model.topology.getNumAtoms()
        result["packages"] = {
            n: importlib.metadata.version(n)
            for n in ("openmm", "openff-toolkit", "openff-nagl", "openforcefields", "rdkit")
        }
        atoms = list(model.topology.atoms())
        assert len(ligand_indices) == parameters["atoms"]
        atom_records = [
            {
                "index": a.index,
                "name": a.name,
                "residue": a.residue.name,
                "residue_id": a.residue.id,
                "chain": a.residue.chain.id,
                "element": a.element.symbol if a.element else None,
            }
            for a in atoms
        ]
        save(args.output / "atom_map.json", atom_records)
        protein_by_atom = {(int(a.residue.id), a.name): a.index for a in protein.topology.atoms()}
        peptide_pairs = [
            (protein_by_atom[(i, "C")], protein_by_atom[(i + 1, "N")])
            for i in range(1, len(mapping))
        ]
        disulfide_pairs = [
            (a.index, b.index) for a, b in protein.topology.bonds() if a.name == b.name == "SG"
        ]
        alpha_quartets = [
            [protein_by_atom[(i, n)] for n in ("N", "CA", "C", "CB")]
            for i in range(1, len(mapping) + 1)
            if (i, "CB") in protein_by_atom
        ]
        alpha_hydrogens = [
            protein_by_atom[(i, "HA")]
            for i in range(1, len(mapping) + 1)
            if (i, "CB") in protein_by_atom
        ]
        alpha_model_ids = [i for i in range(1, len(mapping) + 1) if (i, "CB") in protein_by_atom]
        protein_heavy = [a.index for a in protein.topology.atoms() if a.element.symbol != "H"]
        phosphorus = [
            a.index for a in atoms if a.residue.name in {"POP", "POPC"} and a.element.symbol == "P"
        ]
        assert len(phosphorus) > 50
        if args.reuse_built_system:
            system = openmm.XmlSerializer.deserialize(
                (args.reuse_built_system / "system.xml").read_text()
            )
            assert system.getNumParticles() == model.topology.getNumAtoms()
            barostat = next(
                f for f in system.getForces() if isinstance(f, openmm.MonteCarloMembraneBarostat)
            )
        else:
            system = ff.createSystem(
                model.topology,
                nonbondedMethod=PME,
                nonbondedCutoff=1 * unit.nanometer,
                constraints=HBonds,
                rigidWater=True,
                removeCMMotion=True,
            )
            restraint = openmm.CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2")
            restraint.addGlobalParameter("k", 1000)
            for name in ("x0", "y0", "z0"):
                restraint.addPerParticleParameter(name)
            for i in restrained:
                restraint.addParticle(i, initial[i].tolist())
            system.addForce(restraint)
            barostat = openmm.MonteCarloMembraneBarostat(
                1 * unit.bar,
                0 * unit.bar * unit.nanometer,
                310 * unit.kelvin,
                openmm.MonteCarloMembraneBarostat.XYIsotropic,
                openmm.MonteCarloMembraneBarostat.ZFree,
                0,
            )
            barostat.setRandomNumberSeed(cfg["seed"])
            system.addForce(barostat)
        (args.output / "system.xml").write_text(
            openmm.XmlSerializer.serialize(system), encoding="utf-8"
        )
        assert not any(isinstance(f, openmm.CustomCompoundBondForce) for f in system.getForces())
        result["full_system_minimization_only_chiral_barrier"] = False
        integrator = openmm.LangevinMiddleIntegrator(
            50 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond
        )
        integrator.setRandomNumberSeed(cfg["seed"])
        integrator.setConstraintTolerance(1e-6)
        sim = Simulation(
            model.topology,
            system,
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),
            {"DeviceIndex": "0", "Precision": "mixed"},
        )
        sim.context.setPositions(model.positions)
        phase = "complete_system_minimize"
        print(f"完整体系 {result['atom_count']} 原子；开始最小化", flush=True)
        result["energy_before_min_kJ_mol"] = float(
            sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        sim.minimizeEnergy(
            tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=4000
        )
        state = sim.context.getState(getPositions=True, getEnergy=True, getForces=True)
        reference = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        write_pdb(args.output / "system_minimized.pdb", model.topology, state.getPositions())
        result["energy_after_min_kJ_mol"] = float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        assert np.isfinite(reference).all() and np.isfinite(result["energy_after_min_kJ_mol"])
        nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        total_charge = sum(
            nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in range(system.getNumParticles())
        )
        assert abs(total_charge) < 1e-3
        result["total_system_charge_e"] = float(total_charge)
        degrees_of_freedom = 3 * system.getNumParticles() - system.getNumConstraints() - 3
        result["temperature_degrees_of_freedom"] = degrees_of_freedom
        metrics = []
        diagnostic_path = args.output / "diagnostics.csv"

        def diagnose(stage_name):
            s = sim.context.getState(getPositions=True, getEnergy=True, getForces=True)
            xyz = np.array(s.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            force = s.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            box_vectors = np.array(
                s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            )
            assert np.abs(box_vectors - np.diag(np.diag(box_vectors))).max() < 1e-8
            box = np.diag(box_vectors)
            assert np.isfinite(xyz).all() and np.isfinite(force).all()
            energy = float(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            assert np.isfinite(energy)
            stereo_mol = Chem.Mol(stereo_template)
            for local, global_index in enumerate(ligand_indices):
                stereo_mol.GetConformer().SetAtomPosition(local, xyz[global_index] * 10)
            Chem.RemoveStereochemistry(stereo_mol)
            Chem.AssignStereochemistryFrom3D(stereo_mol, replaceExistingTags=True)
            stereo_key = Chem.MolToInchiKey(Chem.RemoveHs(stereo_mol))
            assert stereo_key == parameters["inchikey"], f"配体坐标立体改变：{stereo_key}"
            kinetic = float(s.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole))
            temperature_K = 2 * kinetic / (0.00831446261815324 * degrees_of_freedom)
            lengths = pair_distances(xyz, peptide_pairs, box) * 10
            ss = pair_distances(xyz, disulfide_pairs, box) * 10
            q = np.array(alpha_quartets)
            vectors = [minimum_image(xyz[q[:, i]] - xyz[q[:, 1]], box) * 10 for i in (0, 2, 3)]
            volumes = np.einsum("ij,ij->i", np.cross(vectors[0], vectors[1]), vectors[2])
            vh = minimum_image(xyz[alpha_hydrogens] - xyz[q[:, 1]], box) * 10
            hvolumes = np.einsum("ij,ij->i", np.cross(vectors[0], vectors[1]), vh)
            hangles = np.rad2deg(
                np.arccos(
                    np.clip(
                        np.einsum("ij,ij->i", vectors[2], vh)
                        / (np.linalg.norm(vectors[2], axis=1) * np.linalg.norm(vh, axis=1)),
                        -1,
                        1,
                    )
                )
            )
            rotation, a, b, ca_rmsd = fit_to_reference(xyz[native_ca], reference[native_ca])
            moved_ligand = (xyz[ligand_heavy] - a) @ rotation + b
            ligand_rmsd = float(
                np.sqrt(np.mean(np.sum((moved_ligand - reference[ligand_heavy]) ** 2, axis=1))) * 10
            )
            distances = np.linalg.norm(
                minimum_image(
                    xyz[np.array(protein_heavy)][:, None, :] - xyz[ligand_heavy][None, :, :], box
                ),
                axis=-1,
            )
            contact_ids = set()
            for i in np.where(distances.min(axis=1) <= 0.4)[0]:
                res = atoms[protein_heavy[i]].residue
                native = native_by_model[int(res.id)]
                contact_ids.add(
                    f"{res.name}{native}" if native is not None else f"synthetic_ALA{res.id}"
                )
            if not metrics:
                result["initial_contacts_4A"] = sorted(contact_ids)
            base_contacts = set(result["initial_contacts_4A"])
            pz = xyz[phosphorus, 2]
            midz = float(np.median(reference[phosphorus, 2]))
            upper, lower = pz[pz > midz], pz[pz < midz]
            assert len(upper) > len(phosphorus) * 0.3 and len(lower) > len(phosphorus) * 0.3
            row = {
                "stage": stage_name,
                "time_ps": float(s.getTime().value_in_unit(unit.picosecond)),
                "potential_energy_kJ_mol": energy,
                "temperature_K": temperature_K,
                "max_force_kJ_mol_nm": float(np.linalg.norm(force, axis=1).max()),
                "box_x_nm": float(box[0]),
                "box_y_nm": float(box[1]),
                "box_z_nm": float(box[2]),
                "area_nm2": float(box[0] * box[1]),
                "volume_nm3": float(np.prod(box)),
                "peptide_CN_min_A": float(lengths.min()),
                "peptide_CN_max_A": float(lengths.max()),
                "disulfide_min_A": float(ss.min()),
                "disulfide_max_A": float(ss.max()),
                "minimum_alpha_volume_A3": float(volumes.min()),
                "maximum_alpha_hydrogen_volume_A3": float(hvolumes.max()),
                "alpha_CB_CA_HA_angle_min_deg": float(hangles.min()),
                "alpha_CB_CA_HA_angle_max_deg": float(hangles.max()),
                "native_CA_aligned_RMSD_A": ca_rmsd * 10,
                "ligand_mapped_heavy_RMSD_after_CA_fit_A": ligand_rmsd,
                "minimum_ligand_receptor_heavy_distance_A": float(distances.min() * 10),
                "contact_recall": len(base_contacts & contact_ids) / len(base_contacts),
                "phosphate_layer_separation_nm": float(upper.mean() - lower.mean()),
            }
            assert row["phosphate_layer_separation_nm"] > 2
            metrics.append(row)
            save(args.output / "diagnostics.json", metrics)
            with diagnostic_path.open("w", newline="", encoding="utf-8") as h:
                writer = csv.DictWriter(h, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(metrics)
            result["resource_samples"].append(resource_check(args.root, cfg))
            if not (lengths.min() > 1.1 and lengths.max() < 1.6):
                raise ValueError(f"肽键C-N异常：{lengths.min():.4f}–{lengths.max():.4f} Å")
            if not (ss.min() > 1.8 and ss.max() < 2.3):
                raise ValueError(f"二硫键异常：{ss.min():.4f}–{ss.max():.4f} Å")
            bad = np.where((volumes <= 1) | (hvolumes >= 0) | (hangles <= 80) | (hangles >= 140))[0]
            if len(bad):
                failures = [
                    {
                        "model_id": alpha_model_ids[i],
                        "native_id": native_by_model[alpha_model_ids[i]],
                        "CB_volume_A3": float(volumes[i]),
                        "HA_volume_A3": float(hvolumes[i]),
                        "CB_CA_HA_angle_deg": float(hangles[i]),
                    }
                    for i in bad
                ]
                save(args.output / "failed_geometry.json", failures)
                raise ValueError("CA完整四面体异常：" + json.dumps(failures))
            return row

        diagnose("minimized")
        sim.context.setVelocitiesToTemperature(50 * unit.kelvin, cfg["seed"])
        sim.reporters.append(
            DCDReporter(str(args.output / "trajectory.dcd"), 5000, enforcePeriodicBox=False)
        )
        pressure_on = False
        result["status"] = "running_staged_equilibration"
        for stage in cfg["stages"]:
            phase = stage["name"]
            if stage["pressure"] and not pressure_on:
                barostat.setFrequency(25)
                sim.context.reinitialize(preserveState=True)
                pressure_on = True
            integrator.setTemperature(stage["temperature_K"] * unit.kelvin)
            integrator.setStepSize(stage["timestep_fs"] * unit.femtosecond)
            sim.context.setParameter("k", stage["restraint_k"])
            start = time.perf_counter()
            steps = round(stage["time_ps"] * 1000 / stage["timestep_fs"])
            chunk = round(10 * 1000 / stage["timestep_fs"])
            for offset in range(0, steps, chunk):
                sim.step(min(chunk, steps - offset))
                diagnose(stage["name"])
                if time.perf_counter() - start > cfg["max_stage_seconds"]:
                    raise TimeoutError("阶段超出固定时间预算")
            seconds = time.perf_counter() - start
            sim.saveCheckpoint(str(args.output / f"{stage['name']}.chk"))
            sim.saveState(str(args.output / f"{stage['name']}_state.xml"))
            stage_state = sim.context.getState(getPositions=True, getParameters=True)
            synchronize_restart_defaults(system, stage_state)
            model.topology.setPeriodicBoxVectors(stage_state.getPeriodicBoxVectors())
            (args.output / f"{stage['name']}_system.xml").write_text(
                openmm.XmlSerializer.serialize(system), encoding="utf-8"
            )
            (args.output / f"{stage['name']}_integrator.xml").write_text(
                openmm.XmlSerializer.serialize(integrator), encoding="utf-8"
            )
            write_pdb(
                args.output / f"{stage['name']}.pdb",
                model.topology,
                stage_state.getPositions(),
            )
            record = dict(
                stage,
                wall_seconds=seconds,
                estimated_ns_per_day=stage["time_ps"] / 1000 / seconds * 86400,
            )
            result["stages"].append(record)
            save(args.output / "manifest.json", result)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        result["status"] = "complete_1ns_technical_sample"
        result["final_diagnostics"] = metrics[-1]
        result["simulation_time_ps"] = metrics[-1]["time_ps"]
        save(args.output / "manifest.json", result)
        print(
            json.dumps({"status": result["status"], "time_ps": result["simulation_time_ps"]}),
            flush=True,
        )
    except Exception as error:
        result.update(
            status="failed_complete_system",
            failed_phase=phase,
            error_type=type(error).__name__,
            error=str(error),
        )
        if sim is not None:
            try:
                sim.saveState(str(args.output / "failure_state.xml"))
                sim.saveCheckpoint(str(args.output / "failure.chk"))
                result["failure_state_saved"] = True
            except Exception as save_error:
                result["failure_state_save_error"] = str(save_error)
        save(args.output / "manifest.json", result)
        raise


if __name__ == "__main__":
    main()
