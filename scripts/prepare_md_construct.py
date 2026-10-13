"""构建有明确原生映射的13ALA受体近似并共同取向实验配体。"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import openmm
import yaml
from openmm import unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer
from rdkit import Chem

from translock_screening.membrane_construct import (
    alanine_linker_map,
    alpha_hydrogen_from_heavy,
    chiral_minimization_barrier,
    l_alanine_cb,
    tm_geometry_transform,
)


def save(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_pdb(path, topology, positions):
    with path.open("w", encoding="utf-8") as h:
        PDBFile.writeFile(topology, positions, h, keepIds=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    sources = {n: args.root / cfg[n] for n in ("receptor", "uniprot", "ligand")}
    before = [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sources.values()
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    for name, p in sources.items():
        shutil.copyfile(p, args.output / ("input_" + name + p.suffix))
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    shutil.copyfile(args.config, args.output / "config.yaml")
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/membrane_construct.py",
        args.output / "membrane_construct.py",
    )
    result = {
        "status": "preparing",
        "inputs": before,
        "config": cfg,
        "scope": "合成连接肽近似；技术体系准备，非完整原生环建模",
    }
    save(args.output / "manifest.json", result)
    try:
        records = [
            s
            for s in sources["receptor"].read_text().splitlines()
            if s.startswith(("ATOM  ", "HETATM")) and s[76:78].strip() not in {"H", "D"}
        ]
        ca = {
            int(s[22:26]): np.array([float(s[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))])
            for s in records
            if s[12:16].strip() == "CA"
        }
        mapping = alanine_linker_map(list(ca), *cfg["replace_native"], len(cfg["synthetic_linker"]))
        if set(cfg["synthetic_linker"]) != {"A"}:
            raise ValueError("当前实现仅接受明确13ALA方案")
        save(args.output / "residue_map.json", mapping)
        model_id = {r["native_id"]: r["model_id"] for r in mapping if r["native_id"] is not None}
        uni = json.loads(sources["uniprot"].read_text())
        ranges = [
            [f["location"][k]["value"] for k in ("start", "end")]
            for f in uni["features"]
            if f["type"] == "Transmembrane"
        ]
        assert len(ranges) == 7
        rotation, origin, tm_summary = tm_geometry_transform(ca, ranges)
        save(
            args.output / "orientation.json",
            {
                "rotation": rotation.tolist(),
                "origin_A": origin.tolist(),
                "determinant": float(np.linalg.det(rotation)),
                "transmembrane_segments": tm_summary,
                "scope": "UniProt跨膜段几何取向；未进行OPM/PPM求解",
            },
        )
        moved_lines = []
        for s in records:
            native_id = int(s[22:26])
            if native_id not in model_id:
                continue
            xyz = np.array([float(s[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))])
            p = rotation @ (xyz - origin)
            s = (
                s[:6]
                + f"{len(moved_lines) + 1:5d}"
                + s[11:21]
                + "R"
                + f"{model_id[native_id]:4d}"
                + " "
                + s[27:30]
                + "".join(f"{v:8.3f}" for v in p)
                + s[54:]
            )
            moved_lines.append(s)
        heavy = args.output / "retained_oriented_heavy.pdb"
        heavy.write_text("\n".join(moved_lines) + "\nTER\nEND\n", encoding="utf-8")
        ligand = Chem.SDMolSupplier(str(sources["ligand"]), removeHs=False)[0]
        assert ligand is not None
        expected = Chem.MolToInchiKey(Chem.MolFromSmiles(cfg["ligand_smiles"]))
        assert Chem.MolToInchiKey(Chem.RemoveHs(ligand)) == expected
        conf = ligand.GetConformer()
        for i, p in enumerate((conf.GetPositions() - origin) @ rotation.T):
            conf.SetAtomPosition(i, p)
        with Chem.SDWriter(str(args.output / "ligand_oriented.sdf")) as writer:
            writer.write(ligand)
        fixer = PDBFixer(filename=str(heavy), platform=openmm.Platform.getPlatformByName("CPU"))
        linker_records = [r for r in mapping if r["native_id"] is None]
        fixer.missingResidues = {
            (0, linker_records[0]["model_id"] - 1): ["ALA"] * len(linker_records)
        }
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=cfg["seed"])
        residues = list(fixer.topology.residues())
        assert len(residues) == len(mapping)
        for res, record in zip(residues, mapping, strict=True):
            res.id = str(record["model_id"])
            if record["native_id"] is None:
                assert res.name == "ALA"
        write_pdb(args.output / "pdbfixer_linked_heavy.pdb", fixer.topology, fixer.positions)
        linked_xyz = np.array(fixer.positions.value_in_unit(unit.angstrom))
        linked_indices = {(int(a.residue.id), a.name): a.index for a in fixer.topology.atoms()}
        linker_geometry_before = []
        for record in linker_records:
            i = record["model_id"]
            n, ca, c, cb = [linked_indices[(i, name)] for name in ("N", "CA", "C", "CB")]
            signed = float(
                np.dot(
                    np.cross(linked_xyz[n] - linked_xyz[ca], linked_xyz[c] - linked_xyz[ca]),
                    linked_xyz[cb] - linked_xyz[ca],
                )
            )
            linker_geometry_before.append({"model_id": i, "signed_volume_A3": signed})
            linked_xyz[cb] = l_alanine_cb(linked_xyz[n], linked_xyz[ca], linked_xyz[c])
        save(args.output / "linker_geometry_before_cb_reconstruction.json", linker_geometry_before)
        fixer.positions = unit.Quantity(linked_xyz, unit.angstrom)
        write_pdb(args.output / "linked_heavy.pdb", fixer.topology, fixer.positions)
        original = {
            (int(s[22:26]), s[12:16].strip()): np.array(
                [float(s[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            )
            for s in moved_lines
        }
        kept_errors = []
        xyz = fixer.positions.value_in_unit(unit.angstrom)
        for a in fixer.topology.atoms():
            key = (int(a.residue.id), a.name)
            if key in original:
                kept_errors.append(np.linalg.norm(np.array(xyz[a.index]) - original[key]))
        assert max(kept_errors) < 1e-6
        ff = ForceField(*cfg["forcefields"])
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(ff, pH=7.4, platform=openmm.Platform.getPlatformByName("CPU"))
        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        indices_h = {(int(a.residue.id), a.name): a.index for a in modeller.topology.atoms()}
        quartets = [
            [indices_h[(r["model_id"], name)] for name in ("N", "CA", "C", "CB")]
            for r in mapping
            if (r["model_id"], "CB") in indices_h
        ]
        chiral_index = system.addForce(chiral_minimization_barrier(quartets))
        result["construct_minimization_only_chiral_barrier"] = True
        restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", 2500.0)
        for key in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(key)
        model_to_native = {r["model_id"]: r["native_id"] for r in mapping}
        pos_nm = modeller.positions.value_in_unit(unit.nanometer)
        for a in modeller.topology.atoms():
            if a.name in {"N", "CA", "C", "O"} and model_to_native[int(a.residue.id)] is not None:
                restraint.addParticle(a.index, list(pos_nm[a.index]))
        system.addForce(restraint)
        integrator = openmm.LangevinMiddleIntegrator(
            310 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond
        )
        integrator.setRandomNumberSeed(cfg["seed"])
        sim = Simulation(
            modeller.topology,
            system,
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),
            {"DeviceIndex": "0", "Precision": "mixed"},
        )
        sim.context.setPositions(modeller.positions)
        initial = float(
            sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        sim.minimizeEnergy(
            tolerance=5 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=2500
        )
        prepared = np.array(
            sim.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstrom)
        )
        hydrogen_repairs = []
        for record in mapping:
            i = record["model_id"]
            if (i, "CB") not in indices_h:
                continue
            n, ca, c, cb, ha = [indices_h[(i, name)] for name in ("N", "CA", "C", "CB", "HA")]
            normal = np.cross(prepared[n] - prepared[ca], prepared[c] - prepared[ca])
            volume_h = float(np.dot(normal, prepared[ha] - prepared[ca]))
            if volume_h >= 0:
                previous = prepared[ha].copy()
                prepared[ha] = alpha_hydrogen_from_heavy(
                    prepared[n], prepared[ca], prepared[c], prepared[cb]
                )
                hydrogen_repairs.append(
                    {
                        "model_id": i,
                        "native_id": record["native_id"],
                        "before_HA_volume_A3": volume_h,
                        "hydrogen_displacement_A": float(np.linalg.norm(prepared[ha] - previous)),
                    }
                )
        result["hydrogen_repairs"] = hydrogen_repairs
        system.removeForce(chiral_index)
        sim.context.reinitialize(preserveState=False)
        sim.context.setPositions(prepared * unit.angstrom)
        # 不用手性辅助力的第二次最小化，避免以辅助力掩盖错误CA氢。
        sim.minimizeEnergy(
            tolerance=5 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=2500
        )
        result["no_chiral_auxiliary_force_in_final_minimization"] = True
        state = sim.context.getState(getPositions=True, getEnergy=True, getForces=True)
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        assert np.isfinite(xyz).all()
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        assert np.isfinite(forces).all()
        write_pdb(args.output / "receptor_minimized.pdb", modeller.topology, state.getPositions())
        by_res = {(int(a.residue.id), a.name): a.index for a in modeller.topology.atoms()}
        peptide_lengths = [
            float(np.linalg.norm(xyz[by_res[(i, "C")]] - xyz[by_res[(i + 1, "N")]]))
            for i in range(1, len(mapping))
        ]
        assert min(peptide_lengths) > 1.1 and max(peptide_lengths) < 1.6
        disulfides = [
            [
                int(a.residue.id),
                int(b.residue.id),
                float(np.linalg.norm(xyz[a.index] - xyz[b.index])),
            ]
            for a, b in modeller.topology.bonds()
            if a.name == b.name == "SG"
        ]
        native_disulfides = [
            {"native_pair": [model_to_native[a], model_to_native[b]], "distance_A": d}
            for a, b, d in disulfides
        ]
        assert {frozenset(d["native_pair"]) for d in native_disulfides} == {
            frozenset((11, 166)),
            frozenset((93, 168)),
        }
        assert all(1.8 < d["distance_A"] < 2.3 for d in native_disulfides)
        # 比較四面体体积符号，拒绝连接肽ALA翻成立体镜像。
        signs = []
        for r in mapping:
            i = r["model_id"]
            if (i, "CB") in by_res:
                v = [xyz[by_res[(i, n)]] - xyz[by_res[(i, "CA")]] for n in ("N", "C", "CB")]
                signs.append((r["role"], float(np.dot(np.cross(v[0], v[1]), v[2]))))
                ha = xyz[by_res[(i, "HA")]] - xyz[by_res[(i, "CA")]]
                hydrogen_volume = float(np.dot(np.cross(v[0], v[1]), ha))
                hydrogen_angle = np.rad2deg(
                    np.arccos(
                        np.clip(
                            np.dot(v[2], ha) / (np.linalg.norm(v[2]) * np.linalg.norm(ha)), -1, 1
                        )
                    )
                )
                assert hydrogen_volume < 0 and 80 < hydrogen_angle < 140
        native_sign = np.sign(np.median([v for role, v in signs if role == "retained_native"]))
        save(
            args.output / "chirality_checks.json",
            {"native_sign": float(native_sign), "signed_volumes": signs},
        )
        assert all(np.sign(v) == native_sign and abs(v) > 1 for _, v in signs)
        result.update(
            status="construct_geometry_passed",
            residue_count=len(mapping),
            synthetic_residues=len(linker_records),
            initial_protein_energy_kJ_mol=initial,
            minimized_protein_energy_kJ_mol=float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ),
            peptide_CN_range_A=[min(peptide_lengths), max(peptide_lengths)],
            disulfides=native_disulfides,
            retained_initial_atom_max_delta_A=float(max(kept_errors)),
            max_force_kJ_mol_nm=float(np.linalg.norm(forces, axis=1).max()),
            chirality_checked_residues=len(signs),
            openmm_version=openmm.__version__,
        )
        after = [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sources.values()
        ]
        assert before == after
        save(args.output / "manifest.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        result.update(status="failed_construct", error_type=type(error).__name__, error=str(error))
        save(args.output / "manifest.json", result)
        raise


if __name__ == "__main__":
    main()
