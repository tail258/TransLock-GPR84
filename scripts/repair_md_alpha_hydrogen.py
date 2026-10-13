"""修复已定位的错误CA氢朝向，并用无手性辅助力的最小化核查。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import openmm
from openmm import unit
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, Simulation

from translock_screening.membrane_construct import alpha_hydrogen_from_heavy


def save(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source / "manifest.json").read_text())
    assert source["status"] == "construct_geometry_passed"
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "residue_map.json",
        "orientation.json",
        "ligand_oriented.sdf",
        "config.yaml",
        "retained_oriented_heavy.pdb",
        "input_receptor.pdb",
        "input_uniprot.json",
        "input_ligand.sdf",
    ):
        shutil.copyfile(args.source / name, args.output / name)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/membrane_construct.py",
        args.output / "membrane_construct.py",
    )
    mapping = json.loads((args.source / "residue_map.json").read_text())
    model_to_native = {r["model_id"]: r["native_id"] for r in mapping}
    pdb = PDBFile(str(args.source / "receptor_minimized.pdb"))
    xyz = np.array(pdb.positions.value_in_unit(unit.angstrom))
    by = {(int(a.residue.id), a.name): a.index for a in pdb.topology.atoms()}
    records = []
    for r in mapping:
        i = r["model_id"]
        if (i, "CB") not in by:
            continue
        n, ca, c, cb, ha = [by[(i, name)] for name in ("N", "CA", "C", "CB", "HA")]
        normal = np.cross(xyz[n] - xyz[ca], xyz[c] - xyz[ca])
        vol_cb = float(np.dot(normal, xyz[cb] - xyz[ca]))
        vol_ha = float(np.dot(normal, xyz[ha] - xyz[ca]))
        assert vol_cb > 1
        if vol_ha >= 0:
            before = xyz[ha].copy()
            xyz[ha] = alpha_hydrogen_from_heavy(xyz[n], xyz[ca], xyz[c], xyz[cb])
            records.append(
                {
                    "model_id": i,
                    "native_id": r["native_id"],
                    "before_HA_signed_volume_A3": vol_ha,
                    "hydrogen_displacement_A": float(np.linalg.norm(xyz[ha] - before)),
                    "after_HA_signed_volume_A3": float(np.dot(normal, xyz[ha] - xyz[ca])),
                }
            )
    if not records:
        raise ValueError("输入没有已定位的CA氢错误，禁止无理由重做")
    result = dict(source)
    result.update(
        status="repairing_alpha_hydrogen",
        source_construct=str(args.source),
        hydrogen_repairs=records,
        construct_minimization_only_chiral_barrier=False,
        scope="13ALA近似；已定位加氢几何修复及无手性辅助力最小化，非MD验收",
    )
    save(args.output / "manifest.json", result)
    try:
        ff = ForceField(*source["config"]["forcefields"])
        system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", 2500)
        for name in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)
        original = {}
        for s in (args.source / "retained_oriented_heavy.pdb").read_text().splitlines():
            if s.startswith("ATOM  "):
                original[(int(s[22:26]), s[12:16].strip())] = (
                    np.array([float(s[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]) / 10
                )
        for atom in pdb.topology.atoms():
            key = (int(atom.residue.id), atom.name)
            if atom.name in {"N", "CA", "C", "O"} and key in original:
                restraint.addParticle(atom.index, original[key].tolist())
        system.addForce(restraint)
        assert not any(isinstance(f, openmm.CustomCompoundBondForce) for f in system.getForces())
        integrator = openmm.LangevinMiddleIntegrator(310, 1, 0.001)
        integrator.setRandomNumberSeed(source["config"]["seed"])
        sim = Simulation(
            pdb.topology,
            system,
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),
            {"DeviceIndex": "0", "Precision": "mixed"},
        )
        sim.context.setPositions(xyz * unit.angstrom)
        before = sim.context.getState(getEnergy=True).getPotentialEnergy()
        sim.minimizeEnergy(
            tolerance=5 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=2500
        )
        state = sim.context.getState(getPositions=True, getEnergy=True, getForces=True)
        final = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        assert np.isfinite(final).all() and np.isfinite(forces).all()
        with (args.output / "receptor_minimized.pdb").open("w", encoding="utf-8") as h:
            PDBFile.writeFile(pdb.topology, state.getPositions(), h, keepIds=True)
        geometry = []
        for r in mapping:
            i = r["model_id"]
            if (i, "CB") not in by:
                continue
            n, ca, c, cb, ha = [final[by[(i, name)]] for name in ("N", "CA", "C", "CB", "HA")]
            normal = np.cross(n - ca, c - ca)
            w, h = cb - ca, ha - ca
            geometry.append(
                {
                    "model_id": i,
                    "native_id": r["native_id"],
                    "CB_signed_volume_A3": float(np.dot(normal, w)),
                    "HA_signed_volume_A3": float(np.dot(normal, h)),
                    "CB_CA_HA_angle_deg": float(
                        np.rad2deg(
                            np.arccos(
                                np.clip(
                                    np.dot(w, h) / (np.linalg.norm(w) * np.linalg.norm(h)), -1, 1
                                )
                            )
                        )
                    ),
                }
            )
        save(args.output / "alpha_tetrahedral_checks.json", geometry)
        assert all(
            r["CB_signed_volume_A3"] > 1
            and r["HA_signed_volume_A3"] < 0
            and 80 < r["CB_CA_HA_angle_deg"] < 140
            for r in geometry
        )
        cn = [
            float(np.linalg.norm(final[by[(i, "C")]] - final[by[(i + 1, "N")]]))
            for i in range(1, len(mapping))
        ]
        assert min(cn) > 1.1 and max(cn) < 1.6
        ss = [
            {
                "native_pair": [
                    model_to_native[int(a.residue.id)],
                    model_to_native[int(b.residue.id)],
                ],
                "distance_A": float(np.linalg.norm(final[a.index] - final[b.index])),
            }
            for a, b in pdb.topology.bonds()
            if a.name == b.name == "SG"
        ]
        assert all(1.8 < s["distance_A"] < 2.3 for s in ss)
        result.update(
            status="construct_geometry_passed",
            disulfides=ss,
            initial_protein_energy_kJ_mol=before.value_in_unit(unit.kilojoule_per_mole),
            minimized_protein_energy_kJ_mol=state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            ),
            peptide_CN_range_A=[min(cn), max(cn)],
            max_force_kJ_mol_nm=float(np.linalg.norm(forces, axis=1).max()),
            alpha_CB_min_volume_A3=min(r["CB_signed_volume_A3"] for r in geometry),
            alpha_HA_max_volume_A3=max(r["HA_signed_volume_A3"] for r in geometry),
            no_chiral_auxiliary_force=True,
        )
        save(args.output / "manifest.json", result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "status",
                        "hydrogen_repairs",
                        "alpha_CB_min_volume_A3",
                        "alpha_HA_max_volume_A3",
                        "max_force_kJ_mol_nm",
                    ]
                }
            ),
            flush=True,
        )
    except Exception as error:
        result.update(status="failed_construct", error_type=type(error).__name__, error=str(error))
        save(args.output / "manifest.json", result)
        raise


if __name__ == "__main__":
    main()
