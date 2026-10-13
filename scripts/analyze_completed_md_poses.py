"""解释已完成轨迹的整体位移、核心变化与接触；只做CPU后处理。"""

import argparse
import json
import platform
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from translock_screening.dcd_reader import read_openmm_dcd
from translock_screening.md_diagnostics import fit_to_reference, minimum_image
from translock_screening.md_interpretation import (
    actual_medoid_index,
    coordinate_only_molecule,
    displacement_metrics,
    murcko_atom_indices,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_path(path):
    value = str(path).replace("\\", "/")
    if len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + value[2:]
    return Path(value)


def state_positions(path):
    state = ET.parse(path).getroot()
    xyz = np.array([[float(p.get(k)) for k in ("x", "y", "z")] for p in state.find("Positions")])
    box = np.array(
        [
            [float(state.find("PeriodicBoxVectors/" + axis).get(k)) for k in ("x", "y", "z")]
            for axis in ("A", "B", "C")
        ]
    )
    assert float(state.get("time")) == 0
    assert float(state.find("Parameters").get("k")) == 0
    assert np.isfinite(xyz).all() and np.linalg.det(box) > 0
    return xyz, box


def input_record(path):
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def prepare_context(bundle, ligand_id):
    folder = bundle / "inputs" / ligand_id
    names = [
        folder / "warmup/atom_map.json",
        folder / "parameters/ligand.sdf",
        folder / "parameters/atom_parameters.json",
        folder / "construct/residue_map.json",
        folder / "construct/retained_oriented_heavy.pdb",
        folder / "parameters/manifest.json",
    ]
    atoms = load_json(names[0])
    ligand_h = Chem.SDMolSupplier(str(names[1]), removeHs=False)[0]
    ligand = Chem.RemoveHs(ligand_h)
    params = load_json(names[2])
    mapping = load_json(names[3])
    pm = load_json(names[5])
    assert Chem.MolToInchiKey(ligand) == pm["inchikey"]
    assert [a.GetAtomicNum() for a in ligand_h.GetAtoms()] == [a["atomic_number"] for a in params]
    protein = [a for a in atoms if a["chain"] == "R"]
    assert [a["index"] for a in protein] == list(range(len(protein)))
    heavy_local = np.array([i for i, atom in enumerate(params) if atom["atomic_number"] != 1])
    lig_indices = len(protein) + heavy_local
    ligand_rows = atoms[len(protein) : len(protein) + len(params)]
    assert len({(a["chain"], a["residue_id"], a["residue"]) for a in ligand_rows}) == 1
    assert ligand_rows[0]["chain"] != "R"
    assert [a["name"] for a in ligand_rows] == [a["name"] for a in params]
    assert [a["element"] for a in ligand_rows] == [a.GetSymbol() for a in ligand_h.GetAtoms()]
    reference = np.array([params[i]["position_A"] for i in heavy_local])
    assert np.allclose(ligand.GetConformer().GetPositions(), reference, atol=1e-4)
    by_name = {(int(a["residue_id"]), a["name"]): a["index"] for a in protein}
    model_native = {r["model_id"]: r["native_id"] for r in mapping}
    native_model = {v: k for k, v in model_native.items() if v is not None}
    ca_reference, receptor_positions, receptor_residues = {}, [], []
    for line in names[4].read_text().splitlines():
        if not line.startswith("ATOM  "):
            continue
        mid = int(line[22:26])
        point = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        if line[12:16].strip() == "CA":
            ca_reference[mid] = point
        if line[76:78].strip() != "H":
            receptor_positions.append(point)
            native = model_native[mid]
            receptor_residues.append(
                f"{line[17:20]}{native}" if native is not None else f"synthetic_ALA{mid}"
            )
    ca_models = sorted(ca_reference)
    ca_indices = np.array([by_name[(mid, "CA")] for mid in ca_models])
    reference_ca = np.array([ca_reference[mid] for mid in ca_models]) / 10
    pocket_ids = [100, 101, 169, 170, 172, 183, 187, 332, 335, 336, 339, 340]
    pocket_ca = np.array([by_name[(native_model[n], "CA")] for n in pocket_ids])
    distances = np.linalg.norm(np.array(receptor_positions)[:, None] - reference[None], axis=2)
    original_contacts = sorted(
        {receptor_residues[i] for i in np.where(distances.min(axis=1) <= 4)[0]}
    )
    core = murcko_atom_indices(ligand)
    return dict(
        atoms=atoms,
        ligand=ligand,
        heavy_indices=lig_indices,
        reference=reference,
        reference_ca=reference_ca,
        ca_indices=ca_indices,
        pocket_ca=pocket_ca,
        core=core,
        original_contacts=original_contacts,
        input_files=names,
        definition={
            "ligand_id": ligand_id,
            "inchikey": pm["inchikey"],
            "smiles": Chem.MolToSmiles(ligand),
            "core_atom_indices": core.tolist(),
            "heavy_atom_count": len(reference),
            "core_atom_count": len(core),
            "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in ligand.GetBonds()],
            "atomic_numbers": [a.GetAtomicNum() for a in ligand.GetAtoms()],
            "atom_names": [params[i]["name"] for i in heavy_local],
            "native_ca_residues": [model_native[mid] for mid in ca_models],
            "original_contacts_4A": original_contacts,
        },
    )


def align_ligand(xyz, vectors, context, reference_ca=None):
    assert len(xyz) == len(context["atoms"])
    assert np.allclose(vectors, np.diag(np.diag(vectors)), atol=1e-8)
    lig = xyz[context["heavy_indices"]]
    delta = lig.mean(axis=0) - xyz[context["pocket_ca"]].mean(axis=0)
    whole = lig + minimum_image(delta, np.diag(vectors)) - delta
    ca = context["reference_ca"] if reference_ca is None else reference_ca
    rotation, a, b, _ = fit_to_reference(xyz[context["ca_indices"]], ca)
    return ((whole - a) @ rotation + b) * 10


def statistics(values):
    v = np.array(values, dtype=float)
    return {
        "median": float(np.median(v)),
        "minimum": float(v.min()),
        "maximum": float(v.max()),
        "first_100_mean": float(v[:100].mean()),
        "last_100_mean": float(v[-100:].mean()),
        "final": float(v[-1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    records = load_json(args.provenance)["records"]
    assert len(records) == len({(r["ligand_id"], r["seed"]) for r in records}) == 9
    if args.limit:
        records = records[: args.limit]
    contexts, summaries, snapshots, contact_rows = {}, [], [input_record(args.provenance)], []
    for record in records:
        ligand_id, seed = record["ligand_id"], record["seed"]
        run = local_path(record["local_raw_copy"])
        if ligand_id not in contexts:
            contexts[ligand_id] = prepare_context(args.bundle, ligand_id)
            context = contexts[ligand_id]
            snapshots.extend(input_record(p) for p in context["input_files"])
            folder = args.output / ligand_id
            folder.mkdir()
            write_json(folder / "ligand_definition.json", context["definition"])
            np.savez_compressed(
                folder / "reference_geometry.npz",
                ligand=context["reference"],
                native_ca=context["reference_ca"] * 10,
            )
        context = contexts[ligand_id]
        output = args.output / ligand_id / str(seed)
        output.mkdir()
        names = [
            run / n
            for n in (
                "manifest.json",
                "initial_state.xml",
                "trajectory.dcd",
                "diagnostics.jsonl",
                "analysis_v1/frames.csv",
                "analysis_v1/independent_readback.json",
            )
        ]
        snapshots.extend(input_record(p) for p in names)
        manifest = load_json(names[0])
        assert manifest["status"] == "complete_fixed_length"
        assert manifest["base_seed"] == seed and manifest["ligand_id"] == ligand_id
        assert manifest["target_time_ps"] == 5000
        initial, box = state_positions(names[1])
        initial_aligned = align_ligand(initial, box, context)
        initial_native = align_ligand(initial, box, context, initial[context["ca_indices"]])
        online = [json.loads(line) for line in names[3].read_text().splitlines()]
        previous = pd.read_csv(names[4])
        readback = load_json(names[5])
        assert len(online) == 501 and len(previous) == 500 and readback["status"] == "passed"
        arrays, rows = [], []
        for i, frame in enumerate(read_openmm_dcd(names[2])):
            assert frame.step == (i + 1) * 5000
            assert online[i + 1]["geometry_passed"]
            aligned = align_ligand(frame.positions_nm, frame.box_vectors_nm, context)
            start_aligned = align_ligand(
                frame.positions_nm, frame.box_vectors_nm, context, initial[context["ca_indices"]]
            )
            row = {"time_ps": (i + 1) * 10}
            for reference_name, a, b in (
                ("original", aligned, context["reference"]),
                ("replica_start", start_aligned, initial_native),
            ):
                row.update(
                    {
                        f"{reference_name}_{k}": v
                        for k, v in displacement_metrics(a, b, context["core"]).items()
                    }
                )
            assert (
                abs(row["original_total_A"] - previous.iloc[i].ligand_RMSD_to_original_pose_A)
                < 2e-4
            )
            assert (
                abs(row["replica_start_total_A"] - previous.iloc[i].ligand_RMSD_to_replica_start_A)
                < 2e-4
            )
            rows.append(row)
            arrays.append(aligned)
        assert len(rows) == 500
        table = pd.DataFrame(rows)
        table.to_csv(output / "displacements.csv", index=False)
        arrays = np.array(arrays)
        medoid = actual_medoid_index(arrays)
        late_medoid = 400 + actual_medoid_index(arrays[-100:])
        np.savez_compressed(
            output / "aligned_ligand_frames.npz",
            xyz_A=arrays,
            initial_A=initial_aligned,
            time_ps=np.arange(1, 501) * 10,
        )
        for label, index in (("representative", medoid), ("late_representative", late_medoid)):
            mol = coordinate_only_molecule(
                context["ligand"],
                arrays[index],
                f"{ligand_id}_{seed}_{label}_{(index + 1) * 10}ps",
            )
            mol.SetProp("reference_frame", "native_receptor_CA_aligned_to_original")
            with Chem.SDWriter(str(output / f"{label}.sdf")) as writer:
                writer.write(mol)
        counts = Counter(c for row in online[1:] for c in set(row["contacts_4A"]))
        late = Counter(c for row in online[-100:] for c in set(row["contacts_4A"]))
        early = Counter(c for row in online[1:101] for c in set(row["contacts_4A"]))
        assert counts == Counter({r["residue"]: r["frames"] for r in readback["contact_occupancy"]})
        starting_contacts = set(online[0]["contacts_4A"])
        union = set(counts) | starting_contacts | set(context["original_contacts"])
        for residue in sorted(union):
            contact_rows.append(
                {
                    "ligand_id": ligand_id,
                    "seed": seed,
                    "residue": residue,
                    "original_contact": residue in context["original_contacts"],
                    "replica_start_contact": residue in starting_contacts,
                    "occupancy": counts[residue] / 500,
                    "first_100_occupancy": early[residue] / 100,
                    "last_100_occupancy": late[residue] / 100,
                }
            )
        summary = {
            "ligand_id": ligand_id,
            "seed": seed,
            "source_run": str(run),
            "frames": 500,
            "original_readback_matched": True,
            "preparation_change": displacement_metrics(
                initial_aligned, context["reference"], context["core"]
            ),
            "metrics": {
                name: statistics(table[name])
                for name in table
                if name != "time_ps" and not table[name].isna().any()
            },
            "representative_frame": medoid + 1,
            "representative_time_ps": (medoid + 1) * 10,
            "late_representative_frame": late_medoid + 1,
            "original_contacts_4A": context["original_contacts"],
            "replica_start_contacts_4A": sorted(starting_contacts),
            "representative_contacts_4A": online[medoid + 1]["contacts_4A"],
            "late_representative_contacts_4A": online[late_medoid + 1]["contacts_4A"],
        }
        write_json(output / "summary.json", summary)
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "ligand": ligand_id,
                    "seed": seed,
                    "frames": 500,
                    "original_RMSD_median": summary["metrics"]["original_total_A"]["median"],
                    "centroid_median": summary["metrics"]["original_centroid_shift_A"]["median"],
                    "internal_median": summary["metrics"]["original_self_fit_internal_A"]["median"],
                }
            ),
            flush=True,
        )
    contacts = pd.DataFrame(contact_rows)
    contacts.to_csv(args.output / "contact_details.csv", index=False)
    flattened = []
    for summary in summaries:
        row = {k: summary[k] for k in ("ligand_id", "seed", "frames", "source_run")}
        for name, stats in summary["metrics"].items():
            row.update({f"{name}_{stat}": value for stat, value in stats.items()})
        flattened.append(row)
    pd.DataFrame(flattened).to_csv(args.output / "replicate_displacements.csv", index=False)
    for entry in snapshots:
        assert input_record(Path(entry["path"])) == entry, "分析期间源文件改变"
    write_json(args.output / "summaries.json", summaries)
    write_json(args.output / "input_manifest.json", snapshots)
    write_json(
        args.output / "manifest.json",
        {
            "status": "complete_postprocessing",
            "created_utc": datetime.now(UTC).isoformat(),
            "replicates": len(summaries),
            "frames": 500 * len(summaries),
            "wall_seconds": time.monotonic() - start,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "scientific_simulation_started": False,
            "hashing": False,
            "scope": "已有短程MD的事后描述性解释；不改变原提名顺序或验证活性",
        },
    )
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    for name in ("md_interpretation.py", "md_diagnostics.py", "dcd_reader.py"):
        shutil.copyfile(
            Path(__file__).parents[1] / "src/translock_screening" / name, snapshot / name
        )


if __name__ == "__main__":
    main()
