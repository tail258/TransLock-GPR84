"""按冻结协议补充氢键、非极性接触、局部RMSF、姿势聚类和二面角。"""

import argparse
import io
import json
import shutil
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_completed_md_poses import input_record, local_path, prepare_context
from openmm.app import PDBFile
from rdkit import Chem

from translock_screening.dcd_reader import read_openmm_dcd
from translock_screening.md_diagnostics import fit_to_reference, minimum_image
from translock_screening.md_interactions import (
    circular_statistics,
    dihedrals,
    gromos_clusters,
    hbond_geometry,
    ligand_features,
    pairwise_pose_distances,
    rmsf,
    torsion_definitions,
)

ORDER = ["GLPG1205", "CHEMBL120691", "CHEMBL1086046"]
SEEDS = [20261011, 20261012, 20261013]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def context(bundle, cid, existing):
    c = prepare_context(bundle, cid)
    folder = bundle / "inputs" / cid
    pdbpath = folder / "warmup/restart_v1/system.pdb"
    text = pdbpath.read_text()
    protein_text = "\n".join(
        line
        for line in text.splitlines()
        if line.startswith(("ATOM  ", "HETATM")) and line[21] == "R"
    )
    protein = [a for a in c["atoms"] if a["chain"] == "R"]
    topology = PDBFile(io.StringIO(protein_text + "\nTER\nEND\n")).topology
    pa = list(topology.atoms())
    assert len(pa) == len(protein)
    assert [(a.name, a.element.symbol) for a in pa] == [(a["name"], a["element"]) for a in protein]
    adjacency = {i: [] for i in range(len(pa))}
    for a, b in topology.bonds():
        adjacency[a.index].append(b.index)
        adjacency[b.index].append(a.index)
    assert all(len(adjacency[a.index]) == 1 for a in pa if a.element.symbol == "H")
    mapping = json.loads((folder / "construct/residue_map.json").read_text())
    native = {r["model_id"]: r["native_id"] for r in mapping}
    labels = {
        i: (
            f"{a['residue']}{native[int(a['residue_id'])]}"
            if native[int(a["residue_id"])] is not None
            else f"synthetic_ALA{a['residue_id']}"
        )
        for i, a in enumerate(protein)
    }
    protein_dh = [
        (a.index, j)
        for a in pa
        if a.element.symbol in ("N", "O")
        for j in adjacency[a.index]
        if pa[j].element.symbol == "H"
    ]
    acceptor_names = {
        "ASP": {"OD1", "OD2"},
        "GLU": {"OE1", "OE2"},
        "ASN": {"OD1"},
        "GLN": {"OE1"},
        "SER": {"OG"},
        "THR": {"OG1"},
        "TYR": {"OH"},
    }
    acceptors = []
    for a in pa:
        attached_h = [j for j in adjacency[a.index] if pa[j].element.symbol == "H"]
        if a.name in ("O", "OXT"):
            eligible = not attached_h
        elif a.residue.name in ("HIS", "HID", "HIE", "HIP") and a.name in ("ND1", "NE2"):
            eligible = not attached_h
        else:
            eligible = a.name in acceptor_names.get(a.residue.name, set())
            if a.residue.name in ("ASP", "GLU") and attached_h:
                eligible = False
        if eligible:
            acceptors.append(a.index)
    protein_c = [
        a.index
        for a in pa
        if a.element.symbol == "C"
        and adjacency[a.index]
        and all(pa[j].element.symbol in ("C", "H") for j in adjacency[a.index])
    ]
    ligand_h = Chem.SDMolSupplier(str(folder / "parameters/ligand.sdf"), removeHs=False)[0]
    features = ligand_features(ligand_h)
    assert not features["charged_atoms"], "出现形式带电配体，需要另行核对盐桥分组，不能静默略过"
    offset = len(pa)
    ligand_dh = [(d + offset, h + offset) for d, h in features["donor_hydrogens"]]
    ligand_a = [a + offset for a in features["Acceptor"]]
    ligand_c = [a + offset for a in features["nonpolar_carbons"]]
    triples = [(d, h, a) for (d, h), a in product(protein_dh, ligand_a)]
    triples += [(d, h, a) for (d, h), a in product(ligand_dh, acceptors)]
    hydro_pairs = list(product(protein_c, ligand_c))
    hb_keys = sorted({(d, a) for d, _, a in triples})
    hb_lookup = {pair: i for i, pair in enumerate(hb_keys)}
    hb_group = np.array([hb_lookup[(d, a)] for d, _, a in triples])
    hydro_keys = sorted({(labels[p], ligand_index) for p, ligand_index in hydro_pairs})
    hydro_lookup = {key: i for i, key in enumerate(hydro_keys)}
    hydro_group = np.array(
        [hydro_lookup[(labels[p], ligand_index)] for p, ligand_index in hydro_pairs]
    )
    pca = [a.index for a in pa if a.name == "CA"]
    native_ca = [native[int(protein[i]["residue_id"])] for i in pca]
    ligand_definition = json.loads((existing / cid / "ligand_definition.json").read_text())
    torsions = torsion_definitions(c["ligand"], c["core"])
    for t in torsions:
        t["atom_names"] = [ligand_definition["atom_names"][i] for i in t["atoms"]]
    # 口袋选择只读取原输入，三分子并集稍后固定，不按轨迹波动挑选。
    receptor_points, pocket_ids = [], []
    for line in (folder / "construct/retained_oriented_heavy.pdb").read_text().splitlines():
        if line.startswith("ATOM  ") and line[76:78].strip() != "H":
            receptor_points.append([float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))])
            pocket_ids.append(native[int(line[22:26])])
    dist = np.linalg.norm(np.array(receptor_points)[:, None] - c["reference"][None], axis=2)
    pocket = sorted(
        {pocket_ids[i] for i in np.where(dist.min(axis=1) <= 6)[0] if pocket_ids[i] is not None}
    )
    c.update(
        protein_labels=labels,
        protein_ca=pca,
        native_ca=native_ca,
        triples=np.array(triples, int),
        hb_keys=hb_keys,
        hb_group=hb_group,
        hydro_pairs=np.array(hydro_pairs, int),
        hydro_keys=hydro_keys,
        hydro_group=hydro_group,
        torsions=torsions,
        pocket=pocket,
        protein_count=offset,
        ligand_atom_count=ligand_h.GetNumAtoms(),
    )
    chemistry = {
        "ligand_id": cid,
        "ligand_features_explicit_H_indices": features,
        "protein_donor_H_pairs": protein_dh,
        "protein_acceptors": acceptors,
        "protein_nonpolar_C": protein_c,
        "protein_labels": labels,
        "hb_heavy_pairs": hb_keys,
        "hb_triples": triples,
        "hydrophobic_atom_pairs": hydro_pairs,
        "hydrophobic_groups": hydro_keys,
        "torsions": torsions,
        "protein_CA_indices": pca,
        "native_CA_ids": native_ca,
        "salt_bridges": "not_applicable_for_actual_neutral_ligand_with_no_formal_charged_atom",
        "partial_charges_are_not_formal_charge": True,
    }
    c["chemistry"] = chemistry
    c["input_files"].append(pdbpath)
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-frames", type=int)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--prior-seconds", type=float, default=0)
    args = parser.parse_args()
    assert args.max_seconds + args.prior_seconds <= 3000  # 留足独立核对与绘图预算。
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status, error = "running", None
    manifest = {
        "started_utc": datetime.now(UTC).isoformat(),
        "pilot_frames": args.pilot_frames,
        "prior_compute_seconds": args.prior_seconds,
        "max_seconds": args.max_seconds,
        "new_MD": False,
        "recomputed_old_RMSD": False,
    }

    def guard():
        if time.monotonic() - start > args.max_seconds:
            raise TimeoutError("达到冻结后处理预算")

    try:
        records = json.loads(args.provenance.read_text())["records"]
        records = sorted(records, key=lambda r: (ORDER.index(r["ligand_id"]), r["seed"]))
        if args.pilot_frames:
            records = [r for r in records if r["ligand_id"] == "CHEMBL1086046"][:1]
        contexts = {cid: context(args.bundle, cid, args.existing) for cid in ORDER}
        pocket = sorted(set().union(*(set(c["pocket"]) for c in contexts.values())))
        dump(
            args.output / "pocket_definition.json",
            {"native_residue_union_6A": pocket, "source": "three_original_poses"},
        )
        snapshots = [input_record(args.provenance)]
        for cid, c in contexts.items():
            (args.output / cid).mkdir()
            dump(args.output / cid / "chemical_definitions.json", c["chemistry"])
            snapshots.extend(input_record(p) for p in c["input_files"])
        summaries, hb_summary, hyd_summary, rmsf_rows, torsion_rows = [], [], [], [], []
        for record in records:
            guard()
            cid, seed = record["ligand_id"], record["seed"]
            c = contexts[cid]
            source = local_path(record["local_raw_copy"])
            traj_path = source / "trajectory.dcd"
            cached_path = args.existing / cid / str(seed) / "aligned_ligand_frames.npz"
            cached = np.load(cached_path)["xyz_A"]
            assert len(cached) == 500
            snapshots.extend([input_record(traj_path), input_record(cached_path)])
            out = args.output / cid / str(seed)
            out.mkdir()
            hb3, hb35, hyd4, hyd45, ca_frames, events, independent_samples = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            read_start = time.monotonic()
            for f, frame in enumerate(read_openmm_dcd(traj_path)):
                guard()
                assert frame.step == (f + 1) * 5000
                assert np.allclose(
                    frame.box_vectors_nm, np.diag(np.diag(frame.box_vectors_nm)), atol=1e-8
                )
                xyz, box = frame.positions_nm * 10, np.diag(frame.box_vectors_nm) * 10
                distance, angle = hbond_geometry(xyz, c["triples"], box)
                permissive = (distance <= 3.5) & (angle >= 150)
                strict = (distance <= 3.0) & (angle >= 150)
                for target, mask in ((hb3, strict), (hb35, permissive)):
                    row = np.zeros(len(c["hb_keys"]), bool)
                    row[c["hb_group"][mask]] = True
                    target.append(row)
                for i in np.flatnonzero(permissive):
                    d, h, a = c["triples"][i]
                    events.append(
                        (
                            f + 1,
                            (f + 1) * 10,
                            int(d),
                            int(h),
                            int(a),
                            distance[i],
                            angle[i],
                            bool(strict[i]),
                        )
                    )
                pairs = c["hydro_pairs"]
                hd = np.linalg.norm(minimum_image(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], box), axis=1)
                for target, cutoff in ((hyd4, 4.0), (hyd45, 4.5)):
                    row = np.zeros(len(c["hydro_keys"]), bool)
                    row[c["hydro_group"][hd <= cutoff]] = True
                    target.append(row)
                rotation, a, b, _ = fit_to_reference(
                    frame.positions_nm[c["ca_indices"]], c["reference_ca"]
                )
                ca_frames.append(((frame.positions_nm[c["protein_ca"]] - a) @ rotation + b) * 10)
                if f in (0, 249, 499):
                    independent_samples.append(
                        {
                            "frame": f + 1,
                            "xyz_A": xyz[: c["protein_count"] + c["ligand_atom_count"]],
                            "box_A": box,
                        }
                    )
                if args.pilot_frames and f + 1 >= args.pilot_frames:
                    break
            n = len(hb3)
            assert n == (args.pilot_frames or 500)
            hb3, hb35, hyd4, hyd45 = map(np.array, (hb3, hb35, hyd4, hyd45))
            ca_frames = np.array(ca_frames)
            np.savez_compressed(
                out / "interaction_series.npz",
                hb3=hb3,
                hb35=hb35,
                hyd4=hyd4,
                hyd45=hyd45,
                ca_xyz_A=ca_frames,
            )
            for sample in independent_samples:
                frame_number = sample.pop("frame")
                np.savez_compressed(out / f"geometry_sample_{frame_number}.npz", **sample)
            pd.DataFrame(
                events,
                columns=[
                    "frame",
                    "time_ps",
                    "donor",
                    "hydrogen",
                    "acceptor",
                    "DA_A",
                    "DHA_deg",
                    "strict_3A",
                ],
            ).to_csv(out / "hbond_events.csv", index=False)
            for i, (d, a) in enumerate(c["hb_keys"]):
                if not hb35[:, i].any():
                    continue
                receptor = d if d < c["protein_count"] else a
                ligand = a if d < c["protein_count"] else d
                hb_summary.append(
                    dict(
                        ligand_id=cid,
                        seed=seed,
                        donor=d,
                        acceptor=a,
                        residue=c["protein_labels"][receptor],
                        receptor_atom=c["atoms"][receptor]["name"],
                        ligand_atom=c["atoms"][ligand]["name"],
                        occupancy_3A=float(hb3[:, i].mean()),
                        occupancy_35A=float(hb35[:, i].mean()),
                        first_100=float(hb3[:100, i].mean()),
                        last_100=float(hb3[-100:, i].mean()),
                    )
                )
            for residue in sorted({k[0] for k in c["hydro_keys"]}):
                indices = [i for i, k in enumerate(c["hydro_keys"]) if k[0] == residue]
                strict_series, wide_series = (
                    hyd4[:, indices].any(axis=1),
                    hyd45[:, indices].any(axis=1),
                )
                if wide_series.any():
                    hyd_summary.append(
                        dict(
                            ligand_id=cid,
                            seed=seed,
                            residue=residue,
                            occupancy_4A=float(strict_series.mean()),
                            occupancy_45A=float(wide_series.mean()),
                            first_100=float(strict_series[:100].mean()),
                            last_100=float(strict_series[-100:].mean()),
                        )
                    )
            for group, points, indices in (
                ("protein_CA", ca_frames, c["protein_ca"]),
                ("ligand_heavy", cached[:n], c["heavy_indices"]),
            ):
                values = rmsf(points)
                for index, value in zip(indices, values, strict=True):
                    residue = c["protein_labels"].get(int(index), cid)
                    native_id = (
                        c["native_ca"][c["protein_ca"].index(index)]
                        if group == "protein_CA"
                        else None
                    )
                    rmsf_rows.append(
                        dict(
                            ligand_id=cid,
                            seed=seed,
                            group=group,
                            atom_index=int(index),
                            atom_name=c["atoms"][index]["name"],
                            residue=residue,
                            native_id=native_id,
                            pocket=native_id in pocket,
                            synthetic=residue.startswith("synthetic"),
                            RMSF_A=float(value),
                        )
                    )
            angles = dihedrals(cached[:n], [t["atoms"] for t in c["torsions"]])
            angle_table = pd.DataFrame(
                angles, columns=[f"bond_{t['bond_index']}" for t in c["torsions"]]
            )
            angle_table.insert(0, "time_ps", np.arange(1, n + 1) * 10)
            angle_table.to_csv(out / "torsion_angles.csv", index=False)
            for i, definition in enumerate(c["torsions"]):
                row = dict(
                    ligand_id=cid,
                    seed=seed,
                    bond_index=definition["bond_index"],
                    atom_names="-".join(definition["atom_names"]),
                    core_boundary=definition["core_boundary"],
                )
                for label, values in (
                    ("full", angles[:, i]),
                    ("first100", angles[:100, i]),
                    ("last100", angles[-100:, i]),
                ):
                    row.update({f"{label}_{k}": v for k, v in circular_statistics(values).items()})
                torsion_rows.append(row)
            one = dict(
                ligand_id=cid,
                seed=seed,
                frames=n,
                processing_seconds=time.monotonic() - read_start,
                mean_hbond_pairs_3A=float(hb3.sum(axis=1).mean()),
                mean_hbond_pairs_35A=float(hb35.sum(axis=1).mean()),
                any_hbond_3A_fraction=float(hb3.any(axis=1).mean()),
                hydrophobic_groups=len(c["hydro_keys"]),
            )
            summaries.append(one)
            dump(out / "manifest.json", one)
            print(json.dumps(one), flush=True)
        pd.DataFrame(hb_summary).to_csv(args.output / "hbond_occupancies.csv", index=False)
        pd.DataFrame(hyd_summary).to_csv(args.output / "hydrophobic_occupancies.csv", index=False)
        pd.DataFrame(rmsf_rows).to_csv(args.output / "RMSF.csv", index=False)
        pd.DataFrame(torsion_rows).to_csv(args.output / "torsion_summary.csv", index=False)
        cluster_rows, cluster_counts = [], []
        if not args.pilot_frames:
            for cid in ORDER:
                guard()
                frames = np.concatenate(
                    [
                        np.load(args.existing / cid / str(seed) / "aligned_ligand_frames.npz")[
                            "xyz_A"
                        ]
                        for seed in SEEDS
                    ]
                )
                distances = pairwise_pose_distances(frames)
                np.savez_compressed(
                    args.output / cid / "pose_distance_matrix.npz", distances_A=distances
                )
                for cutoff in (1.0, 1.5, 2.0):
                    guard()
                    labels, centers = gromos_clusters(distances, cutoff)
                    for i, label in enumerate(labels):
                        cluster_rows.append(
                            dict(
                                ligand_id=cid,
                                cutoff_A=cutoff,
                                seed=SEEDS[i // 500],
                                frame=i % 500 + 1,
                                time_ps=(i % 500 + 1) * 10,
                                cluster=int(label),
                            )
                        )
                    for label, center in enumerate(centers, 1):
                        members = np.where(labels == label)[0]
                        assert distances[center, members].max() <= cutoff + 1e-9
                        cluster_counts.append(
                            dict(
                                ligand_id=cid,
                                cutoff_A=cutoff,
                                cluster=label,
                                frames=len(members),
                                center_seed=SEEDS[center // 500],
                                center_frame=center % 500 + 1,
                                max_center_distance_A=float(distances[center, members].max()),
                            )
                        )
                    np.savez_compressed(
                        args.output / cid / f"clusters_{cutoff:.1f}A.npz",
                        labels=labels,
                        center_indices=centers,
                        center_xyz_A=frames[centers],
                    )
            pd.DataFrame(cluster_rows).to_csv(args.output / "cluster_membership.csv", index=False)
            pd.DataFrame(cluster_counts).to_csv(args.output / "cluster_summary.csv", index=False)
        for snapshot in snapshots:
            assert input_record(Path(snapshot["path"])) == snapshot
        dump(args.output / "input_manifest.json", snapshots)
        dump(args.output / "replicate_summaries.json", summaries)
        manifest.update(
            frames=sum(r["frames"] for r in summaries),
            replicates=len(summaries),
            salt_bridges_applicable=False,
            pocket_residue_count=len(pocket),
        )
        status = "pilot_passed" if args.pilot_frames else "complete_postprocessing"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        manifest.update(
            status=status,
            error=error,
            wall_seconds=time.monotonic() - start,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        dump(args.output / "manifest.json", manifest)
        snapshot = args.output / "code_snapshot"
        snapshot.mkdir(exist_ok=True)
        shutil.copyfile(__file__, snapshot / Path(__file__).name)
        for name in ("md_interactions.py", "md_diagnostics.py", "dcd_reader.py"):
            shutil.copyfile(
                Path(__file__).parents[1] / "src/translock_screening" / name, snapshot / name
            )


if __name__ == "__main__":
    main()
