"""独立复核相互作用几何、去重占有率、RMSF、圆周角与聚类。"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    records = json.loads((args.analysis / "replicate_summaries.json").read_text())
    htable = pd.read_csv(args.analysis / "hbond_occupancies.csv")
    ytable = pd.read_csv(args.analysis / "hydrophobic_occupancies.csv")
    rtable = pd.read_csv(args.analysis / "RMSF.csv")
    sampled_frames = 0
    torsion_checks = 0
    for row in records:
        cid, seed, count = row["ligand_id"], row["seed"], row["frames"]
        folder = args.analysis / cid / str(seed)
        definition = json.loads((args.analysis / cid / "chemical_definitions.json").read_text())
        assert not definition["ligand_features_explicit_H_indices"]["charged_atoms"]
        series = np.load(folder / "interaction_series.npz")
        assert len(series["hb3"]) == count
        assert np.all(~series["hb3"] | series["hb35"])
        assert np.all(~series["hyd4"] | series["hyd45"])
        hb_keys = [tuple(p) for p in definition["hb_heavy_pairs"]]
        hyd_keys = [tuple(p) for p in definition["hydrophobic_groups"]]
        for path in folder.glob("geometry_sample_*.npz"):
            sample = np.load(path)
            xyz, box = sample["xyz_A"], sample["box_A"]
            frame = int(path.stem.split("_")[-1]) - 1
            strict, wide = set(), set()

            def delta(a, b, xyz=xyz, box=box):
                return (xyz[a] - xyz[b] + box / 2) % box - box / 2

            for d, h, a in definition["hb_triples"]:
                separation = float(np.sqrt(sum(delta(d, a) ** 2)))
                if separation > 3.5:
                    continue
                v, w = delta(d, h), delta(a, h)
                cos_angle = np.dot(v, w) / np.sqrt(np.dot(v, v) * np.dot(w, w))
                if cos_angle <= np.cos(np.radians(150)):
                    wide.add((d, a))
                    if separation <= 3:
                        strict.add((d, a))
            assert strict == {hb_keys[i] for i in np.flatnonzero(series["hb3"][frame])}
            assert wide == {hb_keys[i] for i in np.flatnonzero(series["hb35"][frame])}
            strict, wide = set(), set()
            for p, ligand in definition["hydrophobic_atom_pairs"]:
                distance2 = sum(delta(p, ligand) ** 2)
                key = (definition["protein_labels"][str(p)], ligand)
                if distance2 <= 4.5**2:
                    wide.add(key)
                    if distance2 <= 4**2:
                        strict.add(key)
            assert strict == {hyd_keys[i] for i in np.flatnonzero(series["hyd4"][frame])}
            assert wide == {hyd_keys[i] for i in np.flatnonzero(series["hyd45"][frame])}
            sampled_frames += 1
        for _, item in htable[(htable.ligand_id == cid) & (htable.seed == seed)].iterrows():
            i = hb_keys.index((item.donor, item.acceptor))
            assert np.isclose(item.occupancy_3A, series["hb3"][:, i].sum() / count)
            assert np.isclose(item.occupancy_35A, series["hb35"][:, i].sum() / count)
        for _, item in ytable[(ytable.ligand_id == cid) & (ytable.seed == seed)].iterrows():
            ix = [i for i, key in enumerate(hyd_keys) if key[0] == item.residue]
            assert np.isclose(item.occupancy_4A, (series["hyd4"][:, ix].sum(axis=1) > 0).mean())
            assert np.isclose(item.occupancy_45A, (series["hyd45"][:, ix].sum(axis=1) > 0).mean())
        cached = np.load(args.existing / cid / str(seed) / "aligned_ligand_frames.npz")["xyz_A"][
            :count
        ]
        for group, xyz in (("protein_CA", series["ca_xyz_A"]), ("ligand_heavy", cached)):
            mean, m2 = np.zeros(xyz.shape[1:]), np.zeros(xyz.shape[1:])
            for i, point in enumerate(xyz, 1):
                difference = point - mean
                mean += difference / i
                m2 += difference * (point - mean)
            expected = np.sqrt(m2.sum(axis=1) / len(xyz))
            actual = rtable[
                (rtable.ligand_id == cid) & (rtable.seed == seed) & (rtable.group == group)
            ].RMSF_A.to_numpy()
            assert np.allclose(expected, actual, atol=1e-10)
        angles = pd.read_csv(folder / "torsion_angles.csv")
        for f in sorted({0, count // 2, count - 1}):
            conf = Chem.Conformer(cached.shape[1])
            for i, point in enumerate(cached[f]):
                conf.SetAtomPosition(i, point)
            for t in definition["torsions"]:
                actual = angles.iloc[f][f"bond_{t['bond_index']}"]
                expected = rdMolTransforms.GetDihedralDeg(conf, *t["atoms"])
                assert abs((actual - expected + 180) % 360 - 180) < 1e-8
                torsion_checks += 1
    cluster_checks = 0
    if (args.analysis / "cluster_membership.csv").exists():
        membership = pd.read_csv(args.analysis / "cluster_membership.csv")
        for cid in membership.ligand_id.unique():
            xyz = np.concatenate(
                [
                    np.load(args.existing / cid / str(s) / "aligned_ligand_frames.npz")["xyz_A"]
                    for s in (20261011, 20261012, 20261013)
                ]
            )
            for cutoff in (1.0, 1.5, 2.0):
                data = np.load(args.analysis / cid / f"clusters_{cutoff:.1f}A.npz")
                labels = data["labels"]
                assert len(labels) == 1500 and labels.min() == 1
                assert np.array_equal(
                    labels,
                    membership[
                        (membership.ligand_id == cid) & (membership.cutoff_A == cutoff)
                    ].cluster,
                )
                for label, center in enumerate(data["center_indices"], 1):
                    members = np.flatnonzero(labels == label)
                    distances = np.sqrt(
                        np.sum((xyz[members] - xyz[center]) ** 2, axis=2).mean(axis=1)
                    )
                    assert distances.max() <= cutoff + 1e-8
                    assert np.allclose(data["center_xyz_A"][label - 1], xyz[center])
                    cluster_checks += len(members)
    result = {
        "status": "passed",
        "frames": sum(r["frames"] for r in records),
        "geometry_sample_frames": sampled_frames,
        "independent_rdkit_dihedrals": torsion_checks,
        "cluster_memberships_verified": cluster_checks,
        "occupancies_and_Welford_RMSF": True,
        "wall_seconds": time.monotonic() - start,
        "checked_utc": datetime.now(UTC).isoformat(),
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
