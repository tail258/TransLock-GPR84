"""检查外围对称转动对CHEMBL1086046姿势聚类解释的影响。"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from translock_screening.md_interactions import gromos_clusters, pairwise_pose_distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    cid = "CHEMBL1086046"
    core = json.loads((args.existing / cid / "ligand_definition.json").read_text())[
        "core_atom_indices"
    ]
    seeds = [20261011, 20261012, 20261013]
    xyz = np.concatenate(
        [
            np.load(args.existing / cid / str(s) / "aligned_ligand_frames.npz")["xyz_A"][:, core]
            for s in seeds
        ]
    )
    distance = pairwise_pose_distances(xyz)
    summary, windows, memberships = [], [], []
    for cutoff in (1.0, 1.5, 2.0):
        labels, centers = gromos_clusters(distance, cutoff)
        for label, center in enumerate(centers, 1):
            members = np.flatnonzero(labels == label)
            # 独立用坐标差核对中心半径，不复用Gram矩阵结果。
            direct = np.sqrt(((xyz[members] - xyz[center]) ** 2).sum(axis=2).mean(axis=1))
            assert direct.max() <= cutoff + 1e-8
            summary.append(
                dict(
                    cutoff_A=cutoff,
                    cluster=label,
                    frames=len(members),
                    center_seed=seeds[center // 500],
                    center_frame=center % 500 + 1,
                )
            )
            for j, seed in enumerate(seeds):
                sub = labels[j * 500 : (j + 1) * 500]
                windows.append(
                    dict(
                        cutoff_A=cutoff,
                        cluster=label,
                        seed=seed,
                        full=float((sub == label).mean()),
                        first100=float((sub[:100] == label).mean()),
                        last100=float((sub[-100:] == label).mean()),
                    )
                )
        for j, label in enumerate(labels):
            memberships.append(
                dict(cutoff_A=cutoff, seed=seeds[j // 500], frame=j % 500 + 1, cluster=int(label))
            )
    pd.DataFrame(summary).to_csv(args.output / "cluster_summary.csv", index=False)
    pd.DataFrame(windows).to_csv(args.output / "cluster_window_fractions.csv", index=False)
    pd.DataFrame(memberships).to_csv(args.output / "cluster_membership.csv", index=False)
    manifest = dict(
        status="passed",
        ligand_id=cid,
        core_atoms=core,
        frames=1500,
        wall_seconds=time.monotonic() - start,
        scope="Post-result core-subset sensitivity; primary all-heavy-atom clusters retained",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
