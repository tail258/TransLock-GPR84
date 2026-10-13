"""回读重复MD的全部DCD及末态，独立核对几何/时间并汇总每条轨迹。"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from review_membrane_control import state_positions

from translock_screening.dcd_reader import read_openmm_dcd
from translock_screening.md_pilot import PilotDiagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["status"] == "complete_fixed_length"
    initial, box, parameters, time = state_positions(args.run / "initial_state.xml")
    assert time == 0 and float(parameters["k"]) == 0
    diagnostic = PilotDiagnostics(manifest["warmup"], initial)
    assert diagnostic.calculate(initial, box)["geometry_passed"]
    online = [
        json.loads(line) for line in (args.run / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(online) == manifest["target_time_ps"] // 10 + 1
    frames = []
    last = None
    for i, frame in enumerate(read_openmm_dcd(args.run / "trajectory.dcd"), start=1):
        assert frame.step == i * 5000
        xyz, vectors = frame.positions_nm, frame.box_vectors_nm
        assert np.isfinite(xyz).all() and np.linalg.det(vectors) > 0
        # 以分数坐标及行列式独立复核在线的最小像/叉积实现。
        inverse = np.linalg.inv(vectors)

        def periodic(delta, inverse=inverse, vectors=vectors):
            f = delta @ inverse
            return (f - np.rint(f)) @ vectors

        cn, ss = np.array(diagnostic.cn), np.array(diagnostic.ss)
        cnd = np.linalg.norm(periodic(xyz[cn[:, 0]] - xyz[cn[:, 1]]), axis=1) * 10
        ssd = np.linalg.norm(periodic(xyz[ss[:, 0]] - xyz[ss[:, 1]]), axis=1) * 10
        assert cnd.min() > 1.1 and cnd.max() < 1.6 and ssd.min() > 1.8 and ssd.max() < 2.3
        q = diagnostic.alpha
        v = [periodic(xyz[q[:, j]] - xyz[q[:, 1]]) * 10 for j in (0, 2, 3)]
        vh = periodic(xyz[diagnostic.ha] - xyz[q[:, 1]]) * 10
        assert np.linalg.det(np.stack(v, axis=1)).min() > 1
        assert np.linalg.det(np.stack([v[0], v[1], vh], axis=1)).max() < 0
        row = diagnostic.calculate(xyz, vectors)
        assert row["geometry_passed"]
        row["time_ps"] = frame.step * 0.002
        assert abs(row["time_ps"] - online[i]["time_ps"]) < 1e-4
        for key in ("ligand_RMSD_to_original_pose_A", "ligand_RMSD_to_replica_start_A", "area_nm2"):
            assert abs(row[key] - online[i][key]) < 2e-4, (i, key)
        frames.append(row)
        last = frame
    assert len(frames) == manifest["target_time_ps"] // 10
    final, final_box, final_parameters, final_time = state_positions(args.run / "final_state.xml")
    assert abs(final_time - manifest["target_time_ps"]) < 1e-4 and float(final_parameters["k"]) == 0
    assert np.max(np.abs(final - last.positions_nm)) < 2e-6
    assert np.allclose(final_box, last.box_vectors_nm, atol=1e-7)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    for name in ("md_pilot.py", "md_diagnostics.py", "dcd_reader.py"):
        shutil.copyfile(
            Path(__file__).parents[1] / "src/translock_screening" / name, args.output / name
        )
    table = pd.DataFrame([{k: v for k, v in f.items() if not isinstance(v, list)} for f in frames])
    table.to_csv(args.output / "frames.csv", index=False)
    counts = Counter(contact for frame in frames for contact in frame["contacts_4A"])
    occupancy = [
        {"residue": residue, "frames": count, "occupancy": count / len(frames)}
        for residue, count in counts.most_common()
    ]
    pd.DataFrame(occupancy).to_csv(args.output / "contact_occupancy.csv", index=False)
    summary = {}
    for key in (
        "ligand_RMSD_to_original_pose_A",
        "ligand_RMSD_to_replica_start_A",
        "pocket_center_distance_A",
        "initial_contact_recall",
        "area_nm2",
        "volume_nm3",
    ):
        values = table[key].to_numpy()
        summary[key] = dict(
            initial=online[0][key],
            final=float(values[-1]),
            median=float(np.median(values)),
            minimum=float(values.min()),
            maximum=float(values.max()),
            first_20pct_mean=float(values[: max(1, len(values) // 5)].mean()),
            last_20pct_mean=float(values[-max(1, len(values) // 5) :].mean()),
        )
    result = dict(
        status="passed",
        ligand_id=manifest["ligand_id"],
        seed=manifest["base_seed"],
        time_ps=final_time,
        trajectory_frames=len(frames),
        geometry_failed_frames=0,
        random_seeds=manifest["random_seeds"],
        checkpoint_rng_loaded=False,
        metrics=summary,
        contact_occupancy=occupancy,
        scope="固定长度完整回读；帧为相关采样，三条重复共享准备末态；不证明收敛或活性",
    )
    (args.output / "independent_readback.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    labels = [
        ("ligand_RMSD_to_original_pose_A", "Ligand RMSD to input pose (A)"),
        ("ligand_RMSD_to_replica_start_A", "Ligand RMSD to replica start (A)"),
        ("pocket_center_distance_A", "Ligand to pocket center (A)"),
        ("initial_contact_recall", "Starting contact recall"),
        ("area_nm2", "Membrane area (nm2)"),
        ("volume_nm3", "Volume (nm3)"),
    ]
    for ax, (key, label) in zip(axes.ravel(), labels, strict=True):
        ax.plot(table.time_ps / 1000, table[key], lw=1.2, color="#315b91")
        ax.set(xlabel="Time (ns)", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{manifest['ligand_id']} | seed {manifest['base_seed']} | exploratory MD")
    fig.tight_layout()
    fig.savefig(args.output / "trajectory.png", dpi=170)
    fig.savefig(args.output / "trajectory.svg")
    plt.close(fig)
    html = (
        f"""<!doctype html><html lang="zh"><meta charset="utf-8">
        <title>{manifest["ligand_id"]} MD</title>
        <style>body{{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui}}
        img{{width:100%}}td,th{{padding:6px}}</style>
        <h1>{manifest["ligand_id"]} · {manifest["base_seed"]}</h1>
        <p>固定{final_time / 1000:g} ns；{len(frames)}帧全部回读通过。
        短程姿势证据，活性和功能方向仍未知。</p><img src="trajectory.png">"""
        + pd.DataFrame(summary).T.to_html()
        + pd.DataFrame(occupancy).to_html(index=False)
        + "</html>"
    )
    (args.output / "index.html").write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("status", "ligand_id", "seed", "time_ps", "trajectory_frames")}
        )
    )


if __name__ == "__main__":
    main()
