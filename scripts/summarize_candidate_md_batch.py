"""九条固定重复全部回读后，形成逐条及重复之间的对照图表。"""

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    while True:
        status = json.loads((args.batch / "batch_manifest.json").read_text())
        if status["status"] == "complete_9_replicates":
            break
        if not args.wait or status["status"] == "stopped_on_failure":
            raise RuntimeError(f"尚未满足九条完整回读：{status['status']}，{status.get('current')}")
        if time.monotonic() - started > 48 * 3600:
            raise TimeoutError("汇总等待48小时未完成；不标记计算完成")
        time.sleep(30)
    plan = json.loads((args.batch / "batch_plan.json").read_text())
    results = json.loads((args.batch / "replicate_results.json").read_text())
    expected = {(obj["id"], seed) for obj in plan["objects"] for seed in plan["seeds"]}
    assert len(results) == 9 and {(r["ligand_id"], r["seed"]) for r in results} == expected
    assert all(
        r["status"] == "passed"
        and abs(r["time_ps"] - 5000) < 1e-4
        and r["trajectory_frames"] == 500
        for r in results
    )
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    rows = []
    contact_rows = []
    for result in results:
        metrics = result["metrics"]
        row = dict(ligand=result["ligand_id"], seed=result["seed"], frames=500, time_ns=5)
        for key in (
            "ligand_RMSD_to_original_pose_A",
            "ligand_RMSD_to_replica_start_A",
            "initial_contact_recall",
            "pocket_center_distance_A",
        ):
            for stat in ("final", "median", "maximum"):
                row[key + "_" + stat] = metrics[key][stat]
        for key in ("area_nm2", "volume_nm3"):
            row[key + "_first_mean"] = metrics[key]["first_20pct_mean"]
            row[key + "_last_mean"] = metrics[key]["last_20pct_mean"]
            row[key + "_change_pct"] = (
                metrics[key]["last_20pct_mean"] / metrics[key]["first_20pct_mean"] - 1
            ) * 100
        rows.append(row)
        for contact in result["contact_occupancy"]:
            contact_rows.append(dict(ligand=result["ligand_id"], seed=result["seed"], **contact))
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "replicates.csv", index=False)
    pd.DataFrame(contact_rows).to_csv(args.output / "contacts.csv", index=False)
    comparisons = []
    for obj in plan["objects"]:
        subset = table[table.ligand == obj["id"]]
        record = dict(ligand=obj["id"], independent_random_repeats=3, shared_preparation_state=True)
        for key in (
            "ligand_RMSD_to_original_pose_A_median",
            "ligand_RMSD_to_replica_start_A_median",
            "initial_contact_recall_median",
            "area_nm2_change_pct",
            "volume_nm3_change_pct",
        ):
            record[key + "_repeat_min"] = float(subset[key].min())
            record[key + "_repeat_max"] = float(subset[key].max())
        comparisons.append(record)
    pd.DataFrame(comparisons).to_csv(args.output / "between_repeats.csv", index=False)
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    metric_columns = [
        ("ligand_RMSD_to_original_pose_A", "RMSD to input pose (A)"),
        ("initial_contact_recall", "Starting contact recall"),
        ("area_nm2", "Membrane area (nm2)"),
    ]
    colors = ["#315b91", "#a55d38", "#398274"]
    for i, obj in enumerate(plan["objects"]):
        for seed, color in zip(plan["seeds"], colors, strict=True):
            frame = pd.read_csv(
                args.batch / "replicates" / obj["id"] / str(seed) / "analysis_v1/frames.csv"
            )
            for j, (key, label) in enumerate(metric_columns):
                axes[i, j].plot(
                    frame.time_ps / 1000,
                    frame[key],
                    label=str(seed),
                    color=color,
                    lw=1.0,
                    alpha=0.9,
                )
                axes[i, j].set(ylabel=label, title=obj["id"])
                axes[i, j].grid(alpha=0.2)
        axes[i, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (ns)")
    figure.suptitle(
        "GPR84: three random repeats per conditional pose | exploratory 5 ns", fontsize=15
    )
    figure.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(args.output / f"replicate_comparison.{suffix}", dpi=180)
    plt.close(figure)
    summary = dict(
        status="complete_and_read_back",
        objects=3,
        repeats=9,
        exploration_ns=45,
        saved_frames=4500,
        comparisons=comparisons,
        scope="共享准备末态的三条随机重复；短程姿势与接触证据，不证明充分平衡、活性、选择性或结合自由能",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    display = table[
        [
            "ligand",
            "seed",
            "ligand_RMSD_to_original_pose_A_median",
            "ligand_RMSD_to_original_pose_A_final",
            "initial_contact_recall_median",
            "area_nm2_change_pct",
        ]
    ]
    links = "".join(
        f'<li><a href="../replicates/{o["id"]}/{seed}/analysis_v1/index.html">'
        f'{o["id"]} · {seed}</a></li>'
        for o in plan["objects"]
        for seed in plan["seeds"]
    )
    html = (
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>候选重复膜MD</title>
    <style>body{max-width:1250px;margin:40px auto;padding:0 24px;
    font:16px/1.6 system-ui;color:#23334a}
    img{width:100%}table{font-size:13px;border-collapse:collapse}
    td,th{padding:7px;border:1px solid #ddd}</style>
    <h1>两候选与GLPG对照：三次5 ns探索MD</h1>
    <p>九条固定长度轨迹、45 ns、4500帧全部回读通过。每条分别展示，三条重复共享同一准备末态。
    目录中性图及起始姿势是计算条件；短轨迹和膜面积漂移限制动态解释，候选活性与功能方向仍未知。</p>
    <img src="replicate_comparison.png"><h2>逐条结果</h2>"""
        + display.to_html(index=False, float_format=lambda x: f"{x:.3f}")
        + "<h2>单条图与接触占有率</h2><ul>"
        + links
        + "</ul></html>"
    )
    (args.output / "index.html").write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("status", "objects", "repeats", "exploration_ns", "saved_frames")
            }
        )
    )


if __name__ == "__main__":
    main()
