"""为已完成轨迹的分解分析绘制结构叠合、接触占有率与重复比较。"""

import argparse
import html
import json
import re
import shutil
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from translock_screening.md_interpretation import rmsd

ORDER = ["GLPG1205", "CHEMBL120691", "CHEMBL1086046"]
COLORS = ["#2878b5", "#d77a2b", "#239474"]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, output, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def residue_order(name):
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 10000, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = load(args.analysis / "summaries.json")
    assert len(rows) == 9
    groups = {
        name: sorted([r for r in rows if r["ligand_id"] == name], key=lambda r: r["seed"])
        for name in ORDER
    }
    contacts = pd.read_csv(args.analysis / "contact_details.csv")
    definitions = {name: load(args.analysis / name / "ligand_definition.json") for name in ORDER}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharey="row")
    metric_keys = [
        "total_A",
        "centroid_shift_A",
        "centered_A",
        "self_fit_internal_A",
        "core_A",
        "core_self_fit_internal_A",
    ]
    labels = [
        "Whole pose",
        "Center shift",
        "After centering",
        "Self-fit shape",
        "Core pose",
        "Core self-fit",
    ]
    for i, name in enumerate(ORDER):
        for j, ref in enumerate(["original", "replica_start"]):
            ax = axes[i, j]
            for n, row in enumerate(groups[name]):
                medians = [row["metrics"][f"{ref}_{key}"]["median"] for key in metric_keys]
                ax.plot(
                    np.arange(6) + (n - 1) * 0.15,
                    medians,
                    "o",
                    color=COLORS[n],
                    label=str(row["seed"]),
                    ms=7,
                )
            ax.set_xticks(np.arange(6), labels, rotation=25, ha="right")
            ax.set(
                ylabel=f"{name}\nMedian displacement (Å)",
                ylim=(0, None),
                title="Reference: original input pose" if j == 0 else "Reference: 5 ns start",
            )
            ax.grid(axis="y", alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
    fig.suptitle(
        "Receptor-aligned pose changes | one dot per 500-frame replicate", y=1.01, fontsize=16
    )
    fig.text(
        0.5,
        -0.02,
        "Self-fit residual removes ligand rigid motion; it is not pocket stability. "
        "Core = mapped Murcko atom set. Metrics are not all additive.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, args.output, "pose_displacement_components")

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    representative_rows, pair_rows = [], []
    for i, name in enumerate(ORDER):
        reference = np.load(args.analysis / name / "reference_geometry.npz")["ligand"]
        poses, legend, medoids, late = [reference], ["Original input"], [], []
        for n, row in enumerate(groups[name]):
            data = np.load(args.analysis / name / str(row["seed"]) / "aligned_ligand_frames.npz")
            if n == 0:
                poses.append(data["initial_A"])
                legend.append("Prepared start")
            frame_index = row["representative_frame"] - 1
            pose = data["xyz_A"][frame_index]
            poses.append(pose)
            medoids.append(pose)
            late.append(data["xyz_A"][row["late_representative_frame"] - 1])
            legend.append(f"{row['seed']} | {row['representative_time_ps'] / 1000:.2f} ns")
            representative_rows.append(
                {
                    "ligand_id": name,
                    "seed": row["seed"],
                    "representative_time_ps": row["representative_time_ps"],
                    "late_representative_time_ps": row["late_representative_frame"] * 10,
                }
            )
        for a, b in combinations(range(3), 2):
            pair_rows.append(
                {
                    "ligand_id": name,
                    "seed_a": groups[name][a]["seed"],
                    "seed_b": groups[name][b]["seed"],
                    "representative_RMSD_A": rmsd(medoids[a], medoids[b]),
                    "late_representative_RMSD_A": rmsd(late[a], late[b]),
                }
            )
        all_points = np.concatenate(poses)
        center = all_points.mean(axis=0)
        _, _, vt = np.linalg.svd(all_points - center, full_matrices=False)
        projected = [(p - center) @ vt.T for p in poses]
        core = set(definitions[name]["core_atom_indices"])
        colors = ["#78828a", "#8058a5"] + COLORS
        for j, dimensions in enumerate([(0, 1), (0, 2)]):
            ax = axes[i, j]
            for p, color, title in zip(projected, colors, legend, strict=True):
                xy = p[:, dimensions]
                for a, b in definitions[name]["bonds"]:
                    ax.plot(
                        xy[[a, b], 0],
                        xy[[a, b], 1],
                        color=color,
                        lw=2.0 if a in core and b in core else 0.9,
                        alpha=0.85,
                    )
                ax.plot([], [], color=color, lw=2, label=title)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(
                xlabel="Projection axis 1 (Å)",
                ylabel=f"Projection axis {dimensions[1] + 1} (Å)",
                title=f"{name} | {'front' if j == 0 else 'orthogonal'} view",
            )
            ax.grid(alpha=0.13)
            if j == 1:
                ax.legend(loc="upper left", fontsize=8, ncol=1)
    fig.suptitle(
        "Ligand poses in one receptor-aligned frame | two orthogonal views", y=1.0, fontsize=16
    )
    fig.text(
        0.5,
        0.005,
        "Thick bonds: Murcko core. One actual medoid frame per replicate, "
        "chosen using all 500 frames. "
        "No ligand-only fitting.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    save(fig, args.output, "ligand_pose_overlays")
    pd.DataFrame(pair_rows).to_csv(args.output / "between_replicate_poses.csv", index=False)
    pd.DataFrame(representative_rows).to_csv(args.output / "representative_frames.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 13))
    contact_summaries = {}
    for ax, name in zip(axes, ORDER, strict=True):
        c = contacts[contacts.ligand_id == name]
        residues = sorted(set(c.residue), key=residue_order)
        matrix = []
        for residue in residues:
            sub = c[c.residue == residue].set_index("seed")
            first = sub.iloc[0]
            matrix.append(
                [float(first.original_contact), float(first.replica_start_contact)]
                + [
                    float(sub.loc[r["seed"], "occupancy"]) if r["seed"] in sub.index else 0
                    for r in groups[name]
                ]
            )
        matrix = np.array(matrix)
        display = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(residues)), residues, fontsize=8)
        ax.set_xticks(
            range(5), ["Input", "Start", "Seed 11", "Seed 12", "Seed 13"], rotation=35, ha="right"
        )
        ax.set_title(name)
        original = matrix[:, 0] > 0
        starting = matrix[:, 1] > 0
        occupancies = matrix[:, 2:]
        common80 = occupancies.min(axis=1) >= 0.8
        common50 = occupancies.min(axis=1) >= 0.5
        lost20 = original & (occupancies.max(axis=1) <= 0.2)
        gained80 = (~original) & common80
        contact_summaries[name] = {
            "original_count": int(original.sum()),
            "prepared_start_count": int(starting.sum()),
            "common_80pct_all_three": [
                r for r, keep in zip(residues, common80, strict=True) if keep
            ],
            "common_50pct_all_three": [
                r for r, keep in zip(residues, common50, strict=True) if keep
            ],
            "original_lost_max_20pct": [
                r for r, keep in zip(residues, lost20, strict=True) if keep
            ],
            "new_common_80pct": [r for r, keep in zip(residues, gained80, strict=True) if keep],
            "mean_original_contact_recall_per_seed": occupancies[original].mean(axis=0).tolist(),
            "mean_start_contact_recall_per_seed": occupancies[starting].mean(axis=0).tolist(),
        }
    fig.subplots_adjust(wspace=0.5, top=0.93, bottom=0.09, right=0.91)
    colorbar = fig.add_axes([0.94, 0.25, 0.013, 0.5])
    fig.colorbar(display, cax=colorbar, label="Contact occupancy (4 Å)")
    fig.suptitle(
        "Residue contact changes | original input, prepared start, three full repeats", fontsize=16
    )
    fig.text(
        0.5,
        0.015,
        "Input/start are binary contacts; repeats are 500-frame fractions. "
        "Thresholds are descriptive, not activity criteria.",
        ha="center",
        fontsize=10,
    )
    save(fig, args.output, "residue_contact_comparison")
    (args.output / "contact_summary.json").write_text(
        json.dumps(contact_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_table = []
    for name in ORDER:
        row = {
            "对象": name,
            "核心/重原子": (
                f"{definitions[name]['core_atom_count']}/{definitions[name]['heavy_atom_count']}"
            ),
        }
        for label, key in [
            ("原姿势RMSD", "original_total_A"),
            ("整体中心移动", "original_centroid_shift_A"),
            ("核心位置RMSD", "original_core_A"),
            ("核心自身形变", "original_core_self_fit_internal_A"),
        ]:
            values = [r["metrics"][key]["median"] for r in groups[name]]
            row[label + " / Å"] = f"{min(values):.2f}–{max(values):.2f}"
        row["三重复均≥80%的接触数"] = len(contact_summaries[name]["common_80pct_all_three"])
        summary_table.append(row)
    pd.DataFrame(summary_table).to_csv(args.output / "interpretation_summary.csv", index=False)
    sections = []
    for name in ORDER:
        c = contact_summaries[name]
        sections.append(
            f"<h3>{name}</h3><p>三条均有至少80%帧接触：{html.escape('、'.join(c['common_80pct_all_three']))}。</p>"
            "<p>相对原输入新增且三条均≥80%："
            f"{html.escape('、'.join(c['new_common_80pct']) or '无')}；"
            "原输入有、三条均≤20%："
            f"{html.escape('、'.join(c['original_lost_max_20pct']) or '无')}。</p>"
        )
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>MD结构解释</title>
    <style>body{max-width:1180px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;
    color:#243649;background:#fafbfc}
    img{width:100%;background:white}table{border-collapse:collapse;width:100%;background:white}
    td,th{padding:10px;border-bottom:1px solid #dde4eb}
    h1,h2{color:#173752}a{color:#126393}.note{padding:16px;background:#eaf2f7}</style>
    <h1>九条MD的结构解释</h1><p>2026-10-03 · 3个分子 × 3次重复 × 5 ns · 全部4500帧</p>
    <p class="note">这是已有轨迹的进一步分析。每个点对应一条重复的500帧中位数；
    表中范围为三个中位数的范围。
    “核心自身形变”先将核心单独叠合，只看内部形状，不能用它代替口袋中的姿势变化。GLPG1205是已知对照，两项候选活性仍未知。</p>"""
    page += pd.DataFrame(summary_table).to_html(index=False, border=0)
    page += (
        "<h2>1 · 原姿势发生了什么变化</h2><p>整体RMSD同时受位置和形状影响。"
        "几何中心移动衡量平移；核心位置RMSD保留转向和位移；"
        "核心自身形变去除核心的刚体运动。</p>"
    )
    page += (
        '<img src="pose_displacement_components.png">'
        '<p><a href="pose_displacement_components.pdf">PDF</a> · '
        '<a href="pose_displacement_components.svg">SVG</a></p>'
    )
    page += (
        "<h2>2 · 真实代表姿势的叠合</h2><p>灰色是原输入，紫色是准备后起点，"
        "其余颜色是三个重复的实际代表帧。两个投影帮助区分平面内和垂直方向移动；"
        "粗线为Murcko核心。代表帧按全部帧平均距离选取，不挑最稳定的帧。</p>"
        '<img src="ligand_pose_overlays.png"><p><a href="ligand_pose_overlays.pdf">PDF</a> · '
        '<a href="representative_frames.csv">代表帧来源</a> · '
        '<a href="between_replicate_poses.csv">重复间代表姿势差异</a></p>'
    )
    page += (
        "<h2>3 · 接触是否相同</h2><p>接触只表示配体重原子与该残基重原子距离≤4 Å，"
        "不自动等于氢键或有利能量。原输入与准备起点是单帧；种子列是全部500帧占比。</p>"
        '<img src="residue_contact_comparison.png">'
    )
    page += "".join(sections)
    page += (
        '<p><a href="../replicate_displacements.csv">逐重复完整分解统计</a> · '
        '<a href="../contact_details.csv">全部接触表</a> · '
        '<a href="../manifest.json">运行记录</a></p>'
        "<p>三个随机重复共享准备末态，未覆盖替代起始姿势。"
        "膜仍有漂移，5ns结果用于短程姿势分析。</p></html>"
    )
    (args.output / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    print(json.dumps(contact_summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
