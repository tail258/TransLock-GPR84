"""绘制新相互作用与构象分析，不重新读取原始MD轨迹。"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["GLPG1205", "CHEMBL120691", "CHEMBL1086046"]
SEEDS = [20261011, 20261012, 20261013]
COLORS = ["#2878b5", "#d77a2b", "#239474"]


def save(fig, output, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def residue_number(value):
    match = re.search(r"(\d+)$", value)
    return int(match[1]) if match else 10000


def heatmap(table, key, value, cutoff, output, name, title):
    selected = []
    for cid in ORDER:
        sub = table[table.ligand_id == cid]
        matrix = (
            sub.pivot_table(index=key, columns="seed", values=value, aggfunc="max")
            .reindex(columns=SEEDS)
            .fillna(0)
        )
        matrix = matrix[matrix.max(axis=1) >= cutoff]
        matrix = matrix.loc[sorted(matrix.index, key=residue_number)]
        selected.append(matrix)
    fig, axes = plt.subplots(1, 3, figsize=(14, max(5, max(len(t) for t in selected) * 0.3 + 1.4)))
    for ax, cid, matrix in zip(axes, ORDER, selected, strict=True):
        im = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
        ax.set_xticks(range(3), [str(s) for s in SEEDS], rotation=30, ha="right")
        ax.set_title(cid)
        for i in range(len(matrix)):
            for j in range(3):
                value_here = matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value_here:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value_here > 0.6 else "#243649",
                )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.96, 0.97))
    cax = fig.add_axes([0.97, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="Frame fraction")
    save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    hb = pd.read_csv(args.analysis / "hbond_occupancies.csv")
    hyd = pd.read_csv(args.analysis / "hydrophobic_occupancies.csv")
    fluctuations = pd.read_csv(args.analysis / "RMSF.csv")
    membership = pd.read_csv(args.analysis / "cluster_membership.csv")
    clusters = pd.read_csv(args.analysis / "cluster_summary.csv")
    hb["pair"] = hb.residue + ":" + hb.receptor_atom + " / L:" + hb.ligand_atom
    heatmap(
        hb,
        "pair",
        "occupancy_3A",
        0.05,
        args.output,
        "direct_hydrogen_bonds",
        "Direct protein-ligand H bonds | D-A <=3.0 A, D-H-A >=150 deg\n"
        "Pairs >=5% in any repeat; full table retains all detected pairs",
    )
    heatmap(
        hyd,
        "residue",
        "occupancy_4A",
        0.4,
        args.output,
        "nonpolar_contacts",
        "Conservative nonpolar C-C contacts | <=4.0 A\n"
        "Residues >=40% in any repeat; occupancy is not binding energy",
    )
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    for i, cid in enumerate(ORDER):
        for j, group in enumerate(("protein_CA", "ligand_heavy")):
            sub = fluctuations[(fluctuations.ligand_id == cid) & (fluctuations.group == group)]
            if group == "protein_CA":
                sub = sub[sub.pocket & ~sub.synthetic]
            for seed, color in zip(SEEDS, COLORS, strict=True):
                data = sub[sub.seed == seed].sort_values("atom_index")
                axes[i, j].plot(
                    range(len(data)), data.RMSF_A, "o-", color=color, ms=3, lw=1, label=str(seed)
                )
            labels = data.residue.tolist() if j == 0 else data.atom_name.tolist()
            ticks = np.arange(0, len(labels), 2 if j == 0 else 1)
            axes[i, j].set_xticks(
                ticks, [labels[k] for k in ticks], rotation=65, ha="right", fontsize=7
            )
            axes[i, j].set(
                title=f"{cid} | {'Pocket CA' if j == 0 else 'Ligand heavy atoms'}",
                ylabel="RMSF (A)",
            )
            axes[i, j].grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Fluctuation around each repeat's mean | native receptor alignment; ligand not self-fitted"
    )
    fig.tight_layout()
    save(fig, args.output, "local_RMSF")
    primary = membership[membership.cutoff_A == 1.5]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    proportions = []
    for i, cid in enumerate(ORDER):
        sub = primary[primary.ligand_id == cid]
        n_clusters = int(sub.cluster.max())
        cmap = plt.get_cmap("tab20", max(3, n_clusters))
        colors = [cmap(k - 1) for k in range(1, n_clusters + 1)]
        grid = np.array([sub[sub.seed == s].cluster.to_numpy() for s in SEEDS])
        axes[i, 0].imshow(
            grid,
            cmap=cmap,
            vmin=0.5,
            vmax=max(3, n_clusters) + 0.5,
            aspect="auto",
            extent=(0.01, 5, 2.5, -0.5),
            interpolation="nearest",
        )
        axes[i, 0].set_yticks(range(3), SEEDS)
        axes[i, 0].set(title=f"{cid} | cluster membership", xlabel="Time (ns)")
        bottom = np.zeros(3)
        for label in range(1, n_clusters + 1):
            values = np.array([(grid[j] == label).mean() for j in range(3)])
            axes[i, 1].bar(
                np.arange(3), values, bottom=bottom, color=colors[label - 1], label=f"C{label}"
            )
            bottom += values
        axes[i, 1].set_xticks(range(3), SEEDS)
        axes[i, 1].set(ylabel="Fraction", ylim=(0, 1), title="All 500 saved frames")
        axes[i, 1].legend(
            ncol=min(5, n_clusters), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15)
        )
        for seed in SEEDS:
            frame_labels = sub[sub.seed == seed].cluster.to_numpy()
            for label in range(1, n_clusters + 1):
                proportions.append(
                    dict(
                        ligand_id=cid,
                        seed=seed,
                        cluster=label,
                        full=float((frame_labels == label).mean()),
                        first100=float((frame_labels[:100] == label).mean()),
                        last100=float((frame_labels[-100:] == label).mean()),
                    )
                )
    fig.suptitle(
        "Ligand pose clusters | 1.5 A center radius; three repeats pooled per molecule\n"
        "Cluster numbers are molecule-specific; no equilibrium populations inferred"
    )
    fig.tight_layout()
    save(fig, args.output, "pose_clusters")
    pd.DataFrame(proportions).to_csv(args.output / "cluster_window_fractions.csv", index=False)
    for cid in ORDER:
        definitions = json.loads((args.analysis / cid / "chemical_definitions.json").read_text())[
            "torsions"
        ]
        fig, axes = plt.subplots(
            len(definitions),
            1,
            figsize=(12, max(4, len(definitions) * 1.6)),
            squeeze=False,
            sharex=True,
        )
        for seed, color in zip(SEEDS, COLORS, strict=True):
            angles = pd.read_csv(args.analysis / cid / str(seed) / "torsion_angles.csv")
            for i, t in enumerate(definitions):
                ax = axes[i, 0]
                ax.scatter(
                    angles.time_ps / 1000,
                    angles[f"bond_{t['bond_index']}"],
                    s=2,
                    alpha=0.55,
                    color=color,
                    label=str(seed),
                )
                ax.set(ylim=(-180, 180), yticks=[-180, 0, 180], ylabel="deg")
                ax.set_title(
                    f"Bond {t['bond_index']}: {' - '.join(t['atom_names'])}"
                    + (" [core boundary]" if t["core_boundary"] else ""),
                    fontsize=9,
                    loc="left",
                )
        axes[0, 0].legend(ncol=3, markerscale=3, fontsize=8, loc="upper right")
        axes[-1, 0].set_xlabel("Time (ns)")
        fig.suptitle(
            f"{cid} | topology-selected non-ring single-bond dihedrals\n"
            "Dots avoid artificial connecting lines across +/-180 degrees"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, args.output, f"torsions_{cid}")
    # 保存窗口关联表，供结果解释；不将观察关联写成因果或显著性。
    linked = []
    for cid in ORDER:
        for seed in SEEDS:
            labels = primary[(primary.ligand_id == cid) & (primary.seed == seed)].cluster.to_numpy()
            series = np.load(args.analysis / cid / str(seed) / "interaction_series.npz")
            definition = json.loads((args.analysis / cid / "chemical_definitions.json").read_text())
            hydkeys = definition["hydrophobic_groups"]
            residues = sorted({key[0] for key in hydkeys})
            hyd_counts = np.stack(
                [
                    series["hyd4"][:, [i for i, key in enumerate(hydkeys) if key[0] == r]].any(
                        axis=1
                    )
                    for r in residues
                ],
                axis=1,
            ).sum(axis=1)
            for label in sorted(set(labels)):
                keep = labels == label
                linked.append(
                    dict(
                        ligand_id=cid,
                        seed=seed,
                        cluster=int(label),
                        frames=int(keep.sum()),
                        mean_hbond_pairs=float(series["hb3"][keep].sum(axis=1).mean()),
                        mean_nonpolar_residues=float(hyd_counts[keep].mean()),
                    )
                )
    pd.DataFrame(linked).to_csv(args.output / "cluster_interaction_associations.csv", index=False)
    summary = json.loads((args.analysis / "replicate_summaries.json").read_text())
    page = """<!doctype html><html lang="zh"><meta charset="utf-8">
    <title>MD相互作用与构象补充</title>
    <style>body{max-width:1280px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;
    color:#243649;background:#fafbfc}h1,h2{color:#163a54}img{max-width:100%;background:white}
    table{border-collapse:collapse;width:100%;font-size:14px}
    td,th{padding:8px;border-bottom:1px solid #d9e3ea;text-align:left}
    a{color:#166598}.note{background:#eaf2f7;padding:18px}</style>
    <h1>已有MD：相互作用、局部波动与构象变化</h1>
    <p>2026-10-03 · 两候选＋GLPG1205对照 · 9条5ns · 4500帧</p>
    <p class="note">本页分析已有轨迹的氢键、接触、波动和构象分布，复用RMSD及一般接触结果。
    三个配体采用中性微观状态，常规盐桥统计不适用；生理条件下的主要质子化状态待核查。
    各指标描述已采样轨迹。</p>"""
    page += (
        "<h2>氢键：区分化学类型与几何</h2><p>主判据供体–受体≤3.0Å且供体–氢–受体≥150°；"
        "附表保留3.5Å敏感性。多氢共享同一供受体对时，每帧只记一次。"
        "只含直接受体–配体N/O氢键，不含水桥；10ps保存间隔不支持连续氢键寿命。</p>"
    )
    hbshort = pd.DataFrame(summary)[
        ["ligand_id", "seed", "mean_hbond_pairs_3A", "mean_hbond_pairs_35A"]
    ]
    page += hbshort.rename(
        columns={
            "ligand_id": "对象",
            "seed": "种子",
            "mean_hbond_pairs_3A": "每帧平均氢键对数（3.0Å）",
            "mean_hbond_pairs_35A": "3.5Å敏感性",
        }
    ).to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0)
    for title, name, note in [
        (
            "直接氢键占有率",
            "direct_hydrogen_bonds",
            "图中L为配体原子；图展示任一重复至少5%的对，完整表保留所有检测到的对。",
        ),
        (
            "保守疏水接触",
            "nonpolar_contacts",
            "仅统计直接邻接C/H的非极性碳，4Å碳–碳接触。残基图展示任一重复至少40%的条目；不代表完整疏水能、卤素键或芳香堆积。",
        ),
        (
            "局部RMSF",
            "local_RMSF",
            "受体原生CA对齐后，围绕每条轨迹自己的平均位置统计；配体未自身叠合。口袋为三个原输入6Å残基并集，人工连接区不纳入口袋解释。",
        ),
        (
            "配体姿势聚类",
            "pose_clusters",
            "相同分子的三个重复合并建立簇，再分别统计。1.5Å为到中心的半径，不是簇内所有帧两两最大距离；1.0/2.0Å敏感性保留。不同分子的簇编号不能横向等同。",
        ),
    ]:
        page += (
            f"<h2>{title}</h2><p>{note}</p><img src='{name}.png'>"
            f"<p><a href='{name}.pdf'>PDF</a> · <a href='{name}.svg'>SVG</a></p>"
        )
    sensitivity = (
        clusters.groupby(["ligand_id", "cutoff_A"])
        .agg(clusters=("cluster", "max"), largest_cluster_frames=("frames", "max"))
        .reset_index()
    )
    page += "<h3>聚类阈值敏感性</h3>" + sensitivity.to_html(index=False, border=0)
    for cid in ORDER:
        page += (
            f"<h2>{cid}：关键二面角</h2><p>按拓扑预选，未按结果挑角。"
            "每张图保留三次重复；±180°两侧在圆周上相邻。</p>"
            f"<img src='torsions_{cid}.png'><p><a href='torsions_{cid}.pdf'>PDF</a></p>"
        )
    page += "<h2>数据与来源</h2><ul>"
    for label, filename in [
        ("氢键全表", "../hbond_occupancies.csv"),
        ("非极性接触全表", "../hydrophobic_occupancies.csv"),
        ("RMSF全表（含连接区标识）", "../RMSF.csv"),
        ("二面角圆周统计", "../torsion_summary.csv"),
        ("每帧聚类", "../cluster_membership.csv"),
        ("聚类首末窗口", "cluster_window_fractions.csv"),
        ("聚类与接触描述性关联", "cluster_interaction_associations.csv"),
        ("运行记录", "../manifest.json"),
    ]:
        page += f"<li><a href='{filename}'>{label}</a></li>"
    page += (
        "</ul><p>3×5ns描述有限时间内的条件性构象；三个重复共享准备起点，"
        "不表示平衡态或活性确证。</p></html>"
    )
    (args.output / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    (args.output / "plot_manifest.json").write_text(
        json.dumps(
            {"wall_seconds": time.monotonic() - start, "figure_groups": 7, "status": "complete"},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "complete", "seconds": time.monotonic() - start}))


if __name__ == "__main__":
    main()
