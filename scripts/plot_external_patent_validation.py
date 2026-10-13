"""从既有外部评价结果绘图，不改变评分或准入。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
frame = pd.read_csv(args.run / "predictions.csv")
metrics = pd.read_csv(args.run / "metrics.csv").query("view == 'primary'")
methods = ["score_20261001", "score_20261002", "score_20261003", "nearest_strong_similarity"]
names = ["Primary model", "Seed 20261002", "Seed 20261003", "Similarity reference"]
blue, orange = "#176b87", "#d87a35"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.45]})
values = metrics.set_index("method").loc[methods, "hits"].to_numpy()
ax.barh(np.arange(4), values, color=[blue, blue, blue, "#84949e"], height=0.58)
ax.set(
    yticks=np.arange(4),
    yticklabels=names,
    xlim=(0, 14.4),
    xlabel="Strong compounds selected (budget = 13)",
)
ax.invert_yaxis()
ax.axvline(
    metrics.iloc[0].random_hits, color=orange, ls="--", lw=1.4, label="Random expectation: 11.30"
)
for i, value in enumerate(values):
    ax.text(value + 0.15, i, f"{value:.0f}/13", va="center", fontsize=10)
ax.legend(loc="lower left", bbox_to_anchor=(0, -0.31), frameon=False, fontsize=9)
ax.set_title("A  Fixed-budget result", loc="left", fontweight="bold", pad=18)
matrix = []
for method in methods:
    ordered = frame.sort_values([method, "sample_id"], ascending=[False, True])
    matrix.append(ordered.label.to_numpy())
bx.imshow(
    np.array(matrix),
    aspect="auto",
    interpolation="nearest",
    cmap=ListedColormap([orange, blue]),
    extent=(0.5, 61.5, 3.5, -0.5),
    vmin=0,
    vmax=1,
)
bx.set(
    yticks=np.arange(4),
    yticklabels=names,
    xticks=[1, 13, 25, 37, 49, 61],
    xlabel="Rank (higher score first)",
)
bx.tick_params(axis="y", labelsize=9)
bx.axvline(13.5, color="black", lw=1.5, ls="--")
bx.text(6, -0.28, "Top 13", ha="center", fontsize=9, color="white")
bx.set_title("B  Entire ranked panel", loc="left", fontweight="bold", pad=18)
bx.legend(
    handles=[
        Patch(color=blue, label="IC50 <= 1 uM (53)"),
        Patch(color=orange, label="IC50 > 1 uM (8)"),
    ],
    loc="lower left",
    bbox_to_anchor=(0, -0.31),
    frameon=False,
    ncol=2,
    fontsize=9,
)
for a in [ax, bx]:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "GPR84 external retrospective evaluation | WO2024028364A1, Table 10", fontsize=14, y=1.01
)
fig.subplots_adjust(left=0.14, right=0.99, bottom=0.25, top=0.83, wspace=0.5)
fig.text(
    0.14,
    0.015,
    "Original Oct 1 checkpoints; no retraining. Rank ties use expected hits for metrics; "
    "ID order is for display only.",
    fontsize=8,
    color="#52606a",
)
for suffix in ["png", "svg", "pdf"]:
    fig.savefig(args.run / f"external_validation.{suffix}", dpi=220, bbox_inches="tight")
plt.close(fig)
table = metrics[["method", "hits", "ef", "roc_auc", "average_precision"]].copy()
table.columns = ["方法", "前13项强效数", "EF", "ROC-AUC", "AP"]
table["方法"] = names
html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPR84 外部评价</title>
<style>body{max-width:1120px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;
color:#18333e;background:#fafcfb}h1{font-size:30px}img{width:100%}
table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;
border-bottom:1px solid #ccd7d9}a{color:#176b87}.lead{font-size:21px}</style>
<h1>新增外部证据：原模型优于相似性参照</h1><p class="lead">61项公开实测中，
固定主模型前13项命中12个强效分子，相似性参照命中10个。原模型未重新训练。</p>
<img src="external_validation.png">"""
html += table.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0)
html += """<p>主模型EF=1.062，两个预定敏感种子均为1.151；本集合EF上限为1.151，随机预期命中11.30项。
合并一对立体样品后的60连接图分析，主模型为11/12，相似性参照为10/12。</p>
<p>这是一份新来源上的正向回顾评价，支持既有模型在本系列中的强效排序。
结果来自一个专利系列；原十项提名的实际活性仍待实验。</p><p>
<a href="predictions.csv">全部61项评分与排名</a> ·
<a href="metrics.csv">全部方法/敏感性指标</a> ·
<a href="weaker_samples_ranks.csv">8项较弱样品排名</a> ·
<a href="../independent_validation_admission_2026-10-03_v1/admission_v1/admission_all_61.csv">
结构—样品—实测准入表</a> ·
<a href="../independent_validation_admission_2026-10-03_v1/quality_notes.md">人工质检说明</a> ·
<a href="https://patents.google.com/patent/WO2024028364A1/en">专利来源</a></p></html>"""
(args.run / "index.html").write_text(html, encoding="utf-8")
