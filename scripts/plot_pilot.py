"""仅绘制已保存的首轮结果，不重新训练或选择评价来源。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(args.font)).get_name(),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    data = pd.read_csv(args.run / "metrics_by_assay.csv")
    data = data[data["primary_eligible"] & data["budget_fraction"].eq(0.2)]
    names = {
        "CHEMBL4673243": "2020 GLPG系列",
        "CHEMBL5137091": "2022 三嗪系列",
        "CHEMBL5154757": "2022 磷酸酯系列",
        "CHEMBL5380862": "2023 磷酸酯系列",
        "CHEMBL6134846": "2025 功能系列",
    }
    methods = [
        ("lr_graph_weighted", "预选LR", "#266c9a"),
        ("lr_assay_balanced", "assay等权LR", "#e6a245"),
        ("nearest_measured_strong", "相似性检索", "#319b83"),
    ]
    docs = sorted(data.document_id.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.suptitle("第一轮重构：来源不同，检索表现也不同", fontsize=20, x=0.06, ha="left", y=0.97)
    x = np.arange(len(docs))
    for i, (method, label, color) in enumerate(methods):
        block = data[data.method.eq(method)].set_index("document_id").loc[docs]
        axes[0].bar(x + (i - 1) * 0.23, block.precision * 100, width=0.21, label=label, color=color)
        axes[1].plot(x, block.enrichment, "o-", color=color, label=label, linewidth=1.6)
    baseline = data[data.method.eq("random_expectation")].set_index("document_id").loc[docs]
    for i, row in enumerate(baseline.itertuples()):
        axes[0].hlines(row.precision * 100, i - 0.4, i + 0.4, color="#5d6067", linestyle="--")
    axes[0].set(title="A  每个来源固定选前20%", ylabel="入选项中阈值内强效比例（%）", ylim=(0, 108))
    axes[0].text(
        0.01, 1.02, "灰色虚线：该来源随机选择期望", transform=axes[0].transAxes, fontsize=10
    )
    axes[1].axhline(1, color="#5d6067", linestyle="--", linewidth=1)
    axes[1].set(title="B  相对随机选择的富集", ylabel="富集倍数（1 = 随机）", ylim=(0.5, 1.6))
    for ax in axes:
        ax.set_xticks(x, [names[doc].replace(" ", "\n", 1) for doc in docs], fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=3, frameon=False)
    fig.text(
        0.06,
        0.06,
        "6篇来源整篇留出；图中为5个满足预定规模且有双类的主要assay。\n"
        "强效仅指公开拮抗实验IC50≤1µM；并列采用期望值，历史标签已见，非独立盲测。",
        fontsize=10,
        color="#525863",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.27, wspace=0.28)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(args.output / f"pilot_comparison.{ext}", dpi=240)
    plt.close(fig)
    data.to_csv(args.output / "source_data.csv", index=False)


if __name__ == "__main__":
    main()
