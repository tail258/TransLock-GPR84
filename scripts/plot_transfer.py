"""绘制已保存的六方法/五来源比较，不训练模型或重新选择评价结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

DOCUMENTS = {
    "CHEMBL4673243": "2020 GLPG",
    "CHEMBL5137091": "2022 三嗪",
    "CHEMBL5154757": "2022 磷酸酯",
    "CHEMBL5380862": "2023 磷酸酯",
    "CHEMBL6134846": "2025 功能",
}
METHODS = {
    "random_expectation": "随机期望\n首轮",
    "nearest_measured_strong": "强效近邻检索\n首轮",
    "lr_graph_weighted": "图降权 LR\n首轮基准 · 未缩放",
    "lr_assay_balanced": "assay 等权 LR\n首轮",
    "r3_scaled_lr": "r3 缩放 + LR\n新增对照",
    "chemeleon_scaled_lr": "CheMeleon + LR\n新增 · 冻结表示",
}
NEW_METHODS = {
    "chemeleon_scaled_lr": ("CheMeleon + LR", "#0072B2"),
    "r3_scaled_lr": ("r3 缩放 + LR", "#D55E00"),
}
BASELINE = "lr_graph_weighted"


def read_results(run: Path) -> tuple[pd.DataFrame, dict]:
    """完整性检查失败即停止；不静默丢弃来源、方法或异常数值。"""
    data = pd.read_csv(run / "comparison_primary.csv")
    summary = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    required = {
        "document_id",
        "assay_id",
        "method",
        "budget_fraction",
        "primary_eligible",
        "n",
        "positives",
        "budget",
        "expected_hits_with_ties",
        "random_expected_hits",
        "enrichment",
    }
    if missing := required - set(data):
        raise ValueError(f"comparison_primary.csv缺少列：{sorted(missing)}")
    if set(data["document_id"]) != set(DOCUMENTS) or set(data["method"]) != set(METHODS):
        raise ValueError("必须完整包含预定五个主要来源与全部六个方法")
    if len(data) != len(DOCUMENTS) * len(METHODS):
        raise ValueError("需要30个来源×方法记录，不聚合或挑选重复assay")
    if data.duplicated(["document_id", "method"]).any():
        raise ValueError("来源×方法重复，无法直接绘制固定比较")
    if not np.allclose(data["budget_fraction"], 0.2, atol=1e-12, rtol=0):
        raise ValueError("绘图仅接收固定前20%的结果")
    if not data["primary_eligible"].astype(str).str.lower().isin(["true", "1"]).all():
        raise ValueError("输入含非主要评价assay")
    numeric = [
        "n",
        "positives",
        "budget",
        "expected_hits_with_ties",
        "random_expected_hits",
        "enrichment",
    ]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("绘图指标包含非有限数值")
    if (data["random_expected_hits"] <= 0).any():
        raise ValueError("随机命中期望必须大于0")
    if not np.allclose(
        data["enrichment"], data["expected_hits_with_ties"] / data["random_expected_hits"]
    ):
        raise ValueError("EF与保存的命中期望不一致")
    for document, group in data.groupby("document_id"):
        if group[["assay_id", "n", "positives", "budget"]].nunique().max() != 1:
            raise ValueError(f"{document}的方法间样本或预算不一致")
    random = data[data["method"].eq("random_expectation")]
    if not np.allclose(random["enrichment"], 1.0):
        raise ValueError("随机期望的EF不是1")
    if summary.get("primary_document_count") != 5 or summary.get("primary_assay_count") != 5:
        raise ValueError("evaluation.json的主要来源/assay数量与图形定义不同")
    if summary.get("status") != "exploratory_retrospective_source_holdout_not_blind_test":
        raise ValueError("evaluation.json不是预期的回顾性来源留出结果")
    macro = summary["primary_assay_macro"]
    if {record["method"] for record in macro} != set(NEW_METHODS):
        raise ValueError("evaluation.json缺少或含额外的新增方法汇总")
    for record in macro:
        method = record["method"]
        if method in NEW_METHODS:
            mean = data.loc[data["method"].eq(method), "enrichment"].mean()
            if not np.isclose(mean, record["enrichment"]):
                raise ValueError(f"{method}的CSV与JSON平均EF不一致")

    # 只增加绘图标签与明确的算术差，不改变原始测量或模型输出。
    baseline = data[data["method"].eq(BASELINE)].set_index("document_id")
    data["delta_expected_hits_vs_lr_graph_weighted"] = data["expected_hits_with_ties"] - data[
        "document_id"
    ].map(baseline["expected_hits_with_ties"])
    data["figure_source"] = data["document_id"].map(DOCUMENTS)
    data["figure_method"] = data["method"].map(METHODS).str.replace("\n", " / ")
    return data, summary


def build_figure(data: pd.DataFrame, summary: dict) -> Figure:
    documents, methods = list(DOCUMENTS), list(METHODS)
    values = data.pivot(index="method", columns="document_id", values="enrichment")
    heat = values.loc[methods, documents].to_numpy()
    baseline = data[data["method"].eq(BASELINE)].set_index("document_id").loc[documents]
    figure = Figure(figsize=(16, 7), facecolor="white")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.32, 1]})
    figure.subplots_adjust(left=0.135, right=0.97, top=0.80, bottom=0.285, wspace=0.40)
    figure.text(
        0.045,
        0.954,
        "第二轮重构：六种方法在全部主要来源上的表现",
        fontsize=21,
        weight="bold",
        color="#182B39",
    )
    figure.text(
        0.045,
        0.902,
        "来源整篇留出  /  固定前20%预算  /  首轮4方法 + 新增2方法",
        fontsize=11.5,
        color="#56616D",
    )

    # 以随机EF=1为共同中心，正负偏离共用对称色阶；不按来源单独归一化。
    span = max(0.1, float(np.abs(heat - 1).max()))
    norm = TwoSlopeNorm(vmin=1 - span, vcenter=1, vmax=1 + span)
    palette = mpl.colormaps["BrBG"]
    image = left.imshow(heat, cmap=palette, norm=norm, aspect="auto", interpolation="nearest")
    left.set_title("A  前20%的富集倍数 EF", loc="left", fontsize=15, weight="bold", pad=16)
    left.set_yticks(np.arange(len(methods)), list(METHODS.values()), fontsize=10.5)
    labels = [
        f"{DOCUMENTS[doc]}\nn={int(baseline.loc[doc, 'n'])}，k={int(baseline.loc[doc, 'budget'])}"
        for doc in documents
    ]
    left.set_xticks(np.arange(len(documents)), labels, fontsize=9.5)
    left.tick_params(axis="both", length=0, pad=8)
    left.set_xticks(np.arange(-0.5, len(documents), 1), minor=True)
    left.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    left.grid(which="minor", color="white", linewidth=1.8)
    left.tick_params(which="minor", bottom=False, left=False)
    left.axhline(3.5, color="#182B39", linewidth=1.3)
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            rgba = palette(norm(heat[row, column]))
            luminance = np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
            left.text(
                column,
                row,
                f"{heat[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=11.5,
                color="white" if luminance < 0.49 else "#172A35",
            )
    for spine in left.spines.values():
        spine.set_visible(False)
    colour_axis = figure.add_axes([0.135, 0.17, 0.32, 0.018])
    colourbar = figure.colorbar(image, cax=colour_axis, orientation="horizontal")
    colourbar.set_ticks([1 - span, 1, 1 + span])
    colourbar.set_ticklabels([f"{1 - span:.2f}", "1：随机", f"{1 + span:.2f}"])
    colourbar.ax.tick_params(labelsize=9, length=2)
    colourbar.outline.set_visible(False)

    right.set_title("B  相对首轮图降权 LR 的命中差", loc="left", fontsize=15, weight="bold", pad=35)
    positions = np.arange(len(documents))
    largest = 0.0
    for method in NEW_METHODS:
        delta = data.loc[data["method"].eq(method), "delta_expected_hits_vs_lr_graph_weighted"]
        largest = max(largest, float(delta.abs().max()))
    bound = max(1.0, largest * 1.28)
    label_offset = bound * 0.025
    for index, (method, (label, colour)) in enumerate(NEW_METHODS.items()):
        block = data[data["method"].eq(method)].set_index("document_id").loc[documents]
        delta = block["delta_expected_hits_vs_lr_graph_weighted"].to_numpy()
        y = positions + (index - 0.5) * 0.30
        right.barh(y, delta, height=0.255, color=colour, label=label, zorder=3)
        for position, value in zip(y, delta, strict=True):
            text = "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"
            right.text(
                value + (label_offset if value >= 0 else -label_offset),
                position,
                text,
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=10,
                color=colour,
            )
    right.axvline(0, color="#66727B", linewidth=1.0, zorder=2)
    right.set_yticks(positions, list(DOCUMENTS.values()), fontsize=10)
    right.set_ylim(len(documents) - 0.45, -0.55)
    right.set_xlim(-bound, bound)
    right.set_xlabel("前20%期望命中差（个）", fontsize=11, labelpad=10)
    right.grid(axis="x", color="#DCE3E8", linewidth=0.7, zorder=0)
    right.tick_params(axis="both", length=0, pad=7)
    right.legend(
        loc="lower left",
        bbox_to_anchor=(-0.025, 1.015),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        columnspacing=1.4,
    )
    right.spines[["top", "right", "left"]].set_visible(False)
    right.spines["bottom"].set_color("#BAC4CB")
    folds = len(summary.get("folds", []))
    figure.text(
        0.045,
        0.096,
        f"{folds}篇来源整篇留出；图示完整5个主要assay。n为评价项数；k=ceil(0.20×n)；"
        "强效仅指公开拮抗实验IC50≤1 µM。",
        fontsize=9.5,
        color="#4F5B66",
    )
    figure.text(
        0.045,
        0.058,
        "边界并列采用均匀选择的期望命中；数值标注保留3位小数，source_data.csv保留原值。"
        "采用历史已见标签进行回顾评价。",
        fontsize=9.5,
        color="#4F5B66",
    )
    figure.text(
        0.045,
        0.022,
        "该图展示分子表示的区分能力；训练类别来自准入测量，未知候选的实验活性待验证。",
        fontsize=9.5,
        color="#4F5B66",
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录：{args.output}")
    if not args.font.is_file():
        raise FileNotFoundError(f"中文字体文件不存在：{args.font}")
    data, summary = read_results(args.run)
    font_manager.fontManager.addfont(str(args.font))
    style = {
        "font.family": font_manager.FontProperties(fname=str(args.font)).get_name(),
        "font.size": 11,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with mpl.rc_context(style):
        figure = build_figure(data, summary)
        for extension in ("png", "svg", "pdf"):
            figure.savefig(args.output / f"transfer_comparison.{extension}", dpi=260)
        figure.clear()
    data.to_csv(args.output / "source_data.csv", index=False)
    print(
        json.dumps(
            {
                "status": "plotted_existing_results",
                "rows": len(data),
                "sources": len(DOCUMENTS),
                "methods": len(METHODS),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
