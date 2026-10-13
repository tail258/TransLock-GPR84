"""独立回读四组结果并绘图；不拟合模型、不选择配置。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

METHODS = ["random_frozen", "pretrained_frozen", "random_trained", "pretrained_trained"]
LABELS = ["Random / frozen", "Pretrained / frozen", "Random / trained", "Pretrained / tuned"]
COLORS = ["#91a6b8", "#4c84af", "#eea95b", "#c65d43"]
SOURCES = {
    "CHEMBL4675902": "GLPG (2020)",
    "CHEMBL5142410": "Triazines (2022)",
    "CHEMBL5159662": "Series (2022)",
    "CHEMBL5385083": "Phosphates (2023)",
    "CHEMBL6137896": "Functional series (2025)",
}


def independent_check(run):
    """不用生产评价函数重算成员、top-k期望、EF与全部配对差。"""
    units = pd.read_csv(run / "assay_graph_units.csv")
    members = pd.read_csv(run / "fold_members.csv")
    predictions = pd.read_csv(run / "predictions.csv")
    metrics = pd.read_csv(run / "metrics_by_assay.csv")
    records = json.loads((run / "model_records.json").read_text(encoding="utf-8"))
    for document, rows in members.groupby("outer_document"):
        test = units[units.document_id.eq(document)]
        fit = units[~units.document_id.eq(document) & ~units.connection_id.isin(test.connection_id)]
        assert set(rows[rows.role.eq("held_out")].unit_id) == set(test.unit_id)
        assert set(rows[rows.role.eq("fit")].unit_id) == set(fit.unit_id)
        assert len(rows) == len(units)
    assert len(records) == 144
    for record in records:
        rows = predictions[predictions.run_id.eq(record["run_id"])]
        expected_test = units[units.document_id.eq(record["document_id"])]
        assert len(rows) == len(expected_test) and not rows.unit_id.duplicated().any()
        assert set(rows.unit_id) == set(expected_test.unit_id)
        assert np.array_equal(
            rows.set_index("unit_id").loc[expected_test.unit_id, "label"].to_numpy(),
            expected_test.label.to_numpy(),
        )
        assert record["epochs"] == 30
        assert record["optimizer_steps"] == 30 * math.ceil(record["train_units"] / 16)
        if record["method"].endswith("frozen"):
            assert record["encoder_max_abs_change"] == 0
        else:
            assert record["encoder_max_abs_change"] > 0
        assert record["head_max_abs_change"] > 0
    rebuilt = []
    for keys, group in predictions.groupby(
        ["method", "learning_rate", "seed", "document_id", "assay_id"]
    ):
        method, rate, seed, document, assay = keys
        budget = math.ceil(len(group) * 0.2)
        groups = group.assign(s=group.score.round(12)).groupby("s", sort=True)
        remaining, hits = budget, 0.0
        for _, tied in reversed(list(groups)):
            picked = min(remaining, len(tied))
            hits += picked * tied.label.mean()
            remaining -= picked
            if remaining == 0:
                break
        reference = metrics[
            metrics.method.eq(method)
            & metrics.learning_rate.eq(rate)
            & metrics.seed.eq(seed)
            & metrics.document_id.eq(document)
            & metrics.assay_id.eq(assay)
        ]
        assert len(reference) == 1
        row = reference.iloc[0]
        assert row.budget == budget and row.n == len(group)
        assert row.positives == group.label.sum()
        assert row.primary_eligible == (len(group) >= 10 and group.label.nunique() == 2)
        assert abs(hits - row.expected_hits_with_ties) < 1e-10
        assert abs(hits / budget - row.precision) < 1e-10
        if group.label.sum():
            assert abs(hits / (budget * group.label.mean()) - row.enrichment) < 1e-10
            assert abs(hits / group.label.sum() - row.recall) < 1e-10
        rebuilt.append(row.to_dict())
    assert len(rebuilt) == len(metrics) == 408
    pair = pd.read_csv(run / "factor_contrasts.csv")
    assert len(pair) == 30
    keys = ["document_id", "assay_id", "seed", "learning_rate"]
    reconstructed = (
        metrics[metrics.primary_eligible]
        .pivot(index=keys, columns="method", values="enrichment")
        .sort_index()
    )
    stored = pair.set_index(keys).sort_index()
    np.testing.assert_allclose(reconstructed[METHODS], stored[METHODS], atol=1e-12)
    for column, left, right in [
        ("pretraining_when_frozen", "pretrained_frozen", "random_frozen"),
        ("pretraining_when_trained", "pretrained_trained", "random_trained"),
        ("training_from_pretrained", "pretrained_trained", "pretrained_frozen"),
        ("training_from_random", "random_trained", "random_frozen"),
    ]:
        np.testing.assert_allclose(
            stored[column], reconstructed[left] - reconstructed[right], atol=1e-12
        )
    np.testing.assert_allclose(
        pair.interaction,
        pair.pretrained_trained - pair.pretrained_frozen - pair.random_trained + pair.random_frozen,
        atol=1e-12,
    )
    return (
        metrics,
        records,
        {
            "status": "pass",
            "models": len(records),
            "prediction_rows": len(predictions),
            "assay_metrics_recomputed": len(rebuilt),
            "contrast_rows": len(pair),
            "split_members_rebuilt": True,
        "recorded_weight_update_checks_consistent": True,
            "checkpoint_prediction_readback": "all_144_checked_in_production_runner",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / "summarize_training_factor.py")
    metrics, records, check = independent_check(args.run)
    primary = metrics[metrics.primary_eligible].copy()
    macro = primary.groupby(["method", "learning_rate", "seed"]).enrichment.mean().reset_index()
    table = macro.groupby(["method", "learning_rate"]).enrichment.agg(["mean", "min", "max"])
    table.to_csv(args.output / "macro_enrichment_summary.csv")
    source = (
        primary.groupby(["method", "learning_rate", "assay_id"])
        .agg(
            enrichment=("enrichment", "mean"),
            mean_hits=("expected_hits_with_ties", "mean"),
            budget=("budget", "first"),
            n=("n", "first"),
            positives=("positives", "first"),
        )
        .reset_index()
    )
    source.to_csv(args.output / "source_means.csv", index=False)
    contrasts = pd.read_csv(args.run / "factor_contrasts.csv")
    columns = [
        "pretraining_when_frozen",
        "pretraining_when_trained",
        "training_from_pretrained",
        "training_from_random",
        "interaction",
    ]
    contrast_macro = contrasts.groupby(["learning_rate", "seed"])[columns].mean()
    contrast_macro.to_csv(args.output / "contrast_macro_by_seed.csv")
    history = pd.read_csv(args.run / "training_history.csv")
    history.groupby(["method", "learning_rate", "epoch"]).train_weighted_bce.agg(
        ["mean", "min", "max"]
    ).to_csv(args.output / "training_loss_summary.csv")
    baseline = pd.read_csv(args.pilot / "metrics_by_assay.csv")
    baseline = baseline[baseline.primary_eligible & baseline.budget_fraction.eq(0.2)]
    neighbour = baseline[baseline.method.eq("nearest_measured_strong")].enrichment.mean()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=0.17, right=0.94, top=0.87, bottom=0.12, wspace=0.45, hspace=0.52)
    fig.suptitle(
        "Training controls: initialization x encoder updates",
        fontsize=20,
        x=0.09,
        ha="left",
        y=0.987,
        weight="bold",
    )
    fig.text(
        0.09,
        0.913,
        "GPR84 antagonist potency | same architecture | 30 epochs | 3 seeds\n"
        "6 document holdouts; 5 primary assays; top 20% per assay; retrospective known labels",
        fontsize=11,
        color="#4c5660",
    )
    ax = axs[0, 0]
    for index, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
        values = macro[macro.method.eq(method) & macro.learning_rate.eq(0.0001)].enrichment
        ax.plot([values.min(), values.max()], [index, index], color=color, linewidth=4, alpha=0.6)
        ax.scatter(values, np.full(len(values), index) + [-0.08, 0, 0.08], color=color, s=35)
        ax.scatter([values.mean()], [index], marker="D", color="black", s=23, zorder=4)
    ax.set_yticks(range(4), LABELS)
    ax.invert_yaxis()
    ax.axvline(1, color="#59636a", ls="--", lw=1)
    ax.axvline(neighbour, color="#64917b", ls=":", lw=1.5, label="Saved similarity reference")
    ax.set_xlabel("Mean enrichment across 5 assays (random = 1)")
    ax.set_title("A  Primary learning rate: 1e-4", loc="left", weight="bold")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.set_ylim(3.8, -0.5)

    ax = axs[0, 1]
    ordered = list(SOURCES)
    heat = (
        source[source.learning_rate.eq(0.0001)]
        .pivot(index="assay_id", columns="method", values="enrichment")
        .loc[ordered, METHODS]
    )
    span = max(0.15, float(np.abs(heat.to_numpy() - 1).max()))
    im = ax.imshow(
        heat, cmap="RdBu", norm=TwoSlopeNorm(vmin=1 - span, vcenter=1, vmax=1 + span), aspect="auto"
    )
    ax.set_xticks(
        range(4),
        ["Random\nfrozen", "Pretrained\nfrozen", "Random\ntrained", "Pretrained\ntuned"],
        fontsize=9,
    )
    ax.set_yticks(range(5), [SOURCES[key] for key in ordered], fontsize=10)
    for i in range(5):
        for j in range(4):
            value = heat.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value - 1) > span * 0.65 else "#17202a",
            )
    ax.set_title("B  Each source: mean over 3 seeds", loc="left", weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Enrichment")

    ax = axs[1, 0]
    pair = contrasts[contrasts.learning_rate.eq(0.0001)]
    for i, key in enumerate(ordered):
        group = pair[pair.assay_id.eq(key)]
        for column, offset, color, label in [
            ("training_from_random", -0.12, COLORS[2], "Train random encoder"),
            ("training_from_pretrained", 0.12, COLORS[3], "Tune pretrained encoder"),
        ]:
            values = group[column]
            ax.scatter(
                values,
                i + offset + np.array([-0.04, 0, 0.04]),
                s=22,
                color=color,
                label=label if i == 0 else None,
            )
    ax.axvline(0, color="#59636a", ls="--", lw=1)
    ax.set_yticks(range(5), [SOURCES[key] for key in ordered], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Paired change in enrichment: trained minus frozen")
    ax.set_title("C  Does encoder updating help? (1e-4)", loc="left", weight="bold")
    ax.set_ylim(5.1, -0.5)
    ax.legend(fontsize=9, frameon=False, loc="lower left")

    ax = axs[1, 1]
    for i, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
        for seed in sorted(macro.seed.unique()):
            values = macro[macro.method.eq(method) & macro.seed.eq(seed)].sort_values(
                "learning_rate"
            )
            ax.plot(
                [0 + (i - 1.5) * 0.04, 1 + (i - 1.5) * 0.04],
                values.enrichment,
                color=color,
                alpha=0.45,
                lw=1,
                marker="o",
                markersize=4,
            )
    ax.set_xticks([0, 1], ["1e-4 (primary)", "1e-3 (sensitivity)"])
    ax.axhline(1, color="#59636a", ls="--", lw=1)
    ax.set_ylabel("Mean enrichment across 5 assays")
    ax.set_title("D  Predeclared learning-rate sensitivity", loc="left", weight="bold")
    fig.text(
        0.09,
        0.02,
        "Colored dots/lines: seeds; black diamonds: means; horizontal spans: seed ranges "
        "(not confidence intervals). No best run selected.\n"
        "Endpoint: measured IC50 <= 1 uM in antagonist assays; this does not classify "
        "agonists vs antagonists or validate new compounds.",
        fontsize=9,
        color="#4c5660",
    )
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(args.output / f"training_factor.{extension}", dpi=300)
    plt.close(fig)
    check["saved_similarity_reference_macro_ef"] = float(neighbour)
    check["gradient_clip_step_totals"] = {
        part: sum(record["gradient_clip_steps"][part] for record in records)
        for part in ["head", "encoder"]
    }
    (args.output / "independent_readback.json").write_text(
        json.dumps(check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(table.to_string())
    print(contrast_macro.to_string())
    print(json.dumps(check, ensure_ascii=False))


if __name__ == "__main__":
    main()
