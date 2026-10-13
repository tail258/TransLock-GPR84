"""从逐点预测独立核对交叉对照并绘图；不训练或挑选胜者。"""

from __future__ import annotations

import argparse
import importlib.metadata
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

CELLS = [
    "head_low_encoder_low",
    "head_high_encoder_low",
    "head_low_encoder_high",
    "head_high_encoder_high",
]
LABELS = ["H low / E low", "H high / E low", "H low / E high", "H high / E high"]
COLORS = ["#376c9a", "#bd653e", "#398875", "#9668a4"]
SOURCES = {
    "CHEMBL4675902": "GLPG (2020)",
    "CHEMBL5142410": "Triazines (2022)",
    "CHEMBL5159662": "Series (2022)",
    "CHEMBL5385083": "Phosphates (2023)",
    "CHEMBL6137896": "Functional series (2025)",
}
EFFECTS = [
    "head_effect_encoder_low",
    "head_effect_encoder_high",
    "encoder_effect_head_low",
    "encoder_effect_head_high",
    "interaction",
]


def check_outputs(run, base):
    """独立重算名额/并列命中/EF与差分，不调用生产评分模块。"""
    identity_types = {"activity_ids": "string"}
    units = pd.read_csv(run / "assay_graph_units.csv", dtype=identity_types)
    members = pd.read_csv(run / "fold_members.csv", dtype=identity_types)
    pd.testing.assert_frame_equal(
        units, pd.read_csv(base / "assay_graph_units.csv", dtype=identity_types)
    )
    pd.testing.assert_frame_equal(
        members, pd.read_csv(base / "fold_members.csv", dtype=identity_types)
    )
    for document, rows in members.groupby("outer_document"):
        held = units[units.document_id.eq(document)]
        fit = units[~units.document_id.eq(document) & ~units.connection_id.isin(held.connection_id)]
        assert set(rows[rows.role.eq("held_out")].unit_id) == set(held.unit_id)
        assert set(rows[rows.role.eq("fit")].unit_id) == set(fit.unit_id)
        assert len(rows) == len(units)
    predictions = pd.read_csv(run / "predictions.csv", dtype=identity_types)
    records = json.loads((run / "model_records.json").read_text(encoding="utf-8"))
    assert len(predictions) == 1662 and len(records) == 36
    assert len({(r["seed"], r["document_id"], r["condition"]) for r in records}) == 36
    assert set(predictions.condition) == set(CELLS[1:3])
    for record in records:
        rows = predictions[predictions.run_id.eq(record["run_id"])]
        saved = pd.read_csv(
            run / f"models/{record['run_id']}_predictions.csv", dtype=identity_types
        )
        pd.testing.assert_frame_equal(rows.reset_index(drop=True), saved)
        held = units[units.document_id.eq(record["document_id"])]
        assert len(rows) == len(held) and not rows.unit_id.duplicated().any()
        assert set(rows.unit_id) == set(held.unit_id)
        pd.testing.assert_frame_equal(
            rows.set_index("unit_id").loc[held.unit_id, units.columns.drop("unit_id")],
            held.set_index("unit_id"),
        )
        assert record["epochs"] == 30
        assert record["optimizer_steps"] == 30 * math.ceil(record["train_units"] / 16)
        assert record["encoder_max_abs_change"] > 0 and record["head_max_abs_change"] > 0
        assert record["readback_max_abs_error"] == 0
    metrics = pd.read_csv(run / "new_metrics_by_assay.csv")
    keys = ["condition", "seed", "document_id", "assay_id"]
    indexed = metrics.set_index(keys)
    assert len(metrics) == 102 and not indexed.index.duplicated().any()
    checked = 0
    for key, group in predictions.groupby(keys):
        budget = math.ceil(len(group) * 0.2)
        remaining, hits = budget, 0.0
        for _, tied in reversed(list(group.assign(s=group.score.round(12)).groupby("s"))):
            picked = min(remaining, len(tied))
            hits += picked * tied.label.mean()
            remaining -= picked
            if not remaining:
                break
        row = indexed.loc[key]
        assert row.budget == budget and row.n == len(group)
        assert row.positives == group.label.sum()
        assert row.primary_eligible == (len(group) >= 10 and group.label.nunique() == 2)
        assert abs(row.expected_hits_with_ties - hits) < 1e-10
        assert abs(row.precision - hits / budget) < 1e-10
        if group.label.sum():
            assert abs(row.enrichment - hits / (budget * group.label.mean())) < 1e-10
            assert abs(row.recall - hits / group.label.sum()) < 1e-10
        checked += 1
    assert checked == 102
    grid = pd.read_csv(run / "grid_metrics.csv")
    old = pd.read_csv(base / "metrics_by_assay.csv")
    old = old[old.method.eq("pretrained_trained")].reset_index(drop=True)
    reused = grid[grid.result_origin.eq("previous_factor_run_no_refit")]
    pd.testing.assert_frame_equal(
        reused[old.columns].reset_index(drop=True), old, check_exact=False, rtol=0, atol=1e-12
    )
    pd.testing.assert_frame_equal(
        grid[grid.result_origin.eq("new_cross_run")][metrics.columns].reset_index(drop=True),
        metrics,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )
    assert len(grid) == 204 and len(old) == 102
    assert set(grid.primary_eligible.unique()) == {True, False}
    primary = grid[grid.primary_eligible]
    assert set(primary.assay_id) == set(SOURCES)
    assert set(primary.condition) == set(CELLS)
    pivot = primary.pivot(
        index=["document_id", "assay_id", "seed"], columns="condition", values="enrichment"
    ).sort_index()
    pair = pd.read_csv(run / "paired_grid_effects.csv").set_index(pivot.index.names).sort_index()
    assert len(pair) == 15
    np.testing.assert_allclose(pair[CELLS], pivot[CELLS], rtol=0, atol=1e-12)
    a, b, c, d = (pivot[cell].to_numpy() for cell in CELLS)
    for key, expected in zip(EFFECTS, [b - a, d - c, c - a, d - b, d - b - c + a], strict=True):
        np.testing.assert_allclose(pair[key], expected, rtol=0, atol=1e-12)
    macro = primary.groupby(["condition", "seed"]).enrichment.mean()
    saved_macro = pd.read_csv(run / "macro_by_seed.csv").set_index(["condition", "seed"])
    np.testing.assert_allclose(macro, saved_macro.enrichment.loc[macro.index], rtol=0, atol=1e-12)
    return (
        grid,
        pair.reset_index(),
        {
            "status": "pass",
            "new_models": 36,
            "new_predictions": 1662,
            "new_assay_top_k_metrics_recomputed": checked,
            "old_diagonal_assay_rows_reused_and_verified": 102,
            "csv_metric_comparison_atol": 1e-12,
            "paired_rows_recomputed": 15,
            "fold_members_independently_rebuilt": True,
            "stored_model_predictions": "36 exact reload checks in production runner",
            "auc_ap_scope": "existing evaluator; not independently reimplemented here",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    grid, pairs, checked = check_outputs(args.run, args.base)
    checked["analysis_versions"] = {
        name: importlib.metadata.version(name) for name in ["matplotlib", "numpy", "pandas"]
    }
    primary = grid[grid.primary_eligible]
    macro = primary.groupby(["condition", "seed"]).enrichment.mean().reset_index()
    summary = macro.groupby("condition").enrichment.agg(["mean", "min", "max"]).loc[CELLS]
    summary.to_csv(args.output / "macro_enrichment_summary.csv")
    sources = primary.groupby(["assay_id", "condition"]).agg(
        enrichment=("enrichment", "mean"),
        mean_hits=("expected_hits_with_ties", "mean"),
        budget=("budget", "first"),
        n=("n", "first"),
        positives=("positives", "first"),
    )
    sources.to_csv(args.output / "source_means.csv")
    effect_macro = pairs.groupby("seed")[EFFECTS].mean()
    effect_source = pairs.groupby("assay_id")[EFFECTS].mean()
    effect_macro.to_csv(args.output / "effect_macro_by_seed.csv")
    effect_source.to_csv(args.output / "effect_source_means.csv")
    signs = []
    for scope, frame in [
        ("15_paired_source_seed_rows_not_independent", pairs),
        ("5_source_means", effect_source),
        ("3_seed_macro_means", effect_macro),
    ]:
        for effect in EFFECTS:
            values = frame[effect].to_numpy()
            signs.append(
                {
                    "scope": scope,
                    "effect": effect,
                    "mean": float(values.mean()),
                    "positive": int((values > 1e-12).sum()),
                    "zero": int((np.abs(values) <= 1e-12).sum()),
                    "negative": int((values < -1e-12).sum()),
                }
            )
    pd.DataFrame(signs).to_csv(args.output / "effect_sign_counts.csv", index=False)
    old_history = pd.read_csv(args.base / "training_history.csv")
    old_history = old_history[old_history.method.eq("pretrained_trained")].copy()
    old_history["condition"] = old_history.learning_rate.map({0.0001: CELLS[0], 0.001: CELLS[3]})
    history = pd.concat([old_history, pd.read_csv(args.run / "training_history.csv")])
    loss = history.groupby(["condition", "epoch"]).train_weighted_bce.agg(["mean", "min", "max"])
    loss.to_csv(args.output / "training_loss_summary.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.20,
        right=0.94,
        top=0.86,
        bottom=0.13,
        wspace=0.62,
        hspace=0.54,
        width_ratios=[1, 1.15],
    )
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.suptitle(
        "Where does learning-rate sensitivity arise?",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.07,
        0.905,
        "GPR84 antagonist potency | one pretrained encoder | "
        "same initial states, splits and 30 epochs\n"
        "Low = 1e-4; high = 1e-3. Two saved diagonal settings + two new cross settings; "
        "all 3 seeds shown.",
        color="#4c5660",
        fontsize=11,
    )
    norm = TwoSlopeNorm(vmin=0.65, vcenter=1, vmax=1.35)
    ax = axs[0]
    positions = [[CELLS[0], CELLS[1]], [CELLS[2], CELLS[3]]]
    heat = np.array([[summary.loc[cell, "mean"] for cell in row] for row in positions])
    ax.imshow(heat, cmap="RdBu", norm=norm, aspect="auto")
    for i in range(2):
        for j in range(2):
            cell = positions[i][j]
            s = summary.loc[cell]
            status = "saved" if cell in [CELLS[0], CELLS[3]] else "new"
            ax.text(
                j,
                i,
                f"{s['mean']:.3f}\n{s['min']:.3f}–{s['max']:.3f}\n({status})",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if abs(s["mean"] - 1) > 0.23 else "#17202a",
            )
    ax.set_xticks([0, 1], ["Head low", "Head high"])
    ax.set_yticks([0, 1], ["Encoder low", "Encoder high"])
    ax.set_title("A  Macro enrichment: mean + seed range", loc="left", weight="bold", fontsize=12)
    ax = axs[1]
    source_heat = sources.enrichment.unstack("condition").loc[list(SOURCES), CELLS]
    span = max(0.35, float(np.abs(source_heat.to_numpy() - 1).max()))
    im = ax.imshow(
        source_heat,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=1 - span, vcenter=1, vmax=1 + span),
        aspect="auto",
    )
    for i in range(5):
        for j in range(4):
            v = source_heat.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(v - 1) > span * 0.65 else "#17202a",
            )
    ax.set_yticks(range(5), list(SOURCES.values()), fontsize=9)
    ax.set_xticks(
        range(4), ["H low\nE low", "H high\nE low", "H low\nE high", "H high\nE high"], fontsize=9
    )
    ax.set_title("B  Each source: mean of 3 seeds", loc="left", weight="bold", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Enrichment")
    ax = axs[2]
    effect_labels = [
        "Head: E low",
        "Head: E high",
        "Encoder: H low",
        "Encoder: H high",
        "Interaction",
    ]
    for i, col in enumerate(EFFECTS):
        values = effect_macro[col].to_numpy()
        ax.plot([values.min(), values.max()], [i, i], color="#a3bac9", lw=4)
        ax.scatter(values, i + np.array([-0.08, 0, 0.08]), s=25, color="#376c9a", zorder=3)
        ax.scatter([values.mean()], [i], s=24, marker="D", color="#17202a", zorder=4)
    ax.axvline(0, color="#59636a", ls="--", lw=1)
    ax.set_yticks(range(5), effect_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in macro enrichment (high minus low)")
    ax.set_title("C  Conditional effects and interaction", loc="left", weight="bold", fontsize=12)
    ax = axs[3]
    for cell, label, color in zip(CELLS, LABELS, COLORS, strict=True):
        rows = loss.loc[cell]
        ax.plot(rows.index, rows["mean"], color=color, marker="o", lw=2, label=label)
    ax.set_xticks([0, 10, 30])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training weighted BCE")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("D  Training fit: 6 folds x 3 seeds", loc="left", weight="bold", fontsize=12)
    fig.text(
        0.07,
        0.03,
        "A–C: 5 primary assays, top 20% within each assay; random-selection enrichment = 1. "
        "Ranges are seeds, not confidence intervals.\n"
        "C: each dot is one seed's 5-assay mean; diamonds are 3-seed means. "
        "Interaction = F11 - F10 - F01 + F00 on the enrichment scale.\n"
        "Exploratory follow-up using previously seen labels; IC50 <= 1 uM in antagonist assays. "
        "No new compound activity is established.",
        fontsize=9,
        color="#4c5660",
    )
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(args.output / f"training_cross.{extension}", dpi=300)
    plt.close(fig)
    (args.output / "independent_readback.json").write_text(
        json.dumps(checked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string())
    print(effect_macro.to_string())
    print(pd.DataFrame(signs).to_string(index=False))
    print(json.dumps(checked, ensure_ascii=False))


if __name__ == "__main__":
    main()
