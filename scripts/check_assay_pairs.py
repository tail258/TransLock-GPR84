"""以原作者报告值核对来源内两种读出，避免平均数据库重复行。"""

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from translock_screening.cause_audit import exact_mean_sem  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-csv", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        args.author_csv,
        args.pilot / "admitted_measurements.csv",
        args.audit / "bridge_candidates.csv",
        args.audit / "summary.json",
    ]
    before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    source = pd.read_csv(args.author_csv, keep_default_na=False)
    measurements = pd.read_csv(paths[1], keep_default_na=False)
    allowed = set(measurements.query("assay_chembl_id == 'CHEMBL6137890'")["standardized_smiles"])
    rows = []
    for record in source.to_dict("records"):
        left = exact_mean_sem(record["hGPR84 cAMP pIC50 ± SEM"])
        right = exact_mean_sem(record["PRESTO-Tango pIC50 ± SEM"])
        if left is None or right is None:
            continue
        mol = Chem.MolFromSmiles(record["SMILES"])
        if mol is None:
            raise ValueError("原作者结构无法解析")
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles not in allowed:
            raise ValueError(f"两读出样品不在当前准入范围：{record['Compound']}")
        selected = measurements.loc[
            (measurements["standardized_smiles"] == smiles)
            & measurements["assay_chembl_id"].isin(["CHEMBL6137890", "CHEMBL6137896"])
        ]
        rows.append(
            {
                "compound": record["Compound"],
                "smiles": smiles,
                "camp_pic50": left[0],
                "camp_sem": left[1],
                "tango_pic50": right[0],
                "tango_sem": right[1],
                "delta_tango_minus_camp": right[0] - left[0],
                "camp_strong": left[0] >= 6,
                "tango_strong": right[0] >= 6,
                "activity_ids": ";".join(selected["activity_id"].astype(str)),
                "doi": "10.1021/acs.jmedchem.4c02335",
            }
        )
    data = pd.DataFrame(rows)
    if data.empty or data["smiles"].duplicated().any():
        raise ValueError("配对为空或图身份不唯一")
    data.to_csv(args.output / "author_paired_values.csv", index=False)
    comparisons = []
    for left, right in combinations(rows, 2):
        dx = round(left["camp_pic50"] - right["camp_pic50"], 8)
        dy = round(left["tango_pic50"] - right["tango_pic50"], 8)
        comparisons.append(
            {
                "compound_left": left["compound"],
                "compound_right": right["compound"],
                "camp_delta": dx,
                "tango_delta": dy,
                "status": "tie"
                if dx == 0 or dy == 0
                else "concordant"
                if dx * dy > 0
                else "discordant",
            }
        )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output / "descriptive_order_pairs.csv", index=False)
    # 独立按原测量重建共享图，核对前一脚本的关键统计。
    by_graph = {}
    for r in measurements.to_dict("records"):
        by_graph.setdefault(r["standardized_smiles"], set()).add(
            (r["document_chembl_id"], r["assay_chembl_id"])
        )
    cross_graphs = {s for s, groups in by_graph.items() if len({d for d, _ in groups}) > 1}
    audit_summary = json.loads(paths[3].read_text(encoding="utf-8"))
    assert len(cross_graphs) == audit_summary["bridge_graphs_cross_document"]
    bridge_rows = pd.read_csv(paths[2])
    cross = bridge_rows.loc[bridge_rows["cross_document"]]
    assert set(cross["smiles"]) == cross_graphs
    assert all(
        math.isclose(
            r["log10_ic50_right_over_left"],
            math.log10(
                json.loads(r["values_right"])[0]["value_nm"]
                / json.loads(r["values_left"])[0]["value_nm"]
            ),
            abs_tol=1e-12,
        )
        for r in cross.to_dict("records")
    )
    summary = {
        "author_pairs": len(data),
        "same_strong_class": int((data["camp_strong"] & data["tango_strong"]).sum()),
        "class_flips": int((data["camp_strong"] != data["tango_strong"]).sum()),
        "descriptive_order_counts_not_independent_samples": comparison["status"]
        .value_counts()
        .to_dict(),
        "delta_tango_minus_camp_range": [
            float(data["delta_tango_minus_camp"].min()),
            float(data["delta_tango_minus_camp"].max()),
        ],
        "cross_document_graphs_verified": len(cross_graphs),
        "cross_document_measurements_all_exact_and_strong": all(
            item["relation"] == "=" and item["value_nm"] <= 1000
            for r in cross.to_dict("records")
            for side in ["left", "right"]
            for item in json.loads(r[f"values_{side}"])
        ),
        "status": "exploratory_reported_point_values_not_causal_or_significance_test",
    }
    fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    ax.axvline(6, color="#d3a13a", linewidth=1, linestyle="--")
    ax.axhline(6, color="#d3a13a", linewidth=1, linestyle="--")
    ax.plot([5.8, 8.2], [5.8, 8.2], color="#b3bac5", linewidth=1)
    ax.errorbar(
        data["camp_pic50"],
        data["tango_pic50"],
        xerr=data["camp_sem"],
        yerr=data["tango_sem"],
        fmt="o",
        color="#296d88",
        ecolor="#86aaba",
        capsize=3,
    )
    label_offsets = {
        "Cmpd 140": (-67, 10),
        "37g": (-34, 10),
        "40g": (-31, -19),
        "40h": (9, 0),
        "40d": (9, 10),
    }
    for r in rows:
        offset = label_offsets.get(r["compound"], (7, -14 if r["compound"] == "GLPG1205" else 7))
        ax.annotate(
            r["compound"],
            (r["camp_pic50"], r["tango_pic50"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlim=(5.8, 8.2),
        ylim=(5.8, 8.2),
        xlabel="Human GPR84 cAMP pIC50",
        ylabel="PRESTO-Tango pIC50",
        title="Same compounds, two functional readouts",
    )
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.13,
        0.015,
        "Published means +/- SEM; dashed lines: 1 uM threshold.\n"
        "Ieremias et al. 2025, DOI: 10.1021/acs.jmedchem.4c02335",
        fontsize=8,
        color="#596575",
    )
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.savefig(args.output / "paired_readouts.png", dpi=220)
    fig.savefig(args.output / "paired_readouts.svg")
    plt.close(fig)
    after = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    assert before == after
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    shutil.copy2(root / "src/translock_screening/cause_audit.py", args.output / "cause_audit.py")
    shutil.copy2(root / "docs/cause-audit-protocol.md", args.output / "protocol.md")
    for name, value in [
        ("summary", summary),
        (
            "manifest",
            {
                "completed_utc": datetime.now(UTC).isoformat(),
                "input_size_mtime_ns": before,
                "input_size_mtime_unchanged": True,
                "argv": sys.argv,
                "python": sys.version,
                "versions": {
                    n: importlib.metadata.version(n) for n in ["pandas", "rdkit", "matplotlib"]
                },
                "hashes_computed": False,
                "model_fitted": False,
            },
        ),
    ]:
        (args.output / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
