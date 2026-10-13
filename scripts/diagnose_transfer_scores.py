"""对已存模型作固定logit排序诊断，不重新训练，不替代概率主结果。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translock_screening.benchmark import _features, _score_assay  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review["status"] != "passed":
        raise ValueError("主结果独立核查未通过")
    diagnostics = [
        value
        for fold in review["fold_numeric_checks"]
        for key, value in fold.items()
        if key.endswith("_score_diagnostics")
    ]
    if not any(row["probability_tie_groups_with_distinct_logits"] for row in diagnostics):
        raise ValueError("没有触发预定的数值诊断条件")
    shutil.copy2(args.protocol, args.output / "diagnostic_protocol.md")
    units = pd.read_csv(args.run / "assay_graph_units.csv")
    models = joblib.load(args.run / "fold_models.joblib")
    config = yaml.safe_load((args.run / "frozen_config.yaml").read_text(encoding="utf-8"))[
        "benchmark"
    ]
    rows = pd.read_csv(args.run / "embedding_rows.csv")
    vectors = np.load(args.run / "unique_embeddings.npy", allow_pickle=False)
    mapping = dict(zip(rows.smiles, rows.embedding_row, strict=True))
    aligned = vectors[[mapping[s] for s in units.smiles]].astype(np.float64)
    _, features = _features(units, config["radius"], config["include_chirality"])
    primary = pd.read_csv(args.run / "predictions.csv")
    predictions, metrics = [], []
    for document, fold in models.items():
        indices = np.flatnonzero(units.document_id.eq(document).to_numpy())
        held = units.iloc[indices]
        for method, fitted in fold.items():
            if fitted["vectorizer"] is None:
                matrix = aligned[indices]
            else:
                matrix = fitted["vectorizer"].transform([features[i] for i in indices])
            transformed = fitted["scaler"].transform(matrix)
            logits = fitted["model"].decision_function(transformed)
            probabilities = fitted["model"].predict_proba(transformed)[:, 1]
            expected = primary[primary.document_id.eq(document) & primary.method.eq(method)]
            expected = expected.set_index("unit_id").loc[held.unit_id, "score"].to_numpy()
            if not np.allclose(probabilities, expected, rtol=0, atol=1e-10):
                raise ValueError("已存模型不复现主概率结果")
            if not np.isfinite(logits).all():
                raise ValueError("模型原始决策分数非有限")
            for index, row in enumerate(held.itertuples()):
                predictions.append(
                    {
                        "unit_id": row.unit_id,
                        "document_id": document,
                        "assay_id": row.assay_id,
                        "method": method,
                        "label": row.label,
                        "probability": probabilities[index],
                        "score": logits[index],
                        "score_kind": "decision_function_posthoc_numerical_diagnostic",
                    }
                )
            for assay in sorted(held.assay_id.unique()):
                within = held.assay_id.eq(assay).to_numpy()
                labels = held.label.to_numpy()[within]
                eligible = (
                    len(labels) >= config["minimum_primary_assay_units"] and len(set(labels)) == 2
                )
                for fraction in config["budget_fractions"]:
                    metrics.append(
                        {
                            "document_id": document,
                            "assay_id": assay,
                            "method": method,
                            "budget_fraction": fraction,
                            "primary_eligible": eligible,
                            **_score_assay(labels, logits[within], fraction),
                        }
                    )
    data = pd.DataFrame(metrics)
    main = data[data.primary_eligible & data.budget_fraction.eq(config["primary_budget_fraction"])]
    macro = (
        main.groupby("method")[
            ["precision", "recall", "enrichment", "roc_auc", "average_precision"]
        ]
        .mean()
        .reset_index()
    )
    summary = {
        "status": "posthoc_numerical_diagnostic_not_replacement_primary",
        "source_run": str(args.run),
        "review": str(args.review),
        "score_kind": "decision_function",
        "models_refitted": False,
        "feature_or_scaling_changed": False,
        "primary_result_preserved": True,
        "primary_assay_macro": macro.to_dict("records"),
        "primary_assay_count": int(main.assay_id.nunique()),
        "primary_document_count": int(main.document_id.nunique()),
        "uncertainty": "稳定logit保留排序；仍可能严重外推，不能解释成活性概率",
        "hashes_computed": False,
    }
    pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
    data.to_csv(args.output / "metrics_by_assay.csv", index=False)
    main.to_csv(args.output / "primary_metrics.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
