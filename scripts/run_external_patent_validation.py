"""专利外部评价：先准入，再加载既有模型评分，禁止在此训练。"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from translock_screening.external_validation import (  # noqa: E402
    collapse_connections,
    fixed_metrics,
    identity_keys,
    patent_label,
)
from translock_screening.screening_application import (  # noqa: E402
    nearest_support,
    normalize,
    properties,
)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def admit(args):
    out = args.output
    out.mkdir(exist_ok=False)
    inputs = args.admission
    manual = pd.DataFrame(json.loads((inputs / "manual_structures_all_v1.json").read_text()))
    # 用第二次视觉复核建议确认已修正的结构，无预测参与此检查。
    review = pd.read_csv(
        inputs / "first_structures_independent_review/independent_review.csv"
    ).fillna("")
    for row in review.itertuples():
        proposed = row.proposed_smiles or row.reviewed_smiles
        actual = manual.set_index("sample_id").loc[row.sample_id, "manual_smiles"]
        assert Chem.MolToSmiles(Chem.MolFromSmiles(proposed)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(actual)
        ), row.sample_id
    labels = pd.read_csv(
        args.storage_root
        / "artifacts/track3_rebuild/independent_validation_source_search_2026-10-03_v1"
        / "patent_activity_table_transcription.csv"
    )
    labels = labels[labels.patent.eq("WO2024028364A1")]
    assert len(manual) == len(labels) == 61
    assert set(manual.sample_id) == set(labels.compound_id)
    labels = labels.rename(columns={"compound_id": "sample_id", "pdf_page": "activity_pdf_page"})
    frame = manual.merge(
        labels[["sample_id", "reported_bin", "activity_pdf_page"]], validate="one_to_one"
    )
    frame["label"] = frame.reported_bin.map(patent_label)
    identities = pd.DataFrame([identity_keys(s) for s in frame.manual_smiles])
    frame = pd.concat([frame, identities], axis=1)
    props = pd.DataFrame([properties(normalize(s)) for s in frame.manual_smiles])
    frame = pd.concat([frame, props], axis=1)
    sources = {
        "training": (args.models / "training_members.csv", "smiles", "unit_id"),
        "chembl_seen": (
            args.storage_root
            / "artifacts/track3_rebuild/screening_identity_bridge_2026-10-01_v1"
            / "legacy_identity.csv",
            "standardized_smiles",
            "molecule_chembl_id",
        ),
        "bindingdb_seen": (
            args.storage_root
            / "artifacts/track3_rebuild/candidate_public_validation_2026-10-02_v1"
            / "bindingdb_gpr84_structures_v2.csv",
            "smiles",
            "row",
        ),
    }
    matches, source_counts, source_ids = [], {}, {}
    for name, (path, column, id_col) in sources.items():
        source = pd.read_csv(path)
        source_counts[name] = len(source)
        cache = {s: identity_keys(s) for s in source[column].dropna().unique()}
        rows = [
            {"source_id": str(row[id_col]), **cache[row[column]]}
            for row in source.to_dict("records")
            if isinstance(row[column], str)
        ]
        source_ids[name] = pd.DataFrame(rows)
        keys = set(source_ids[name].broad_identity)
        frame[f"overlap_{name}"] = frame.broad_identity.isin(keys)
        for row in frame.itertuples():
            matched = source_ids[name][source_ids[name].broad_identity.eq(row.broad_identity)]
            for old in matched.itertuples():
                matches.append(
                    {
                        "sample_id": row.sample_id,
                        "source": name,
                        "source_id": old.source_id,
                        "exact": row.smiles == old.smiles,
                        "connection": row.connection == old.connection,
                    }
                )
    frame["within_scope"] = frame.heavy_atoms.between(15, 43)
    for i, s in enumerate(frame.smiles):
        mol = Chem.MolFromSmiles(s)
        frame.loc[i, "within_scope"] &= all(
            a.GetSymbol() in {"H", "Br", "C", "Cl", "F", "N", "O", "P", "S"}
            and a.GetNumRadicalElectrons() == 0
            for a in mol.GetAtoms()
        )
    conflicts = frame.groupby("smiles").label.nunique()
    conflict_keys = set(conflicts[conflicts.gt(1)].index)
    frame["status"] = "admitted"
    frame.loc[~frame.within_scope, "status"] = "outside_existing_scope"
    frame.loc[frame.smiles.isin(conflict_keys), "status"] = "same_graph_label_conflict"
    frame.loc[frame[[f"overlap_{x}" for x in sources]].any(axis=1), "status"] = (
        "previously_seen_identity"
    )
    duplicate = frame[frame.status.eq("admitted")].duplicated("smiles")
    frame.loc[duplicate[duplicate].index, "status"] = "duplicate_complete_graph"
    frame.to_csv(out / "admission_all_61.csv", index=False)
    pd.DataFrame(
        matches, columns=["sample_id", "source", "source_id", "exact", "connection"]
    ).to_csv(out / "overlap_matches.csv", index=False)
    accepted = frame[frame.status.eq("admitted")].copy()
    accepted.to_csv(out / "evaluation_units.csv", index=False)
    collapse_connections(accepted).to_csv(out / "connection_sensitivity_units.csv", index=False)
    summary = {
        "created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_counts": source_counts,
        "input_samples": len(frame),
        "status_counts": frame.status.value_counts().to_dict(),
        "admitted_classes": accepted.label.value_counts().to_dict(),
        "overlap_rows": len(matches),
        "distinct_connections": accepted.isotope_free.nunique(),
        "scores_generated": False,
    }
    save(out / "admission_summary.json", summary)
    shutil.copy2(
        ROOT / "docs/external-patent-validation-protocol-2026-10-03.md", out / "fixed_protocol.md"
    )
    writer = Chem.SDWriter(str(out / "source_structures.sdf"))
    for row in frame.itertuples():
        mol = Chem.MolFromSmiles(row.manual_smiles)
        mol.SetProp("_Name", row.sample_id)
        mol.SetProp("source_pdf_page", str(row.pdf_page))
        mol.SetProp("activity_bin", row.reported_bin)
        writer.write(mol)
    writer.close()
    for start in range(0, len(frame), 12):
        chunk = frame.iloc[start : start + 12]
        image = Draw.MolsToGridImage(
            [Chem.MolFromSmiles(s) for s in chunk.manual_smiles],
            molsPerRow=3,
            subImgSize=(500, 330),
            legends=[
                f"{r.sample_id}; source PDF p{r.pdf_page}; {r.reported_bin}"
                for r in chunk.itertuples()
            ],
        )
        image.save(out / f"structures_{start + 1:02d}.png")
    print(json.dumps(summary), flush=True)


def score(args):
    from translock_screening.training_factor import (
        GraphCache,
        build_model,
        configure_torch,
        predict,
    )

    out = args.output
    out.mkdir(exist_ok=False)
    frame = pd.read_csv(args.admission / "evaluation_units.csv")
    assert frame.status.eq("admitted").all() and set(frame.label) == {0, 1}
    config = json.loads((args.models / "original_training_config.json").read_text())
    assert (
        config["training"]["primary_learning_rate"] == 0.0001 and config["training"]["epochs"] == 30
    )
    assert config["training"]["seeds"] == [20261001, 20261002, 20261003]
    train = pd.read_csv(args.models / "training_members.csv")
    fit_predictions = pd.read_csv(args.models / "training_fit_predictions.csv")
    torch = configure_torch(config["encoder"])
    weight = args.storage_root / "downloads/chemeleon_probe_2026-10-01_v1/weights/chemeleon_mp.pt"
    payload = torch.load(weight, weights_only=True, map_location="cpu")
    device = config["encoder"]["device"]
    graphs = GraphCache(frame.smiles.tolist(), device)
    check_graphs = GraphCache(train.smiles.iloc[:16].tolist(), device)
    restore_checks = []
    model_files = [args.models / f"model_{s}.pt" for s in config["training"]["seeds"]]
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in model_files]
    for seed, path in zip(config["training"]["seeds"], model_files, strict=True):
        model = build_model(payload, True, True, seed, config["training"], device)
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        restored = predict(model, check_graphs, np.arange(16), 16)
        stored = (
            fit_predictions[fit_predictions.seed.eq(seed)]
            .set_index("unit_id")
            .loc[train.unit_id.iloc[:16], "training_logit"]
            .to_numpy()
        )
        np.testing.assert_allclose(restored, stored, rtol=0, atol=1e-5)
        restore_checks.append(
            {"seed": seed, "maximum_abs_difference": float(np.max(np.abs(restored - stored)))}
        )
        frame[f"score_{seed}"] = predict(model, graphs, np.arange(len(frame)), 16)
        del model
        torch.cuda.empty_cache()
    assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in model_files]
    support = nearest_support(frame.smiles.tolist(), train)
    frame = pd.concat([frame, support], axis=1)
    train_scaffolds = {properties(normalize(s))["scaffold"] for s in train.smiles.unique()}
    frame["training_scaffold_overlap"] = frame.scaffold.isin(train_scaffolds)
    columns = [f"score_{s}" for s in config["training"]["seeds"]] + ["nearest_strong_similarity"]
    metrics = []
    for view, data in [("primary", frame), ("connection_collapsed", collapse_connections(frame))]:
        for col in columns:
            metrics.append({"view": view, "method": col, **fixed_metrics(data.label, data[col])})
    for col in columns:
        # 名次表展示并列范围，主预算仍使用无标签的并列期望。
        rounded = frame[col].round(12)
        frame[f"rank_min_{col}"] = rounded.rank(method="min", ascending=False).astype(int)
        frame[f"rank_max_{col}"] = rounded.rank(method="max", ascending=False).astype(int)
    frame.to_csv(out / "predictions.csv", index=False)
    frame[frame.label.eq(0)].to_csv(out / "weaker_samples_ranks.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    summary = {
        "created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "n": len(frame),
        "model_restore_checks": restore_checks,
        "models_unchanged_by_size_mtime": True,
        "training_scaffold_overlaps": int(frame.training_scaffold_overlap.sum()),
        "nearest_any_similarity": frame.nearest_any_similarity.describe().to_dict(),
        "metrics": metrics,
        "retraining": False,
    }
    save(out / "summary.json", summary)
    snapshot = out / "code_snapshot"
    snapshot.mkdir()
    for p in [
        Path(__file__),
        ROOT / "src/translock_screening/external_validation.py",
        ROOT / "docs/external-patent-validation-protocol-2026-10-03.md",
        args.models / "original_training_config.json",
    ]:
        shutil.copy2(p, snapshot / p.name)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["admit", "score"], required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"admit": admit, "score": score}[args.stage](args)
