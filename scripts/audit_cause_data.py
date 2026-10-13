"""从已有试点测量核查可辨识性，将全部新结果写入新目录。"""

import argparse
import importlib.metadata
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from translock_screening.cause_audit import build_cause_audit  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--assays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        args.pilot / "admitted_measurements.csv",
        args.pilot / "assay_graph_units.csv",
        args.assays,
    ]
    before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    measurements, units = (pd.read_csv(p, keep_default_na=False) for p in paths[:2])
    tables = build_cause_audit(measurements, units)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    raw_assays = {
        r["assay_chembl_id"]: r
        for r in (json.loads(line) for line in args.assays.read_text(encoding="utf-8").splitlines())
    }
    assay_rows = []
    for assay, group in measurements.groupby("assay_chembl_id", sort=True):
        raw = raw_assays[assay]
        assay_rows.append(
            {
                "assay_id": assay,
                "document_id": group.iloc[0]["document_chembl_id"],
                "document_doi": group.iloc[0]["document_doi"],
                "measurement_rows": len(group),
                "graphs": group["standardized_smiles"].nunique(),
                "exact_rows": int((group["standard_relation"] == "=").sum()),
                "description": raw.get("description"),
                **{
                    key: raw.get(key)
                    for key in [
                        "assay_type",
                        "assay_organism",
                        "assay_cell_type",
                        "assay_tissue",
                        "assay_subcellular_fraction",
                        "bao_format",
                        "variant_sequence",
                    ]
                },
            }
        )
    pd.DataFrame(assay_rows).to_csv(args.output / "assay_metadata.csv", index=False)
    bridges = tables["bridge_candidates"]
    cross = bridges.loc[bridges["cross_document"] == True]  # noqa: E712
    scaffold_groups = tables["scaffold_groups"]
    eligible = scaffold_groups.loc[
        (scaffold_groups["graphs"] > 1) & scaffold_groups["usable_as_scaffold_group"]
    ]
    summary = {
        "measurements": len(measurements),
        "units": len(units),
        "encoded_graphs": units["smiles"].nunique(),
        "assays": units["assay_id"].nunique(),
        "documents": units["document_id"].nunique(),
        "bridge_graphs_any_assay": bridges["smiles"].nunique(),
        "bridge_graphs_cross_document": cross["smiles"].nunique(),
        "assay_edges_cross_document": len(tables["assay_edges"].query("cross_document == True")),
        "max_shared_graphs_per_cross_document_assay_edge": int(
            tables["assay_edges"].query("cross_document == True")["shared_graphs"].max()
        )
        if len(cross)
        else 0,
        "nonempty_scaffold_groups_with_multiple_graphs": len(eligible),
        "mixed_label_scaffold_groups": len(eligible.query("strong_units > 0 and weak_units > 0")),
        "scope": "existing_admitted_pilot_only_not_all_public_gpr84_data",
        "identity": "encoded_graph_candidates_not_confirmed_author_samples",
    }
    after = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    if before != after:
        raise RuntimeError("输入文件大小或修改时间变化")
    root = Path(__file__).resolve().parents[1]
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in [
        Path(__file__),
        root / "src/translock_screening/cause_audit.py",
        root / "docs/cause-audit-protocol.md",
    ]:
        shutil.copy2(path, snapshot / path.name)
    manifest = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "inputs_size_mtime_ns": before,
        "input_size_mtime_unchanged": True,
        "python": sys.version,
        "versions": {n: importlib.metadata.version(n) for n in ["pandas", "rdkit"]},
        "argv": sys.argv,
        "hashes_computed": False,
        "model_fitted": False,
    }
    for name, value in [("summary", summary), ("manifest", manifest)]:
        (args.output / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
