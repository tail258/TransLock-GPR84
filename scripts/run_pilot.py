"""从原测量审计表重建任务并运行固定回顾检索，所有输出写到新目录。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translock_screening.benchmark import run_source_benchmark  # noqa: E402
from translock_screening.measurements import admit_measurements, assay_graph_units  # noqa: E402


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    shutil.copy2(args.config, args.output / "frozen_config.yaml")
    audit_path = args.storage_root / config["inputs"]["audit"]
    raw_path = args.storage_root / config["inputs"]["raw_activities"]
    quarantine_path = args.legacy_root / config["inputs"]["quarantine"]
    input_paths = [audit_path, raw_path, quarantine_path]
    stats_before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in input_paths}
    audit = pd.read_parquet(audit_path)
    raw = {
        int(r["activity_id"]): r
        for r in (json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines())
    }
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    policy = config["policy"]
    policy["excluded_alias_smiles"] = [r["author_smiles"] for r in quarantine["records"]]
    policy["excluded_alias_smiles"] += (
        audit.loc[
            audit["molecule_chembl_id"].isin(policy["excluded_molecule_ids"]), "standardized_smiles"
        ]
        .dropna()
        .unique()
        .tolist()
    )
    measurements, exclusions, data_summary = admit_measurements(audit, raw, policy)
    units, conflicts = assay_graph_units(measurements)
    measurements.to_csv(args.output / "admitted_measurements.csv", index=False)
    exclusions.to_csv(args.output / "exclusions.csv", index=False)
    units.to_csv(args.output / "assay_graph_units.csv", index=False)
    conflicts.to_csv(args.output / "conflicts.csv", index=False)
    data_summary.update(
        {
            "units": len(units),
            "conflicting_units": len(conflicts),
            "source_documents": int(units["document_id"].nunique()),
            "assays": int(units["assay_id"].nunique()),
            "input_graphs": int(units["smiles"].nunique()),
            "unit_class_counts": units["label"].value_counts().to_dict(),
        }
    )
    write_json(args.output / "data_summary.json", data_summary)
    summary, predictions, metrics = run_source_benchmark(units, config["benchmark"])
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics_by_assay.csv", index=False)
    write_json(args.output / "evaluation.json", summary)
    stats_after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in input_paths}
    project_root = Path(__file__).resolve().parents[1]
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in [
        Path(__file__),
        *sorted((project_root / "src/translock_screening").glob("*.py")),
        project_root / "pyproject.toml",
        project_root / "uv.lock",
    ]:
        shutil.copy2(path, snapshot / path.name)
    write_json(
        args.output / "manifest.json",
        {
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "status": summary["status"],
            "inputs": list(stats_before),
            "input_size_mtime_unchanged": stats_before == stats_after,
            "versions": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "rdkit",
                    "scikit-learn",
                    "scipy",
                    "pyarrow",
                    "pyyaml",
                ]
            },
            "python": sys.version,
            "argv": sys.argv,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "hashes_computed": False,
            "expanded_policy": policy,
            "original_inputs_modified": False,
            "gpu_used": False,
        },
    )
    print(json.dumps({"data": data_summary, "evaluation": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
