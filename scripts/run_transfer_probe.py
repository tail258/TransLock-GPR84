"""运行冻结表示探针和第二轮来源留出，完整保存输入、失败与拟合成员。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translock_screening.chemeleon_encoder import FrozenCheMeleon  # noqa: E402


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def probes(config, root):
    panel_path = root / config["inputs"]["triazine_panel"]
    panel = {row["author_id"]: row for row in json.loads(panel_path.read_text(encoding="utf-8"))}
    groups_path = root / config["inputs"]["binop_groups"]
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    ids = config["probes"]["binop_sample_pair"]
    group = next(row for row in groups if set(row["sample_ids"]) == set(ids))
    members = {row["sample_id"]: row for row in group["members"]}
    result = [
        {
            "name": "tetrahedral",
            "smiles": config["probes"]["tetrahedral"],
            "expected": "distinct_if_encoded_signal_survives",
            "source": "synthetic_structure_probe",
        }
    ]
    result.extend(
        {
            "name": f"triazine_{a}_{b}",
            "smiles": [panel[a]["smiles"], panel[b]["smiles"]],
            "expected": "report_distinguishability",
            "source": str(panel_path),
        }
        for a, b in config["probes"]["triazine_author_pairs"]
    )
    result.append(
        {
            "name": "binop_15S_15R",
            "smiles": [members[k]["canonical_smiles"] for k in ids],
            "expected": "same_missing_sample_identity",
            "source": str(groups_path),
        }
    )
    return result


def run(args):
    from translock_screening.transfer_benchmark import run_transfer_benchmark

    started = datetime.now(UTC).isoformat()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if importlib.metadata.version("chemprop") != config["encoder"]["chemprop"]:
        raise ValueError("Chemprop版本与冻结配置不符")
    pilot = args.storage_root / config["inputs"]["pilot"]
    units = pd.read_csv(pilot / "assay_graph_units.csv")
    checkpoint = args.storage_root / config["inputs"]["weight"]
    paths = [
        args.config,
        pilot / "assay_graph_units.csv",
        pilot / "metrics_by_assay.csv",
        checkpoint,
        args.storage_root / config["inputs"]["triazine_panel"],
        args.storage_root / config["inputs"]["binop_groups"],
    ]
    input_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    shutil.copy2(args.config, args.output / "frozen_config.yaml")
    shutil.copy2(pilot / "assay_graph_units.csv", args.output / "assay_graph_units.csv")
    probe_inputs = probes(config, args.storage_root)
    write_json(args.output / "probe_inputs.json", probe_inputs)
    before = perf_counter()
    encoder = FrozenCheMeleon(checkpoint, config["encoder"])
    encoder_load_seconds = perf_counter() - before
    print("编码器已加载；开始固定结构和接口检查。", flush=True)
    rtol, atol = config["encoder"]["probe_rtol"], config["encoder"]["probe_atol"]
    probe_results = []
    for item in probe_inputs:
        values = encoder.encode(item["smiles"])
        delta = values[0] - values[1]
        same = bool(np.allclose(values[0], values[1], rtol=rtol, atol=atol))
        probe_results.append(
            {
                "name": item["name"],
                "within_numeric_tolerance": same,
                "max_abs_difference": float(np.abs(delta).max()),
                "l2_difference": float(np.linalg.norm(delta)),
                "expected": item["expected"],
            }
        )
        if item["expected"] == "same_missing_sample_identity" and not same:
            raise ValueError("相同输入未产生相同表示")
    small = [s for item in probe_inputs for s in item["smiles"]]
    custom = encoder.encode(small)
    official = encoder.official_fingerprint(small)
    if not np.allclose(custom, official, rtol=rtol, atol=atol):
        raise ValueError("消息传递聚合不等价于官方fingerprint路径")
    for index, smiles in enumerate(small):
        if not np.allclose(custom[index], encoder.encode([smiles])[0], rtol=rtol, atol=atol):
            raise ValueError("固定小样单条/批次表示不一致")
    write_json(
        args.output / "representation_checks.json",
        {
            "status": "numeric_checks_pass",
            "probes": probe_results,
            "official_path_max_abs_difference": float(np.abs(custom - official).max()),
            "single_vs_batch_pass": True,
            "rtol": rtol,
            "atol": atol,
            "input_atom_bond_dimensions": [72, 14],
            "output_dimensions": encoder.dimensions,
            "encoder_parameters": encoder.parameter_count,
            "hyper_parameters": {
                key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
                for key, value in encoder.hyper_parameters.items()
            },
        },
    )
    unique_smiles = units["smiles"].drop_duplicates().tolist()
    print(f"数值检查通过；生成{len(unique_smiles)}个固定图的表示。", flush=True)
    before = perf_counter()
    embeddings = encoder.encode(unique_smiles)
    embedding_seconds = perf_counter() - before
    np.save(args.output / "unique_embeddings.npy", embeddings, allow_pickle=False)
    mapping = pd.DataFrame(
        {"embedding_row": np.arange(len(unique_smiles)), "smiles": unique_smiles}
    )
    mapping.to_csv(args.output / "embedding_rows.csv", index=False)
    # 序列重排检查不重算标签；仅复核向量映射与批次归并的数值稳定性。
    reversed_values = encoder.encode(list(reversed(unique_smiles)))
    if not np.allclose(embeddings, reversed_values[::-1], rtol=rtol, atol=atol):
        raise ValueError("全体图的逆序分批表示检查失败")
    lookup = {smiles: row for row, smiles in enumerate(unique_smiles)}
    unit_embeddings = embeddings[[lookup[s] for s in units["smiles"]]].astype(np.float64)
    print("表示行序检查通过；开始一次冻结的来源留出比较。", flush=True)
    summary, predictions, metrics, members, models = run_transfer_benchmark(
        units, unit_embeddings, config["benchmark"]
    )
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics_by_assay.csv", index=False)
    members.to_csv(args.output / "fold_members.csv", index=False)
    joblib.dump(models, args.output / "fold_models.joblib")
    write_json(args.output / "evaluation.json", summary)
    original = pd.read_csv(pilot / "metrics_by_assay.csv")
    combined = pd.concat([original, metrics], ignore_index=True)
    combined.to_csv(args.output / "comparison_all_metrics.csv", index=False)
    combined[
        combined["primary_eligible"]
        & combined["budget_fraction"].eq(config["benchmark"]["primary_budget_fraction"])
    ].to_csv(args.output / "comparison_primary.csv", index=False)
    code_root = Path(__file__).resolve().parents[1]
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for source in [
        Path(__file__),
        *sorted((code_root / "src/translock_screening").glob("*.py")),
        code_root / "environments/chemeleon/pyproject.toml",
        code_root / "environments/chemeleon/uv.lock",
    ]:
        shutil.copy2(source, snapshot / source.name)
    manifest = {
        "status": "complete_exploratory_transfer_probe",
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "pilot": str(pilot),
        "weight": str(checkpoint),
        "weight_license": "MIT",
        "data_license": "CC BY-SA 3.0",
        "unique_graphs": len(mapping),
        "units": len(units),
        "encoder_frozen": True,
        "encoder_load_seconds": encoder_load_seconds,
        "embedding_seconds": embedding_seconds,
        "reversed_order_check": True,
        "gpu_used": False,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "python": sys.version,
        "argv": sys.argv,
        "input_size_mtime_unchanged": input_stats
        == {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths},
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "chemprop",
                "torch",
                "lightning",
                "rdkit",
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
            ]
        },
        "hashes_computed": False,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "evaluation": summary}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "utc": datetime.now(UTC).isoformat(),
                "argv": sys.argv,
            },
        )
        raise


if __name__ == "__main__":
    main()
