"""补齐两项交叉步长；旧对角结果只读，始终输出全部新模型。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from translock_screening.training_cross import (  # noqa: E402
    join_reference,
    paired_grid,
    split_optimizer,
)
from translock_screening.training_factor import (  # noqa: E402
    GraphCache,
    build_model,
    configure_torch,
    fit_model,
    make_splits,
    predict,
    score_frame,
    tensors_cpu,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def preflight(payload, config, torch):
    """合成全网络验证单组/双组同速更新，正式来源不参与接口验收。"""
    training = {**config["training"], "epochs": 2}
    seed = training["seeds"][0]
    units = pd.DataFrame(
        {
            "smiles": ["C" * n + end for n in range(2, 10) for end in ["N", "O", "Cl", "F"]],
            "label": [0, 1] * 16,
            "assay_id": ["synthetic"] * 32,
        }
    )
    graphs = GraphCache(units.smiles.tolist(), config["encoder"]["device"])
    train, held = np.arange(24), np.arange(24, 32)
    checks = []
    for rate in [0.0001, 0.001]:
        original = build_model(payload, True, True, seed, training, graphs.device)
        grouped = build_model(payload, True, True, seed, training, graphs.device)
        fit_model(original, graphs, units, train, seed, rate, training)
        fit_model(
            grouped,
            graphs,
            units,
            train,
            seed,
            rate,
            training,
            optimizer=split_optimizer(grouped, rate, rate, 0.0),
        )
        grouped_state = tensors_cpu(grouped)
        for name, value in tensors_cpu(original).items():
            torch.testing.assert_close(value, grouped_state[name], rtol=0, atol=0)
        np.testing.assert_array_equal(
            predict(original, graphs, held, 16), predict(grouped, graphs, held, 16)
        )
        checks.append({"learning_rate": rate, "equal_rate_full_network_exact": True})
    return {"status": "pass", "synthetic_labels_only": True, "checks": checks}


def run(args):
    root = Path(__file__).resolve().parents[1]
    cross = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.storage_root / cross["base_run"]
    config_path = root / cross["base_config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    archived = yaml.safe_load(
        (base / "code_snapshot/training_factor_2026-10-01.yaml").read_text(encoding="utf-8")
    )
    if config != archived:
        raise ValueError("当前基线配置不等于已运行版本")
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in [
        Path(__file__),
        args.config,
        config_path,
        root / "docs/training-cross-protocol.md",
        root / "src/translock_screening/training_cross.py",
        root / "src/translock_screening/training_factor.py",
        root / "src/translock_screening/benchmark.py",
        root / "environments/chemeleon/pyproject.toml",
        root / "environments/chemeleon/uv.lock",
    ]:
        shutil.copy2(path, snapshot / path.name)
    weight = args.storage_root / config["inputs"]["weight"]
    paths = [
        args.config,
        config_path,
        weight,
        base / "assay_graph_units.csv",
        base / "metrics_by_assay.csv",
        base / "fold_members.csv",
        *[
            base / f"initial_states/{seed}_pretrained_trained.pt"
            for seed in config["training"]["seeds"]
        ],
    ]
    stats = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}
    if importlib.metadata.version("chemprop") != config["encoder"]["chemprop"]:
        raise ValueError("Chemprop版本不一致")
    torch = configure_torch(config["encoder"])
    payload = torch.load(weight, weights_only=True, map_location="cpu")
    if payload["hyper_parameters"]["dropout"] != 0.0:
        raise ValueError("基线含dropout")
    print("开始合成全网络参数组等价检查。", flush=True)
    write_json(args.output / "preflight.json", preflight(payload, config, torch))
    if args.mode == "preflight":
        return {"status": "preflight_complete", "input_stats": stats}
    print("合成接口检查通过，开始36组固定交叉对照。", flush=True)
    units = pd.read_csv(base / "assay_graph_units.csv")
    splits, members = make_splits(units)
    pd.testing.assert_frame_equal(
        members, pd.read_csv(base / "fold_members.csv"), check_dtype=False
    )
    members.to_csv(args.output / "fold_members.csv", index=False)
    units.to_csv(args.output / "assay_graph_units.csv", index=False)
    graphs = GraphCache(units.smiles.tolist(), config["encoder"]["device"])
    training = config["training"]
    model_dir = args.output / "models"
    model_dir.mkdir()
    records, predictions, histories = [], [], []
    for seed in training["seeds"]:
        initial_path = base / f"initial_states/{seed}_pretrained_trained.pt"
        expected_initial = torch.load(initial_path, weights_only=True, map_location="cpu")
        for document, (train, test) in splits.items():
            for condition in cross["configurations"]:
                count = len(records) + 1
                run_id = f"cross_{count:03d}"
                identity = {
                    "run_id": run_id,
                    "seed": seed,
                    "document_id": document,
                    "condition": condition["name"],
                    "method": condition["name"],
                    "learning_rate": condition["head_lr"],
                    "head_lr": condition["head_lr"],
                    "encoder_lr": condition["encoder_lr"],
                }
                model = build_model(payload, True, True, seed, training, graphs.device)
                for name, value in tensors_cpu(model).items():
                    if not torch.equal(value, expected_initial[name]):
                        raise ValueError("初值不等于原对角组")
                optimizer = split_optimizer(
                    model, condition["head_lr"], condition["encoder_lr"], training["weight_decay"]
                )
                history, record = fit_model(
                    model,
                    graphs,
                    units,
                    train,
                    seed,
                    condition["head_lr"],
                    training,
                    optimizer=optimizer,
                )
                scores = predict(model, graphs, test, training["batch_size"])
                path = model_dir / f"{run_id}.pt"
                torch.save(tensors_cpu(model), path)
                restored = build_model(payload, True, True, seed, training, graphs.device)
                restored.load_state_dict(
                    torch.load(path, weights_only=True, map_location="cpu"), strict=True
                )
                replay = predict(restored, graphs, test, training["batch_size"])
                np.testing.assert_array_equal(replay, scores)
                record = {
                    **identity,
                    **record,
                    "initial_state": str(initial_path),
                    "train_units": len(train),
                    "test_units": len(test),
                    "readback_max_abs_error": float(np.abs(replay - scores).max()),
                }
                records.append(record)
                histories.extend({**identity, **row} for row in history)
                frame = units.iloc[test].copy()
                for name, value in identity.items():
                    frame[name] = value
                frame["score"] = scores
                frame.to_csv(model_dir / f"{run_id}_predictions.csv", index=False)
                predictions.append(frame)
                write_json(model_dir / f"{run_id}_record.json", {**record, "history": history})
                write_json(
                    args.output / "progress.json",
                    {"completed": count, "total": cross["expected_models"], "last": identity},
                )
                print(
                    f"{count}/36 {condition['name']} seed={seed} {document} "
                    f"{record['seconds']:.1f}s",
                    flush=True,
                )
                del model, restored, optimizer
    frame = pd.concat(predictions, ignore_index=True)
    if len(records) != cross["expected_models"] or len(frame) != cross["expected_predictions"]:
        raise ValueError("交叉模型或预测数量不符")
    frame.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(args.output / "training_history.csv", index=False)
    write_json(args.output / "model_records.json", records)
    new_metrics = score_frame(frame, config["benchmark"])
    conditions = {row["name"]: row for row in cross["configurations"]}
    new_metrics["condition"] = new_metrics.method
    new_metrics["head_lr"] = new_metrics.method.map(lambda name: conditions[name]["head_lr"])
    new_metrics["encoder_lr"] = new_metrics.method.map(lambda name: conditions[name]["encoder_lr"])
    new_metrics.to_csv(args.output / "new_metrics_by_assay.csv", index=False)
    combined = join_reference(new_metrics, pd.read_csv(base / "metrics_by_assay.csv"))
    combined.to_csv(args.output / "grid_metrics.csv", index=False)
    paired = paired_grid(combined)
    if len(paired) != 15 or paired.assay_id.nunique() != 5:
        raise ValueError("主要来源或配对数量不符")
    paired.to_csv(args.output / "paired_grid_effects.csv", index=False)
    macro = (
        combined[combined.primary_eligible]
        .groupby(["condition", "seed"])[
            [
                "expected_hits_with_ties",
                "enrichment",
                "precision",
                "recall",
                "roc_auc",
                "average_precision",
            ]
        ]
        .mean()
        .reset_index()
    )
    macro.to_csv(args.output / "macro_by_seed.csv", index=False)
    if stats != {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}:
        raise ValueError("运行期间输入变更")
    return {
        "status": "complete_bounded_exploratory_cross_check",
        "models": len(records),
        "predictions": len(frame),
        "old_models_refit": 0,
        "primary_assays": 5,
        "input_stats": stats,
        "input_size_mtime_unchanged": True,
        "all_initial_states_equal_to_original": True,
        "all_saved_predictions_exact": True,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "macro_by_seed": macro.to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["run", "preflight"], default="run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started, before = datetime.now(UTC).isoformat(), perf_counter()
    try:
        summary = run(args)
        write_json(args.output / "summary.json", summary)
        write_json(
            args.output / "manifest.json",
            {
                "status": "complete",
                "started_utc": started,
                "completed_utc": datetime.now(UTC).isoformat(),
                "seconds": perf_counter() - before,
                "argv": sys.argv,
                "python": sys.version,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ["torch", "chemprop", "numpy", "pandas", "rdkit", "scikit-learn"]
                },
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "hashes_computed": False,
            },
        )
        print(f"完成，总耗时{perf_counter() - before:.1f}秒。", flush=True)
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {
                "status": "failed",
                "error": str(error),
                "error_type": type(error).__name__,
                "utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
