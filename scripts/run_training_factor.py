"""同架构四组对照；合成小样验收和完整运行分别进入新产物目录。"""

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

from translock_screening.training_factor import (  # noqa: E402
    ARMS,
    GraphCache,
    build_model,
    configure_torch,
    contrasts,
    fit_model,
    make_splits,
    predict,
    score_frame,
    tensors_cpu,
    vector_cache,
)


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def snapshot(args):
    root = Path(__file__).resolve().parents[1]
    target = args.output / "code_snapshot"
    target.mkdir()
    paths = [
        Path(__file__),
        args.config,
        root / "docs/training-factor-protocol.md",
        root / "src/translock_screening/training_factor.py",
        root / "src/translock_screening/benchmark.py",
        root / "environments/chemeleon/pyproject.toml",
        root / "environments/chemeleon/uv.lock",
    ]
    for path in paths:
        shutil.copy2(path, target / path.name)


def preflight(args, config, payload, torch):
    """用合成标签检验真实完整网络，不观察科研测试性能。"""
    training = {**config["training"], "epochs": 2}
    seed = training["seeds"][0]
    smiles = ["C" * n + end for n in range(2, 10) for end in ["O", "N", "F", "Cl"]]
    units = pd.DataFrame(
        {
            "smiles": smiles,
            "label": [i % 2 for i in range(len(smiles))],
            "assay_id": ["synthetic"] * len(smiles),
        }
    )
    graphs = GraphCache(smiles, config["encoder"]["device"])
    train, test = np.arange(24), np.arange(24, 32)
    initial_states, results, saved_scores = {}, [], {}
    for arm, (pretrained, train_encoder) in ARMS.items():
        model = build_model(payload, pretrained, train_encoder, seed, training, graphs.device)
        initial_states[arm] = tensors_cpu(model)
        vectors = vector_cache(model, graphs, training["batch_size"]) if not train_encoder else None
        if vectors is not None:
            direct = predict(model, graphs, np.arange(32), 16)
            cached = predict(model, graphs, np.arange(32), 16, vectors)
            np.testing.assert_allclose(direct, cached, rtol=1e-5, atol=1e-5)
        history, record = fit_model(model, graphs, units, train, seed, 0.0001, training, vectors)
        scores = predict(model, graphs, test, 16, vectors)
        saved_scores[arm] = scores
        target = args.output / f"{arm}.pt"
        torch.save(tensors_cpu(model), target)
        loaded = build_model(payload, pretrained, train_encoder, seed, training, graphs.device)
        loaded.load_state_dict(torch.load(target, weights_only=True, map_location=graphs.device))
        np.testing.assert_allclose(scores, predict(loaded, graphs, test, 16), rtol=1e-5, atol=1e-5)
        results.append({"arm": arm, **record, "history": history, "save_readback_pass": True})
        print(f"合成小样 {arm} 通过，用时{record['seconds']:.2f}秒。", flush=True)
        del model, loaded, vectors
    for state in initial_states.values():
        for key in ["head.weight", "head.bias"]:
            assert torch.equal(state[key], initial_states["random_frozen"][key])
    for prefix in ["random", "pretrained"]:
        for key, value in initial_states[f"{prefix}_frozen"].items():
            assert torch.equal(value, initial_states[f"{prefix}_trained"][key])
    assert any(
        not torch.equal(value, initial_states["pretrained_frozen"][key])
        for key, value in initial_states["random_frozen"].items()
        if key.startswith("encoder.")
    )
    # 翻转所有留出标签后重建、重训，训练侧未变时输出必须保持。
    alternate = units.copy()
    alternate.loc[test, "label"] = 1 - alternate.loc[test, "label"]
    repeated = build_model(payload, False, True, seed, training, graphs.device)
    fit_model(repeated, graphs, alternate, train, seed, 0.0001, training)
    replay = predict(repeated, graphs, test, 16)
    np.testing.assert_array_equal(replay, saved_scores["random_trained"])
    return {
        "status": "synthetic_full_architecture_preflight_pass",
        "initial_head_paired": True,
        "initial_encoder_paired": True,
        "heldout_label_flip_invariance": True,
        "deterministic_replay_exact": True,
        "training_records": results,
        "research_labels_used": False,
    }


def experiment(args, config, payload, torch):
    pilot = args.storage_root / config["inputs"]["pilot"]
    units = pd.read_csv(pilot / "assay_graph_units.csv")
    shutil.copy2(pilot / "assay_graph_units.csv", args.output / "assay_graph_units.csv")
    splits, members = make_splits(units)
    members.to_csv(args.output / "fold_members.csv", index=False)
    graphs = GraphCache(units["smiles"].tolist(), config["encoder"]["device"])
    training = config["training"]
    initial_dir, model_dir = args.output / "initial_states", args.output / "models"
    initial_dir.mkdir()
    model_dir.mkdir()
    records, predictions, history_rows = [], [], []
    count = 0
    total = len(splits) * len(ARMS) * len(training["seeds"]) * len(training["learning_rates"])
    for seed in training["seeds"]:
        states, vectors_by_initialization = {}, {}
        for arm, (pretrained, train_encoder) in ARMS.items():
            model = build_model(payload, pretrained, train_encoder, seed, training, graphs.device)
            states[arm] = tensors_cpu(model)
            torch.save(states[arm], initial_dir / f"{seed}_{arm}.pt")
            if not train_encoder:
                vectors_by_initialization[pretrained] = vector_cache(
                    model, graphs, training["batch_size"]
                )
            del model
        for state in states.values():
            for key in ["head.weight", "head.bias"]:
                assert torch.equal(state[key], states["random_frozen"][key])
        for prefix in ["random", "pretrained"]:
            assert all(
                torch.equal(value, states[f"{prefix}_trained"][key])
                for key, value in states[f"{prefix}_frozen"].items()
            )
        for rate in training["learning_rates"]:
            for document, (train, test) in splits.items():
                for arm, (pretrained, train_encoder) in ARMS.items():
                    count += 1
                    run_id = f"run_{count:03d}"
                    identity = {
                        "run_id": run_id,
                        "method": arm,
                        "seed": seed,
                        "learning_rate": rate,
                        "document_id": document,
                    }
                    model = build_model(
                        payload, pretrained, train_encoder, seed, training, graphs.device
                    )
                    for key, value in tensors_cpu(model).items():
                        if not torch.equal(value, states[arm][key]):
                            raise ValueError("训练初值不一致")
                    vectors = None if train_encoder else vectors_by_initialization[pretrained]
                    history, record = fit_model(
                        model, graphs, units, train, seed, rate, training, vectors
                    )
                    scores = predict(model, graphs, test, training["batch_size"], vectors)
                    state = tensors_cpu(model)
                    saved = (
                        state
                        if train_encoder
                        else {key: value for key, value in state.items() if key.startswith("head.")}
                    )
                    model_path = model_dir / f"{run_id}.pt"
                    torch.save(saved, model_path)
                    # 每个模型保存后重新加载回读，冻结组由已存初值+最终头重建。
                    restored = build_model(
                        payload, pretrained, train_encoder, seed, training, graphs.device
                    )
                    restored_state = {
                        **states[arm],
                        **torch.load(model_path, weights_only=True, map_location="cpu"),
                    }
                    restored.load_state_dict(restored_state, strict=True)
                    replay = predict(restored, graphs, test, training["batch_size"])
                    np.testing.assert_allclose(scores, replay, rtol=1e-5, atol=1e-5)
                    record = {
                        **identity,
                        **record,
                        "train_units": len(train),
                        "test_units": len(test),
                        "initial_state": f"initial_states/{seed}_{arm}.pt",
                        "final_state": f"models/{run_id}.pt",
                        "final_state_kind": "full" if train_encoder else "head_only",
                        "readback_max_abs_error": float(np.abs(scores - replay).max()),
                    }
                    records.append(record)
                    history_rows.extend({**identity, **row} for row in history)
                    frame = units.iloc[test].copy()
                    for key, value in identity.items():
                        frame[key] = value
                    frame["score"] = scores
                    predictions.append(frame)
                    frame.to_csv(model_dir / f"{run_id}_predictions.csv", index=False)
                    write_json(model_dir / f"{run_id}_record.json", {**record, "history": history})
                    write_json(
                        args.output / "progress.json",
                        {
                            "completed": count,
                            "total": total,
                            "last": identity,
                            "utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    print(
                        f"{count}/{total} {arm} seed={seed} lr={rate} {document} "
                        f"{record['seconds']:.1f}s",
                        flush=True,
                    )
                    del model, restored, state, saved, restored_state
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(history_rows).to_csv(args.output / "training_history.csv", index=False)
    write_json(args.output / "model_records.json", records)
    metrics = score_frame(prediction_frame, config["benchmark"])
    metrics.to_csv(args.output / "metrics_by_assay.csv", index=False)
    paired = contrasts(metrics)
    paired.to_csv(args.output / "factor_contrasts.csv", index=False)
    primary = metrics[metrics.primary_eligible]
    macro = (
        primary.groupby(["method", "learning_rate", "seed"])[
            [
                "expected_hits_with_ties",
                "precision",
                "recall",
                "enrichment",
                "roc_auc",
                "average_precision",
            ]
        ]
        .mean()
        .reset_index()
    )
    macro.to_csv(args.output / "macro_by_seed.csv", index=False)
    return {
        "status": "complete_exploratory_fixed_budget_factor_experiment",
        "units": len(units),
        "graphs": units.smiles.nunique(),
        "documents": len(splits),
        "model_count": count,
        "prediction_count": len(prediction_frame),
        "primary_assays": primary.assay_id.nunique(),
        "no_test_selection": True,
        "score": "raw_logit",
        "all_model_readbacks_pass": True,
        "macro_by_seed": macro.to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["preflight", "run"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start, started = perf_counter(), datetime.now(UTC).isoformat()
    try:
        snapshot(args)
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if importlib.metadata.version("chemprop") != config["encoder"]["chemprop"]:
            raise ValueError("Chemprop版本不符")
        paths = [args.config, args.storage_root / config["inputs"]["weight"]]
        if args.mode == "run":
            paths.append(args.storage_root / config["inputs"]["pilot"] / "assay_graph_units.csv")
        stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
        torch = configure_torch(config["encoder"])
        payload = torch.load(paths[1], weights_only=True, map_location="cpu")
        if payload["hyper_parameters"]["d_h"] != config["encoder"]["dimensions"]:
            raise ValueError("checkpoint维度不符")
        if payload["hyper_parameters"]["dropout"] != 0.0:
            raise ValueError("checkpoint含随机失活，不符合四组共同无dropout协议")
        if (
            sum(v.numel() for v in payload["state_dict"].values())
            != config["encoder"]["parameters"]
        ):
            raise ValueError("checkpoint参数量不符")
        print(f"{args.mode} 启动；设备 {torch.cuda.get_device_name(0)}", flush=True)
        result = (
            preflight(args, config, payload, torch)
            if args.mode == "preflight"
            else experiment(args, config, payload, torch)
        )
        write_json(args.output / "summary.json", result)
        after = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
        if stats != after:
            raise ValueError("运行期间输入变更")
        write_json(
            args.output / "manifest.json",
            {
                "status": "complete",
                "mode": args.mode,
                "started_utc": started,
                "completed_utc": datetime.now(UTC).isoformat(),
                "seconds": perf_counter() - start,
                "input_stats": stats,
                "input_size_mtime_unchanged": True,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ["torch", "chemprop", "numpy", "pandas", "rdkit", "scikit-learn"]
                },
                "device": torch.cuda.get_device_name(0),
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "python": sys.version,
                "argv": sys.argv,
                "hashes_computed": False,
                "deterministic_algorithms": True,
                "tf32": False,
                "mixed_precision": False,
                "data_license": "CC BY-SA 3.0",
                "checkpoint_license": "MIT",
            },
        )
        print(f"完成 {args.output}；总耗时{perf_counter() - start:.1f}秒", flush=True)
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
