"""固定全量拟合、真实小样验收和公开目录筛选，三阶段均拒绝覆盖。"""

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
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdDepictor  # noqa: E402

from translock_screening.screening_application import (  # noqa: E402
    add_rankings,
    connection,
    nearest_support,
    nominate,
    normalize,
    prepare_catalogue,
    validate_source,
    validate_training_scope,
)
from translock_screening.training_factor import (  # noqa: E402
    GraphCache,
    build_model,
    configure_torch,
    fit_model,
    predict,
    tensors_cpu,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def file_stats(paths):
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def snapshot(root, output, config_path):
    destination = output / "code_snapshot"
    destination.mkdir()
    paths = [
        Path(__file__),
        config_path,
        root / "docs/screening-application-protocol.md",
        *[
            root / f"src/translock_screening/{name}.py"
            for name in ["screening_application", "training_factor", "benchmark"]
        ],
        root / "environments/chemeleon/pyproject.toml",
        root / "environments/chemeleon/uv.lock",
    ]
    for path in paths:
        shutil.copy2(path, destination / path.name)


def initialize(args, config):
    base = args.storage_root / config["inputs"]["factor"]
    base_config = yaml.safe_load(
        (base / "code_snapshot/training_factor_2026-10-01.yaml").read_text(encoding="utf-8")
    )
    if base_config["training"]["primary_learning_rate"] != 0.0001:
        raise ValueError("原主步长不符")
    if base_config["training"]["epochs"] != 30:
        raise ValueError("原轮数不符")
    seeds = [config["ranking"]["primary_seed"], *config["ranking"]["sensitivity_seeds"]]
    if seeds != base_config["training"]["seeds"]:
        raise ValueError("种子不符")
    if importlib.metadata.version("chemprop") != base_config["encoder"]["chemprop"]:
        raise ValueError("Chemprop版本不符")
    torch = configure_torch(base_config["encoder"])
    weight = args.storage_root / config["inputs"]["weight"]
    payload = torch.load(weight, weights_only=True, map_location="cpu")
    if payload["hyper_parameters"]["dropout"] != 0:
        raise ValueError("编码器dropout与原协议不同")
    units = pd.read_csv(base / "assay_graph_units.csv", dtype={"activity_ids": "string"})
    validate_training_scope(units, config["scope"])
    return base, base_config, seeds, torch, payload, units


def fit(args, config):
    base, original, seeds, torch, payload, units = initialize(args, config)
    paths = [
        args.config,
        base / "assay_graph_units.csv",
        args.storage_root / config["inputs"]["weight"],
    ]
    inputs_before = file_stats(paths)
    units.to_csv(args.output / "training_members.csv", index=False)
    write_json(args.output / "original_training_config.json", original)
    graphs = GraphCache(units.smiles.tolist(), original["encoder"]["device"])
    members = np.arange(len(units))
    training = original["training"]
    records, histories, predictions = [], [], []
    for seed in seeds:
        model = build_model(payload, True, True, seed, training, graphs.device)
        expected = torch.load(
            base / f"initial_states/{seed}_pretrained_trained.pt",
            weights_only=True,
            map_location="cpu",
        )
        for name, value in tensors_cpu(model).items():
            if not torch.equal(value, expected[name]):
                raise ValueError("全量拟合初值与原对照不一致")
        history, record = fit_model(
            model, graphs, units, members, seed, training["primary_learning_rate"], training
        )
        scores = predict(model, graphs, members, training["batch_size"])
        torch.save(tensors_cpu(model), args.output / f"model_{seed}.pt")
        restored = build_model(payload, True, True, seed, training, graphs.device)
        restored.load_state_dict(
            torch.load(args.output / f"model_{seed}.pt", weights_only=True, map_location="cpu")
        )
        replay = predict(restored, graphs, members, training["batch_size"])
        np.testing.assert_array_equal(scores, replay)
        records.append({"seed": seed, **record, "exact_reload_predictions": True})
        histories.extend({"seed": seed, **row} for row in history)
        frame = units[["unit_id", "label"]].copy()
        frame["seed"], frame["training_logit"] = seed, scores
        predictions.append(frame)
        print(
            f"全量拟合完成 seed={seed}，{record['seconds']:.1f}秒；保存回读精确一致。", flush=True
        )
        del model, restored
    pd.DataFrame(histories).to_csv(args.output / "training_history.csv", index=False)
    pd.concat(predictions).to_csv(args.output / "training_fit_predictions.csv", index=False)
    write_json(args.output / "model_records.json", records)
    assert inputs_before == file_stats(paths)
    return {
        "models": len(seeds),
        "training_units": len(units),
        "training_graphs": units.smiles.nunique(),
        "seeds": seeds,
        "primary_seed": seeds[0],
        "learning_rate": 0.0001,
        "epochs": 30,
        "purpose": "full_data_refit_for_application_not_new_validation",
        "all_initial_states_match_original": True,
        "all_reloaded_predictions_exact": True,
        "input_stats": inputs_before,
        "input_size_mtime_unchanged": True,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def known_registry(args, config, units, catalogue):
    pilot = args.storage_root / config["inputs"]["pilot"]
    policy = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))["expanded_policy"]
    audit_path = args.storage_root / config["inputs"]["audit"]
    bridge = json.loads((args.legacy_registry.parent / "manifest.json").read_text(encoding="utf-8"))
    if (
        bridge["status"] != "complete"
        or Path(bridge["source"]) != audit_path
        or bridge["source_stats"] != [audit_path.stat().st_size, audit_path.stat().st_mtime_ns]
    ):
        raise ValueError("旧身份清单投影与原Parquet来源不符")
    audit = pd.read_csv(args.legacy_registry)
    if len(audit) != bridge["rows"]:
        raise ValueError("旧身份清单行数不符")
    ids = set(audit.molecule_chembl_id.dropna()) | set(policy["excluded_molecule_ids"])
    source_smiles = [
        *audit.standardized_smiles.dropna(),
        *policy["excluded_alias_smiles"],
        *units.smiles,
    ]
    current_path = catalogue / "known_target_molecules.csv"
    activity_path = catalogue / "known_target_activities.csv"
    registry_note = {"current_missing_structure_ids": [], "activity_ids_without_structure_row": []}
    if args.stage == "screen" and (not current_path.exists() or not activity_path.exists()):
        raise ValueError("正式筛选要求当前靶点身份清单完整")
    if current_path.exists():
        current = pd.read_csv(current_path)
        ids.update(current.molecule_chembl_id.dropna())
        source_smiles.extend(current.canonical_smiles.dropna())
        registry_note["current_missing_structure_ids"] = current.loc[
            current.canonical_smiles.isna(), "molecule_chembl_id"
        ].tolist()
    if activity_path.exists():
        activity_ids = set(pd.read_csv(activity_path).molecule_chembl_id.dropna())
        ids.update(activity_ids)
        registry_note["activity_ids_without_structure_row"] = sorted(
            activity_ids - set(current.molecule_chembl_id)
            if current_path.exists()
            else activity_ids
        )
    write_json(args.output / "known_registry_scope.json", registry_note)
    connections, failed = set(), []
    for value in dict.fromkeys(source_smiles):
        try:
            connections.add(connection(normalize(value)))
        except (ValueError, RuntimeError) as error:
            failed.append({"smiles": value, "error": str(error)})
    if failed:
        write_json(args.output / "known_registry_failures.json", failed)
        raise ValueError("已知/隔离结构存在未处理解析失败")
    pd.DataFrame({"known_id": sorted(ids)}).to_csv(
        args.output / "excluded_known_ids.csv", index=False
    )
    pd.DataFrame({"connection_id": sorted(connections)}).to_csv(
        args.output / "excluded_connections.csv", index=False
    )
    return ids, connections


def screen(args, config):
    catalogue = args.catalogue or args.storage_root / config["inputs"]["catalogue"]
    source = json.loads((catalogue / "manifest.json").read_text(encoding="utf-8"))
    fitted = json.loads((args.model_run / "manifest.json").read_text(encoding="utf-8"))
    if source["status"] != "complete" or fitted["status"] != "complete" or fitted["stage"] != "fit":
        raise ValueError("数据获取或模型拟合未完成")
    raw = pd.read_csv(catalogue / "structures.csv")
    validate_source(source, len(raw))
    if args.stage == "screen" and len(raw) != config["expected_input_records"]:
        raise ValueError("正式筛选目录数量不符")
    if args.stage == "smoke":
        raw = raw.iloc[:256].copy()
    if raw.molecule_chembl_id.tolist() != sorted(raw.molecule_chembl_id.tolist()):
        raise ValueError("来源ID不是固定升序")
    _, original, seeds, torch, payload, units = initialize(args, config)
    stored_original = json.loads(
        (args.model_run / "original_training_config.json").read_text(encoding="utf-8")
    )
    stored_app = yaml.safe_load(
        (args.model_run / "code_snapshot" / args.config.name).read_text(encoding="utf-8")
    )
    fit_summary = json.loads((args.model_run / "summary.json").read_text(encoding="utf-8"))
    if original != stored_original or config != stored_app:
        raise ValueError("推理配置不等于全量拟合记录")
    if fit_summary["input_stats"] != file_stats(
        [Path(path) for path in fit_summary["input_stats"]]
    ):
        raise ValueError("全量拟合输入在推理前发生变更")
    pd.testing.assert_frame_equal(
        units,
        pd.read_csv(args.model_run / "training_members.csv", dtype={"activity_ids": "string"}),
    )
    paths = [
        args.config,
        catalogue / "manifest.json",
        catalogue / "structures.csv",
        args.model_run / "manifest.json",
        args.legacy_registry,
        args.legacy_registry.parent / "manifest.json",
        args.storage_root / config["inputs"]["audit"],
        args.storage_root / config["inputs"]["pilot"] / "manifest.json",
        *[args.model_run / f"model_{seed}.pt" for seed in seeds],
    ]
    paths += [
        p
        for p in [
            catalogue / "known_target_molecules.csv",
            catalogue / "known_target_activities.csv",
        ]
        if p.exists()
    ]
    before_inputs = file_stats(paths)
    ids, connections = known_registry(args, config, units, catalogue)
    before = perf_counter()

    def progress(done, total):
        print(f"结构准入 {done}/{total}", flush=True)
        write_json(
            args.output / "progress.json", {"stage": "structures", "done": done, "total": total}
        )

    prepared = prepare_catalogue(raw, ids, connections, config["scope"], progress=progress)
    prepared.to_csv(args.output / "prepared_records.csv", index=False)
    ready = prepared[prepared.status.eq("ready_for_scoring")].copy().reset_index(drop=True)
    preparation_seconds = perf_counter() - before
    if ready.empty:
        raise ValueError("没有可评分图；已保留全部准入状态，不能提名")
    models = {}
    for seed in seeds:
        model = build_model(
            payload, True, True, seed, original["training"], original["encoder"]["device"]
        )
        model.load_state_dict(
            torch.load(args.model_run / f"model_{seed}.pt", weights_only=True, map_location="cpu")
        )
        models[seed] = model
    before = perf_counter()
    chunks, replay_checks = [], []
    for start in range(0, len(ready), config["runtime"]["processing_chunk"]):
        part = ready.iloc[start : start + config["runtime"]["processing_chunk"]].copy()
        graphs = GraphCache(part.smiles.tolist(), original["encoder"]["device"])
        indices = np.arange(len(part))
        for seed, model in models.items():
            values = predict(model, graphs, indices, config["runtime"]["prediction_batch"])
            part[f"score_{seed}"] = values
            if start == 0:
                replay_indices = indices[: min(32, len(indices))]
                replay = predict(
                    model, graphs, replay_indices, config["runtime"]["prediction_batch"]
                )
                np.testing.assert_array_equal(replay, values[: len(replay)])
                replay_checks.append({"seed": seed, "rows": len(replay), "exact_repeat": True})
        chunks.append(part)
        completed = start + len(part)
        write_json(
            args.output / "progress.json",
            {"stage": "neural_scores", "done": completed, "total": len(ready)},
        )
        print(f"三种子评分 {completed}/{len(ready)}", flush=True)
    scored = pd.concat(chunks, ignore_index=True)
    inference_seconds = perf_counter() - before
    before = perf_counter()
    support = nearest_support(scored.smiles, units)
    scored = pd.concat([scored, support], axis=1)
    support_seconds = perf_counter() - before
    scored = add_rankings(scored, seeds)
    scored["status"] = "scored_candidate_hypothesis_only"
    scored.to_csv(args.output / "scored_records.csv", index=False)
    extra = [name for name in scored.columns if name not in prepared.columns]
    complete = prepared.merge(
        scored[["candidate_id", *extra]], on="candidate_id", how="left", validate="one_to_one"
    )
    complete.loc[complete.status.eq("ready_for_scoring"), "status"] = (
        "scored_candidate_hypothesis_only"
    )
    complete.to_csv(args.output / "all_records.csv", index=False)
    nominated, pool = nominate(scored, config["ranking"])
    if args.stage == "smoke":
        nominated = nominated.iloc[:0]
    nominated.to_csv(args.output / "candidate_hypotheses.csv", index=False)
    support_rows = []
    for row in nominated.to_dict("records"):
        for category in ["any", "strong", "weak"]:
            matched = units[units.unit_id.eq(row[f"nearest_{category}_unit_id"])].iloc[0]
            support_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "neighbor_role": category,
                    "similarity": row[f"nearest_{category}_similarity"],
                    **matched.to_dict(),
                }
            )
    pd.DataFrame(
        support_rows, columns=["candidate_id", "neighbor_role", "similarity", *units.columns]
    ).to_csv(args.output / "candidate_support_measurements.csv", index=False)
    writer = Chem.SDWriter(str(args.output / "candidate_hypotheses_2d.sdf"))
    for row in nominated.to_dict("records"):
        mol = Chem.MolFromSmiles(row["smiles"])
        rdDepictor.Compute2DCoords(mol)
        mol.SetProp("_Name", row["candidate_id"])
        for key, value in row.items():
            if pd.notna(value):
                mol.SetProp(key, str(value))
        mol.SetProp("structure_role", "2D_identity_not_binding_pose")
        writer.write(mol)
    writer.close()
    write_json(args.output / "prediction_repeat_checks.json", replay_checks)
    assert before_inputs == file_stats(paths)
    return {
        "input_records": len(raw),
        "scored_graphs": len(scored),
        "candidate_hypotheses": len(nominated),
        "candidate_pool": pool,
        "status_counts": complete.status.value_counts().to_dict(),
        "preparation_seconds": preparation_seconds,
        "inference_seconds_three_models": inference_seconds,
        "nearest_support_seconds": support_seconds,
        "scored_graphs_per_second_three_models": len(scored) / inference_seconds,
        "model_run": str(args.model_run),
        "catalogue": str(catalogue),
        "primary_seed": seeds[0],
        "known_ids": len(ids),
        "known_connections": len(connections),
        "input_stats": before_inputs,
        "input_size_mtime_unchanged": True,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "activity_claim": "unknown; no prospective activity or direction validation",
        "data_license": "CC BY-SA 3.0",
        "checkpoint_license": "MIT",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fit", "smoke", "screen"], required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--legacy-registry", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.stage != "fit" and args.model_run is None:
        parser.error("推理阶段需要--model-run")
    if args.stage != "fit" and args.legacy_registry is None:
        parser.error("推理阶段需要--legacy-registry，先用已有数据环境导出身份列")
    if shutil.disk_usage(args.storage_root).free < config["runtime"]["minimum_free_gib"] * 1024**3:
        raise RuntimeError("E盘空间低于预定界限，不删除文件")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    snapshot(root, args.output, args.config)
    started, clock = datetime.now(UTC).isoformat(), perf_counter()
    try:
        summary = fit(args, config) if args.stage == "fit" else screen(args, config)
        write_json(args.output / "summary.json", summary)
        manifest = {
            "status": "complete",
            "stage": args.stage,
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "total_seconds": perf_counter() - clock,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["torch", "chemprop", "numpy", "pandas", "rdkit"]
            },
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "python": sys.version,
            "argv": sys.argv,
            "hashes_computed": False,
        }
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({**manifest, **summary}, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
