"""竞赛提交入口：复用已完成研究的模型和规则，仅适配相对路径与输出格式。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from rdkit import Chem

from translock_screening.screening_application import (
    add_rankings,
    nearest_support,
    nominate,
    prepare_catalogue,
    validate_source,
    validate_training_scope,
)
from translock_screening.training_factor import (
    GraphCache,
    build_model,
    configure_torch,
    fit_model,
    predict,
    tensors_cpu,
)


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def save_json(target, value):
    Path(target).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def settings():
    return json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))


def runtime(device):
    torch = configure_torch({"device": device, "threads": 4})
    return torch


def catalogue_input(input_csv):
    """训练编排和筛选使用同一正式输入契约，错误输入在计算开始前拒绝。"""
    source = path(input_csv)
    raw = pd.read_csv(source)
    required = {"molecule_chembl_id", "canonical_smiles"}
    if not required.issubset(raw.columns):
        raise ValueError("输入缺少molecule_chembl_id或canonical_smiles列")
    ids = raw.molecule_chembl_id
    if raw.empty or ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("输入为空、候选ID缺失或重复")
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    validate_source(manifest, len(raw))
    if len(raw) != 100000 or ids.tolist() != sorted(ids):
        raise ValueError("正式目录应为版本固定的100000条不重复ID升序记录")
    return raw


def model_source(model_dir):
    """新训练目录必须提供固定配置、成员和完成记录，不能静默回退到归档模型。"""
    directory = path(model_dir).resolve()
    cfg = settings()
    for seed in cfg["training"]["seeds"]:
        if not (directory / f"model_{seed}.pt").is_file():
            raise FileNotFoundError(f"缺少固定种子权重：model_{seed}.pt")
    archived = directory == (ROOT / "models").resolve()
    if not archived:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        training = json.loads((directory / "training_config.json").read_text(encoding="utf-8"))
        if training != cfg["training"] or any(manifest.get(k) != v for k, v in {
            "status": "complete", "stage": "train_full", "units": 277,
            "graphs": 248, "epochs": 30, "package_version": cfg["package_version"],
        }.items()):
            raise ValueError("权重目录不是本包固定配置的已完成正式训练")
        pd.testing.assert_frame_equal(
            pd.read_csv(directory / "training_members.csv"),
            pd.read_csv(ROOT / "data/training/training_units.csv"), check_dtype=False,
            check_exact=True,
        )
    return directory, archived


def load_model(seed, device, model_dir="models"):
    torch = runtime(device)
    cfg = settings()
    payload = torch.load(
        ROOT / "models/pretrained/chemeleon_mp.pt", weights_only=True, map_location="cpu"
    )
    if payload["hyper_parameters"] != cfg["encoder_hyper_parameters"]:
        raise ValueError("预训练架构与封存配置不一致")
    model = build_model(payload, True, True, seed, cfg["training"], device)
    model.load_state_dict(
        torch.load(path(model_dir) / f"model_{seed}.pt",
                   weights_only=True, map_location="cpu"), strict=True
    )
    return model


def metadata(stage, started, clock, args, **extra):
    torch = runtime(args.device)
    return {
        "status": "complete", "stage": stage,
        "started_utc": started, "completed_utc": datetime.now(UTC).isoformat(),
        "seconds": perf_counter() - clock,
        "package_version": settings()["package_version"],
        "python": platform.python_version(), "platform": platform.platform(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "versions": {n: importlib.metadata.version(n) for n in
                     ["torch", "chemprop", "numpy", "pandas", "rdkit", "scikit-learn"]},
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "hashes_computed": False, **extra,
    }


def results_table(frame, run_version, scope, structure_prefix="", *, model_version=None):
    """至少提供附件5规定字段；原始logit及敏感性不写成效力或置信区间。"""
    structure_index = ROOT / "results/structure_evidence/exported_complexes.json"
    # 公开版可不附带大型条件性姿势；缺少该资源时结构链接留空。
    records = json.loads(structure_index.read_text(encoding="utf-8")) if structure_index.is_file() else []
    by_id = {}
    for record in records:
        by_id.setdefault(record["candidate_id"], []).append(record)
    rows = []
    for row in frame.to_dict("records"):
        matches = by_id.get(row["candidate_id"], [])
        query = Chem.MolFromSmiles(row["smiles"]) if matches else None
        files = []
        for record in matches:
            molecule = Chem.MolFromSmiles(record["smiles"])
            # 允许既有条件性构型细化未指定中心，但必须保持输入已指定的构型与完整连接图。
            if (query is not None and molecule.GetNumAtoms() == query.GetNumAtoms()
                    and molecule.GetNumBonds() == query.GetNumBonds()
                    and molecule.HasSubstructMatch(query, useChirality=True)):
                files.append(record)
        rows.append({
            "candidate_id": row["candidate_id"], "track": "赛道三-大规模虚拟筛选",
            "candidate_smiles": row["smiles"],
            "prediction_raw_logit": row["score_20261001"], "prediction_unit": "dimensionless",
            "primary_rank": int(row["rank_20261001"]),
            "seed_20261002_rank": row.get("rank_20261002", ""),
            "seed_20261003_rank": row.get("rank_20261003", ""),
            "nearest_training_similarity": row["nearest_any_similarity"],
            "nomination_order": row.get("nomination_order", ""),
            "model_version": model_version or "CheMeleon_GPR84_full_fit_2026-10-01_seed20261001",
            "run_version": run_version,
            "structure_pdb_files": ";".join(structure_prefix + "results/structures/" +
                                             Path(r["pdb"]).name for r in files),
            "ligand_sdf_files": ";".join(structure_prefix + "results/structures/" +
                                          Path(r["sdf"]).name for r in files),
            "md_completed_replicates": 3 if files and row["candidate_id"] in
                                       {"CHEMBL120691", "CHEMBL1086046"} else 0,
            "notes": scope + "；候选GPR84活性与拮抗方向未知；logit为排序分数；"
                     "种子排名仅描述敏感性；三维文件为条件性姿势假设",
        })
    columns = ["candidate_id", "track", "candidate_smiles", "prediction_raw_logit",
               "prediction_unit", "primary_rank", "seed_20261002_rank", "seed_20261003_rank",
               "nearest_training_similarity", "nomination_order", "model_version", "run_version",
               "structure_pdb_files", "ligand_sdf_files", "md_completed_replicates", "notes"]
    return pd.DataFrame(rows, columns=columns)


def screen_main(predict_only=False, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/catalogue/structures.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args(argv)
    output = path(args.output)
    started, clock = datetime.now(UTC).isoformat(), perf_counter()
    cfg = settings()
    raw = catalogue_input(args.input)
    if output.exists():
        raise FileExistsError("筛选输出目录已存在，不能覆盖")
    if importlib.metadata.version("chemprop") != "2.2.0":
        raise ValueError("Chemprop版本不符，请使用附带锁文件")
    units = pd.read_csv(ROOT / "data/training/training_units.csv")
    validate_training_scope(units, cfg["scope"])
    model_dir, archived = model_source(args.model_dir)
    ids = set(pd.read_csv(ROOT / "data/registry/excluded_known_ids.csv").known_id)
    connections = set(pd.read_csv(ROOT / "data/registry/excluded_connections.csv").connection_id)
    output.mkdir(parents=True, exist_ok=False)
    prepared = prepare_catalogue(raw, ids, connections, cfg["scope"])
    prepared.to_csv(output / "prepared_records.csv", index=False)
    ready = prepared[prepared.status.eq("ready_for_scoring")].copy().reset_index(drop=True)
    if ready.empty:
        save_json(output / "failure.json", {"reason": "无可评分输入", "records": len(raw)})
        raise ValueError("没有准入图，请检查prepared_records.csv")
    chunks = []
    seeds = cfg["training"]["seeds"]
    models = {s: load_model(s, args.device, model_dir) for s in seeds}
    for start in range(0, len(ready), cfg["runtime"]["processing_chunk"]):
        part = ready.iloc[start:start + cfg["runtime"]["processing_chunk"]].copy()
        graphs = GraphCache(part.smiles.tolist(), args.device)
        for seed, model in models.items():
            part[f"score_{seed}"] = predict(model, graphs, np.arange(len(part)), 16)
        chunks.append(part)
        print(f"{len(seeds)}个模型评分 {start + len(part)}/{len(ready)}", flush=True)
    scored = pd.concat(chunks, ignore_index=True)
    scored = pd.concat([scored, nearest_support(scored.smiles, units)], axis=1)
    scored = add_rankings(scored, seeds)
    scored.to_csv(output / "scored_records.csv", index=False)
    nominated, pool = nominate(scored, cfg["ranking"])
    nominated.to_csv(output / "candidate_hypotheses.csv", index=False)
    chosen = scored if predict_only else nominated
    scope = "归档权重完整目录重算" if archived else "本次正式重拟合权重完整目录筛选"
    run_version = cfg["package_version"] + "_full_" + started
    model_version = ("CheMeleon_GPR84_full_fit_2026-10-01_seed20261001" if archived else
                     "CheMeleon_GPR84_refit_seed20261001_" + json.loads(
                         (model_dir / "run_manifest.json").read_text(encoding="utf-8")
                     )["started_utc"])
    # 每次运行的三维文件路径明确相对代码包根目录，避免相对新输出目录误解。
    table = results_table(chosen, run_version, scope, model_version=model_version)
    table["structure_path_base"] = "package_root"
    table.to_csv(output / "results.csv", index=False, encoding="utf-8")
    manifest = metadata("predict" if predict_only else "screen",
        started, clock, args, input_records=len(raw), scored_graphs=len(scored),
        candidate_pool=pool, nominations=len(nominated), status_counts=prepared.status.value_counts().to_dict(),
        scope=scope, seeds=seeds, model_directory=str(model_dir),
        archived_weights=archived, model_version=model_version, run_version=run_version)
    if archived:
        expected = pd.read_csv(ROOT / "results/screening/candidate_hypotheses.csv")
        if nominated.candidate_id.tolist() != expected.candidate_id.tolist():
            manifest["status"] = "failed_archive_comparison"
            save_json(output / "run_manifest.json", manifest)
            raise ValueError("本次全量提名与归档结果不同，请保留运行记录核查")
    save_json(output / "run_manifest.json", manifest)
    print(f"完成：{len(raw)}条输入，{len(scored)}个评分图，结果在 {args.output}/results.csv")
    return output


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="按原固定配置训练277个实验内单位、30轮、三个种子")
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    started, clock = datetime.now(UTC).isoformat(), perf_counter()
    cfg = settings()
    units = pd.read_csv(ROOT / "data/training/training_units.csv")
    validate_training_scope(units, cfg["scope"])
    if len(units) != 277 or units.smiles.nunique() != 248:
        raise ValueError("正式训练要求277单位、248个输入图")
    output = path(args.output)
    if output.exists():
        raise FileExistsError("训练输出目录已存在，不能覆盖")
    training = dict(cfg["training"])
    torch = runtime(args.device)
    payload = torch.load(ROOT / "models/pretrained/chemeleon_mp.pt", weights_only=True,
                         map_location="cpu")
    if payload["hyper_parameters"] != cfg["encoder_hyper_parameters"]:
        raise ValueError("预训练架构与封存配置不一致")
    output.mkdir(parents=True, exist_ok=False)
    graphs = GraphCache(units.smiles.tolist(), args.device)
    indices = np.arange(len(units))
    histories, records = [], []
    for seed in training["seeds"]:
        model = build_model(payload, True, True, seed, training, args.device)
        history, record = fit_model(model, graphs, units, indices, seed,
                                   training["primary_learning_rate"], training)
        torch.save(tensors_cpu(model), output / f"model_{seed}.pt")
        restored = build_model(payload, True, True, seed, training, args.device)
        restored.load_state_dict(torch.load(output / f"model_{seed}.pt", weights_only=True,
                                           map_location="cpu"), strict=True)
        np.testing.assert_array_equal(predict(model, graphs, indices, 16),
                                      predict(restored, graphs, indices, 16))
        histories.extend({"seed": seed, **r} for r in history)
        records.append({"seed": seed, **record, "reload_exact": True})
    units.to_csv(output / "training_members.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
    save_json(output / "training_config.json", training)
    save_json(output / "model_records.json", records)
    save_json(output / "run_manifest.json", metadata("train_" + args.mode, started, clock, args,
        units=len(units), graphs=units.smiles.nunique(), epochs=training["epochs"],
        scope="按固定配置全量重拟合"))
    print(f"完成训练：输出 {args.output}")
    return output


def pipeline(input_csv, output_dir, device):
    """一键衔接正式训练（含Adam优化）与完整目录筛选，不运行MD或改变科研规则。"""
    output = path(output_dir)
    if output.exists():
        raise FileExistsError("流程输出目录已存在，不能覆盖")
    # 先检查目录输入契约，避免无效输入触发正式训练。
    source = path(input_csv)
    catalogue_input(source)
    output.mkdir(parents=True, exist_ok=False)
    try:
        fitted = train_main(argv=["--output", str(output / "training"), "--device", device])
        screened = screen_main(argv=["--input", str(source), "--output", str(output / "screening"),
                                     "--model-dir", str(fitted), "--device", device])
        # CSV结构路径继续以包根目录为基准；不会为新提名生成未计算的三维证据。
        (output / "results.csv").write_bytes((screened / "results.csv").read_bytes())
        save_json(output / "pipeline_manifest.json", {
            "status": "complete", "workflow": "train_optimize_screen", "device": device,
            "input": str(source), "training_output": str(fitted), "screening_output": str(screened),
            "result_file": "results.csv", "training_config": "training/training_config.json",
            "md_recomputed": False,
        })
    except Exception as error:
        save_json(output / "pipeline_manifest.json", {
            "status": "failed", "workflow": "train_optimize_screen", "error": str(error),
        })
        raise
    return output
