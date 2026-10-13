"""固定条件性混合物诊断，按样品聚合；无训练与自动调参。"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors

from translock_gpr84.data.quarantine import quarantine_mask
from translock_gpr84.docking.mixture_inputs import validate_coordinates
from translock_gpr84.docking.pilot import (
    fixed_command,
    gnina_env,
    ranking_metrics,
    run_command,
    runtime,
    write_json,
)
from translock_gpr84.models.ecfp_baseline import build_fingerprints


def pose_audit(path: Path, smiles: str, engine: dict, config: dict) -> list[dict]:
    spacing = config["grid_spacing_A"]
    size = np.ceil(np.asarray(engine["size"]) / spacing) * spacing
    lower, upper = np.asarray(engine["center"]) - size / 2, np.asarray(engine["center"]) + size / 2
    tolerance = config["grid_tolerance_A"]
    rows = []
    for rank, molecule in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), 1):
        row = {"rank": rank, "valid": False}
        try:
            geometry = validate_coordinates(molecule, smiles)
            scores = {
                k: float(molecule.GetProp(k))
                for k in ["CNNscore", "minimizedAffinity", "CNNaffinity", "CNN_VS"]
                if molecule.HasProp(k)
            }
            if "CNNscore" not in scores or not all(np.isfinite(list(scores.values()))):
                raise ValueError("分数缺失或非有限")
            if not 0 <= scores["CNNscore"] <= 1:
                raise ValueError("CNNscore越出0到1，不能使用-1失败下界")
            xyz = Chem.RemoveHs(molecule).GetConformer().GetPositions()
            if not ((xyz >= lower - tolerance) & (xyz <= upper + tolerance)).all():
                raise ValueError("重原子越出有效网格")
            row.update(valid=True, scores=scores, geometry=geometry)
        except (ValueError, RuntimeError) as error:
            row["reason"] = str(error)
        rows.append(row)
    return rows


def aggregate_samples(samples: list[dict], states: list[dict], results: list[dict]) -> list[dict]:
    result_ids = [r["state_id"] for r in results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("状态结果重复")
    lookup = {r["state_id"]: r for r in results}
    if set(lookup) - {s["state_id"] for s in states}:
        raise ValueError("出现未预登记状态")
    rows = []
    for sample in samples:
        expected = [s["state_id"] for s in states if s["sample_id"] == sample["sample_id"]]
        if not expected:
            raise ValueError("样品没有输入状态")
        successful = [lookup[k] for k in expected if k in lookup and lookup[k]["status"] == "valid"]
        complete = len(successful) == len(expected)
        values = [s["score"] for s in successful]
        rows.append(
            {
                **sample,
                "complete": complete,
                "state_count": len(expected),
                "valid_states": len(successful),
                "failed_or_missing_states": [
                    k for k in expected if k not in lookup or lookup[k]["status"] != "valid"
                ],
                "partial_max": max(values) if values else None,
                "partial_min": min(values) if values else None,
                "maximum_over_states": max(values) if complete else -1.0,
                "minimum_over_states": min(values) if complete else -1.0,
            }
        )
    return rows


def evaluate(rows: list[dict], k: int) -> dict:
    if not rows:
        return {"n": 0, "not_evaluable": "无完整样品"}
    labels = [int(r["strong_at_1uM"]) for r in rows]
    return {
        name: ranking_metrics(labels, [r[name] for r in rows], k=k)
        for name in [
            "maximum_over_states",
            "minimum_over_states",
            "positive_similarity",
            "molecular_weight",
        ]
    }


def fixed_baselines(samples: list[dict], states: list[dict], root: Path, config: dict) -> tuple:
    entities = pd.read_parquet(root / config["entities"])
    policy = json.loads(Path(config["quarantine_policy"]).read_text(encoding="utf-8"))
    if quarantine_mask(entities, policy).any():
        raise ValueError("基线实体仍含隔离身份")
    membership = json.loads((root / config["membership"]).read_text(encoding="utf-8"))
    keys = {
        r["key"]
        for r in membership
        if r["task"] == "classification"
        and r["split"] == "time_2025"
        and r["role"] == "train"
        and not r["quarantined"]
    }
    train = entities[entities.standardized_inchikey.isin(keys)].sort_values("standardized_inchikey")
    if set(train.standardized_inchikey) != keys or not (train.first_year <= 2024).all():
        raise ValueError("固定时间训练成员不符")
    positives = train[train.median_ic50_nm <= 1000]
    _, fps = build_fingerprints(positives.standardized_smiles.tolist(), radius=2, size=2048)
    if not fps:
        raise ValueError("基线无训练强类")
    train_connectivity = {key.split("-")[0] for key in keys}
    rows = []
    for sample in samples:
        state = next(
            s
            for s in states
            if s["sample_id"] == sample["sample_id"]
            and s["ionization"] == "author_neutral"
            and s["stereo"] == "A"
        )
        molecule = Chem.MolFromSmiles(state["smiles"])
        if Chem.MolToInchiKey(molecule).split("-")[0] in train_connectivity:
            raise ValueError("新样品与训练连接结构重叠")
        _, fp = build_fingerprints([state["smiles"]], radius=2, size=2048)
        similarities = DataStructs.BulkTanimotoSimilarity(fp[0], fps)
        best = int(np.argmax(similarities))
        rows.append(
            {
                **sample,
                "positive_similarity": float(similarities[best]),
                "nearest_positive_key": positives.iloc[best].standardized_inchikey,
                "molecular_weight": Descriptors.MolWt(molecule),
                "rf_probability": None,
            }
        )
    return rows, {
        "time_train_count": len(train),
        "positive_count": len(positives),
        "train_keys": sorted(keys),
        "rf": "unavailable_no_weights_no_fit",
    }


def run(config_path: Path, root: Path, gnina: Path) -> dict:
    start = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["primary_eligible"] or config["automatic_retry"] or config["parameter_tuning"]:
        raise ValueError("诊断边界设置错误")
    prepared = root / config["prepared"]
    states = json.loads((prepared / "state_preparation.json").read_text(encoding="utf-8"))
    samples = json.loads((prepared / "sample_evidence.json").read_text(encoding="utf-8"))
    expected = {str(i) for i in config["expected_samples"]}
    if (
        len(samples) != len(expected)
        or {s["sample_id"] for s in samples} != expected
        or len(states) != config["expected_states"]
        or len({s["state_id"] for s in states}) != len(states)
        or {s["sample_id"] for s in states} != expected
    ):
        raise ValueError("固定样品/输入集合不符")
    for state in states:
        if state["status"] != "prepared":
            raise ValueError("输入未验收")
        validate_coordinates(
            Chem.SDMolSupplier(str(prepared / state["sdf"]), removeHs=False)[0], state["smiles"]
        )
    fixed = yaml.safe_load((prepared / "snapshots/gate4_pilot.yaml").read_text(encoding="utf-8"))
    if fixed["engine"]["autobox_extend"]:
        raise ValueError("禁止扩盒")
    output = (root / config["output"]).resolve()
    if not output.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("输出必须位于artifacts")
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "snapshots"
    shutil.copytree(prepared / "snapshots", snapshots)
    shutil.copytree(prepared / "sdf", output / "inputs")
    source_paths = [
        config_path,
        Path(config["quarantine_policy"]),
        root / config["entities"],
        root / config["membership"],
        prepared / "state_preparation.json",
        prepared / "sample_evidence.json",
    ]
    for path in source_paths:
        shutil.copyfile(path, snapshots / path.name)
    for path in [
        Path(__file__),
        Path(fixed_baselines.__code__.co_filename),
        Path(build_fingerprints.__code__.co_filename),
        Path(quarantine_mask.__code__.co_filename),
    ]:
        if not (snapshots / path.name).exists():
            shutil.copyfile(path, snapshots / path.name)
    version = subprocess.run(
        [str(gnina), "--version"],
        env=gnina_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    manifest = {
        "status": "started",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "runtime": runtime(),
        "gnina": str(gnina),
        "gnina_version": version.stdout.strip(),
        "frozen_engine": fixed["engine"],
        "independent_blind": False,
        "screening_validated": False,
        "scores_exposure": "this_run_is_first_new_project_score_generation",
        "inputs": [{"path": str(p), "bytes": p.stat().st_size} for p in source_paths],
    }
    write_json(output / "manifest.json", manifest)
    rows, baseline_audit = fixed_baselines(samples, states, root, config)
    write_json(output / "baselines.json", rows)
    write_json(output / "baseline_audit.json", baseline_audit)
    receptor = snapshots / Path(fixed["paths"]["receptor_pdb"]).name
    results = []
    for index, state in enumerate(states):
        directory = output / "states" / state["state_id"]
        pose_path = directory / "poses.sdf"
        result = {
            "state_id": state["state_id"],
            "sample_id": state["sample_id"],
            "status": "not_run_session_budget",
            "score": None,
            "poses": [],
        }
        remaining = config["max_session_seconds"] - (time.perf_counter() - start)
        if remaining >= config["max_state_seconds"]:
            command = fixed_command(
                gnina,
                receptor,
                output / "inputs" / Path(state["sdf"]).name,
                pose_path,
                fixed["engine"],
            )
            execution = run_command(command, directory, config["max_state_seconds"])
            result["execution"] = execution
            result["status"] = execution["status"]
            if execution["status"] == "completed":
                try:
                    poses = pose_audit(pose_path, state["smiles"], fixed["engine"], config)
                    valid = [p for p in poses if p["valid"]]
                    result.update(poses=poses, status="valid" if valid else "no_valid_pose")
                    if valid:
                        best = max(valid, key=lambda p: p["scores"]["CNNscore"])
                        result.update(score=best["scores"]["CNNscore"], best_pose_rank=best["rank"])
                except (ValueError, RuntimeError, OSError) as error:
                    result.update(status="pose_read_failure", error=str(error))
        results.append(result)
        write_json(output / "state_results.json", results)
        print(f"{index + 1}/{len(states)} {state['state_id']}: {result['status']}", flush=True)
    aggregate = aggregate_samples(rows, states, results)
    common = [r for r in aggregate if r["complete"]]
    summary = {
        "n_samples": len(rows),
        "n_states": len(states),
        "complete_samples": len(common),
        "valid_states": sum(r["status"] == "valid" for r in results),
        "total_poses": sum(len(r["poses"]) for r in results),
        "valid_poses": sum(p["valid"] for r in results for p in r["poses"]),
        "failed_states": [r["state_id"] for r in results if r["status"] != "valid"],
        "common_complete_metrics": evaluate(common, config["top_k"]),
        "all_samples_failure_worst_metrics": evaluate(aggregate, config["top_k"]),
        "without_boundary_sample_16": evaluate(
            [
                r
                for r in aggregate
                if r["sample_id"] != config["threshold_sensitivity_exclude_sample"]
            ],
            config["top_k"],
        ),
        "rf_comparison": "not_available_weights_not_exported_no_refitting",
        "screening_validated": False,
        "primary_eligible": 0,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    write_json(output / "sample_results.json", aggregate)
    write_json(output / "summary.json", summary)
    manifest.update(
        status="completed", scores_generated=True, elapsed_seconds=summary["elapsed_seconds"]
    )
    write_json(output / "manifest.json", manifest)
    return summary
