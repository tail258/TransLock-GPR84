"""只读既有诊断产物，导出带用途限制的表格、说明卡和可视化复合物。"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem

from translock_gpr84.docking.mixture_diagnostic import aggregate_samples, evaluate
from translock_gpr84.docking.mixture_inputs import validate_coordinates

ROLE = {
    "record_role": "retrospective_diagnostic",
    "primary_eligible": False,
    "submission_candidate": False,
    "use_restriction": "not_submission_candidate",
}
SCOPE = "external_retrospective_conditional_mixture_diagnostic_not_gate4_validation"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _indexed(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row[key]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError(f"不安全的标识符：{key}")
        if identifier in result:
            raise ValueError(f"重复标识符：{key}={identifier}")
        if row.get("primary_eligible", False) or row.get("submission_candidate", False):
            raise ValueError("禁止将主准入或候选记录混入诊断导出")
        result[identifier] = row
    return result


def _same(actual, expected, context: str) -> None:
    """核对已有数值，不产生新预测；只容许浮点回算尾差。"""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"归档内容不符：{context}")
        for key in expected:
            _same(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"归档内容不符：{context}")
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same(left, right, f"{context}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"归档数值类型不符：{context}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"归档数值不符：{context}")
    elif actual != expected:
        raise ValueError(f"归档内容不符：{context}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    """面向可复现脚本的 UTF-8/LF 数据表；缺失值留空，不代入零。"""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: str(value).lower() if isinstance(value, bool) else value
                    for key, value in row.items()
                }
            )


def _complex_pdb(receptor_text: str, molecule: Chem.Mol) -> tuple[str, str]:
    """受体记录保留，配体另占未用链；PDB 仅供可视化，SDF 保存化学身份。"""
    lines = receptor_text.splitlines()
    if any(line.startswith("MODEL ") for line in lines):
        raise ValueError("仅支持单模型受体")
    atoms = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    if not atoms:
        raise ValueError("受体没有原子记录")
    chains = {line[21] for line in atoms}
    available = [
        letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if letter not in chains
    ]
    if not available:
        raise ValueError("没有可用配体链编号")
    chain = available[0]
    numbered = [
        line
        for line in lines
        if line.startswith(("ATOM  ", "HETATM", "TER   ")) and line[6:11].strip()
    ]
    serial = max(int(line[6:11]) for line in numbered) + 1
    if serial + molecule.GetNumAtoms() > 99999:
        raise ValueError("PDB 原子序号超出格式范围")
    ligand = []
    for i, (atom, xyz) in enumerate(
        zip(molecule.GetAtoms(), molecule.GetConformer().GetPositions(), strict=True)
    ):
        if not np.isfinite(xyz).all() or any(len(f"{value:8.3f}") > 8 for value in xyz):
            raise ValueError("配体坐标不能无溢出写入 PDB")
        name = f"{atom.GetSymbol()}{i + 1}"
        if len(name) > 4:
            raise ValueError("配体原子名超出 PDB 格式范围")
        charge = atom.GetFormalCharge()
        charge_text = f"{abs(charge)}{'+' if charge > 0 else '-'}" if charge else "  "
        if abs(charge) > 9:
            raise ValueError("电荷超出 PDB 格式范围")
        ligand.append(
            f"HETATM{serial + i:5d} {name:<4s} UNL {chain}   1    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
            f"          {atom.GetSymbol():>2s}{charge_text}"
        )
    # 只移除终止标记以追加配体；所有原受体原子、TER 和 CONECT 行逐字保留。
    body = [line for line in lines if line.strip() not in {"END", "ENDMDL"}]
    remarks = [
        "REMARK 900 RETROSPECTIVE_DIAGNOSTIC; NOT_SUBMISSION_CANDIDATE",
        "REMARK 900 PRIMARY_ELIGIBLE=FALSE; SUBMISSION_CANDIDATE=FALSE",
        "REMARK 900 VISUALIZATION ONLY; LIGAND SDF IS CHEMICAL IDENTITY SOURCE",
        "REMARK 900 PREDICTED POSE; ANGSTROM; NO ALIGNMENT OR MINIMIZATION",
    ]
    # CONECT 属于坐标区之后；部分 PDB 解析器遇到它即停止读取坐标。
    tail_start = next(
        (i for i, line in enumerate(body) if line.startswith(("CONECT", "MASTER"))), len(body)
    )
    return "\n".join(remarks + body[:tail_start] + ligand + body[tail_start:] + ["END", ""]), chain


def export_structures(
    source: Path, output: Path, selected: list[dict], results: dict, receptor: Path
) -> list[dict]:
    """仅导出预先指定样品的全部状态；沿用归档的姿势选择，不另行排名。"""
    receptor_text = receptor.read_text(encoding="utf-8")
    original_atoms = [
        line for line in receptor_text.splitlines() if line.startswith(("ATOM  ", "HETATM"))
    ]
    rows = []
    for state in selected:
        identifier = state["state_id"]
        result = results[identifier]
        if result["status"] != "valid":
            raise ValueError("结构示例的全部预登记状态必须成功")
        rank = result["best_pose_rank"]
        path = source / "states" / identifier / "poses.sdf"
        molecules = list(Chem.SDMolSupplier(str(path), removeHs=False))
        if len(molecules) != len(result["poses"]) or not 1 <= rank <= len(molecules):
            raise ValueError("原始 SDF 与归档姿势数不符")
        molecule = molecules[rank - 1]
        checks = validate_coordinates(molecule, state["smiles"])
        _same(float(molecule.GetProp("CNNscore")), result["score"], f"{identifier}.CNNscore")
        for key, value in ROLE.items():
            molecule.SetProp(key, str(value).lower() if isinstance(value, bool) else value)
        molecule.SetProp("state_id", identifier)
        molecule.SetProp("sample_id", state["sample_id"])
        molecule.SetProp("source_pose_rank", str(rank))
        molecule.SetProp("state_interpretation", state["interpretation"])
        sdf_name, pdb_name = f"structures/{identifier}.sdf", f"structures/{identifier}.pdb"
        with Chem.SDWriter(str(output / sdf_name)) as writer:
            writer.write(molecule)
        readback = Chem.SDMolSupplier(str(output / sdf_name), removeHs=False)[0]
        validate_coordinates(readback, state["smiles"])
        if not np.allclose(
            readback.GetConformer().GetPositions(),
            molecule.GetConformer().GetPositions(),
            atol=0.0001,
            rtol=0,
        ):
            raise ValueError("SDF 导出改变坐标")
        pdb, chain = _complex_pdb(receptor_text, molecule)
        (output / pdb_name).write_text(pdb, encoding="utf-8")
        pdb_atoms = [line for line in pdb.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
        if pdb_atoms[: len(original_atoms)] != original_atoms:
            raise ValueError("PDB 导出改变受体原子")
        ligand_xyz = np.array(
            [
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                for line in pdb_atoms[len(original_atoms) :]
            ]
        )
        delta = float(np.abs(ligand_xyz - molecule.GetConformer().GetPositions()).max())
        if delta > 0.000501:
            raise ValueError("PDB 导出改变配体坐标，超过三位小数精度")
        rows.append(
            {
                **ROLE,
                "sample_id": state["sample_id"],
                "state_id": identifier,
                "smiles": state["smiles"],
                "source_pose_rank": rank,
                "stored_CNNscore": result["score"],
                "ligand_sdf": sdf_name,
                "visualization_complex_pdb": pdb_name,
                "ligand_chain": chain,
                "coordinate_unit": "angstrom",
                "pdb_coordinate_max_rounding_A": delta,
                "receptor_atom_records_unchanged": True,
                "identity": checks,
            }
        )
    return rows


def _cards(
    output: Path,
    manifest: dict,
    source_meta: dict,
    audit: dict,
    samples: list,
    structure_sample: str,
) -> None:
    config = manifest["config"]
    citation = source_meta["source"]
    retrieval = source_meta["retrieval"]
    counts = len(samples), config["expected_states"], sum(s["strong_at_1uM"] for s in samples)
    missing = sum(s.get("main_sem") is None for s in samples)
    si_covered = sum(s.get("SI_mean") is not None and s.get("SI_sem") is not None for s in samples)
    data_card = f"""# 诊断数据卡

用途：retrospective_diagnostic；primary_eligible=false；submission_candidate=false。
本包不是候选清单，不能提交为本项目新发现分子；不代表 Gate 4 通过。

来源：{citation["title"]}，DOI {citation["doi"]}，{citation["citation"]}。
正文：{retrieval["main_pdf"]}（{retrieval["main_license"]}）。
作者 CSV：{retrieval["supporting_csv"]["url"]}（{retrieval["supporting_csv"]["license"]}）。
原始下载日期：{retrieval["query_date"]}。本包只读取既有诊断产物，未重新下载原始材料。

共 {counts[0]} 份实验样品、{counts[1]} 个条件性状态；{counts[2]} 强、{counts[0] - counts[2]} 弱。
强类按来源实验均值 IC50≤1 μM；弱类不等于绝对无活性。
主表逐项 SEM 缺失 {missing}/{counts[0]}；空单元格表示未报告/不可用，不是零。
SI 中具有均值和 SEM 的样品为 {si_covered}/{counts[0]}，不以此填补其他样品缺失值。
main_pic50 来自原论文实验结果，stored_CNNscore 来自历史计算，二者不得互换。
样品均为作者外消旋样品；状态 A/B 是条件性计算假设，不是实验 E1/E2 的身份确认。
样品表保留实验标签；状态表仅以 sample_id 关联，不把样品活性赋予单一微观状态。

原评分开始时间：{manifest["started_utc"]}；标签与评分均已见，不能称为盲测。
纳入是来源复核后的条件性诊断集合，不是主验证准入；50c/50d 隔离未改变。
训练参照版本：{config["entities"]}；固定成员登记：{config["membership"]}。
基线参照 {audit["time_train_count"]} 个训练实体，其中 {audit["positive_count"]} 个强类。
原系列标签/结构审查限制继续适用，输出不表示剩余源数据全部正确。

样品表与状态表保留全部归档记录；当前仅样品 {structure_sample} 导出所有预登记状态复合物。
其他状态的结构文件字段留空表示本示例未导出，不表示原始姿势不存在。
PDB 是预测姿势可视化文件；化学连接、键级、手性、电荷以配套 SDF/SMILES 为准。
坐标单位 Å；沿用原受体坐标系，未旋转、平移或优化。PDB 三位小数有≤0.0005 Å舍入。
此导出不包含新化学空间筛选、实验验证或对接筛选有效性认证。
"""
    model_card = f"""# 诊断方法与模型卡

用途：retrospective_diagnostic；primary_eligible=false；submission_candidate=false。
这是既有结果的离线导出。没有模型训练、新预测、新对接或新分数。

原模型：{manifest["gnina_version"]}，内置默认 CNN 集成。
原评分时间：{manifest["started_utc"]}。
原参数：{json.dumps(manifest["frozen_engine"], ensure_ascii=False, sort_keys=True)}。
软件环境：{json.dumps(manifest["runtime"]["packages"], sort_keys=True)}。
本包不分发 GNINA 二进制或权重；其原始分发许可未在本次导出中重新核验。

输入：现存样品/状态结果、基线结果、运行快照和预先指定样品的原始 SDF 姿势。
输出：样品表、状态表、汇总复核、说明卡及可视化复合物。
每状态沿用归档有效姿势中最高 CNNscore 的既定姿势；样品同时报告状态 max/min。
两种聚合均不代表混合物药效模型；CNNscore 不是活性概率、实验 IC50 或结合自由能。
相似度基线：ECFP 半径 2、2048 位、不编码手性；读取已存相似度，不重新计算。
RF：历史未导出权重，输出 rf_probability 为空；本入口不可推断新样品，未调用 fit。
已知限制：单文献、少量强类、条件性状态组成未知、SEM 缺失、聚合敏感性与适用域限制。
GNINA 预训练重叠未完全核清；没有优于冻结 RF 的证据；Gate 4 未通过。

重现范围仅是“归档诊断产物 → 标准字段与结构文件”。不能代替完整设计/筛选主入口。
正式候选生产仍须项目后续决策；诊断行不得通过改名被提升为提交候选。
"""
    (output / "DATA_CARD.md").write_text(data_card, encoding="utf-8")
    (output / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")


def run(source: Path, output: Path, structure_sample: str) -> dict:
    """只消费完整、已完成的混合物诊断归档，不覆盖源目录或已有输出。"""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("已有输出拒绝覆盖")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("导出与输入目录必须互相独立")
    manifest = _read_json(source / "manifest.json")
    config = manifest["config"]
    if manifest["status"] != "completed" or config.get("primary_eligible") is not False:
        raise ValueError("仅允许已完成、非主准入的诊断归档")
    if (
        config["scope"] != SCOPE
        or config["random_forest"] != "unavailable_no_serialized_weights_no_refitting"
    ):
        raise ValueError("归档研究范围改变，不由本入口解释")
    states = _read_json(source / "snapshots/state_preparation.json")
    results = _read_json(source / "state_results.json")
    samples = _read_json(source / "sample_results.json")
    baselines = _read_json(source / "baselines.json")
    state_index = _indexed(states, "state_id")
    result_index = _indexed(results, "state_id")
    sample_index = _indexed(samples, "sample_id")
    baseline_index = _indexed(baselines, "sample_id")
    expected = {str(i) for i in config["expected_samples"]}
    if set(sample_index) != expected or set(baseline_index) != expected:
        raise ValueError("样品与预登记集合不符")
    if set(result_index) != set(state_index) or len(states) != config["expected_states"]:
        raise ValueError("状态缺失或多出")
    for state in states:
        result = result_index[state["state_id"]]
        if state["sample_id"] not in expected or result["sample_id"] != state["sample_id"]:
            raise ValueError("状态与样品对应不符")
        if result["status"] == "valid":
            valid = [pose for pose in result["poses"] if pose["valid"]]
            if not valid:
                raise ValueError("状态没有归档有效姿势")
            best = max(valid, key=lambda pose: pose["scores"]["CNNscore"])
            _same(result["score"], best["scores"]["CNNscore"], "状态最高分")
            _same(result["best_pose_rank"], best["rank"], "状态姿势编号")
    computed = aggregate_samples(baselines, states, results)
    for sample in computed:
        _same(sample, sample_index[sample["sample_id"]], f"样品{sample['sample_id']}")
    summary = _read_json(source / "summary.json")
    metrics = evaluate([row for row in computed if row["complete"]], config["top_k"])
    _same(metrics, summary["common_complete_metrics"], "完整样品指标")
    selected = [state for state in states if state["sample_id"] == structure_sample]
    if not selected or {state["stereo"] for state in selected} != {"A", "B"}:
        raise ValueError("示例必须包含所选样品全部预登记 A/B 状态")
    fixed = yaml.safe_load((source / "snapshots/gate4_pilot.yaml").read_text(encoding="utf-8"))
    receptor = source / "snapshots" / Path(fixed["paths"]["receptor_pdb"]).name
    source_meta = yaml.safe_load(
        (source / "snapshots/jmedchem_5c03367.yaml").read_text(encoding="utf-8")
    )
    audit = _read_json(source / "baseline_audit.json")
    output.mkdir(parents=True, exist_ok=False)
    (output / "structures").mkdir()
    run_record = {
        **ROLE,
        "status": "running",
        "source_run_name": source.name,
        "source_version": config["version"],
        "exporter_version": "diagnostic_export_v1",
        "export_runtime": {
            "python": sys.version.split()[0],
            "packages": {
                name: version(name) for name in ["rdkit", "numpy", "scikit-learn", "pyyaml"]
            },
        },
        "exported_utc": datetime.now(UTC).isoformat(),
        "scores_generated": False,
        "models_fitted": False,
        "structure_sample": structure_sample,
    }
    _write_json(output / "export_manifest.json", run_record)
    try:
        structures = export_structures(source, output, selected, result_index, receptor)
        structure_index = {row["state_id"]: row for row in structures}
        sample_rows = [
            {
                **ROLE,
                "sample_id": row["sample_id"],
                "source_doi": row["source_doi"],
                "sample_type": row["sample_type"],
                "main_pic50": row["main_pic50"],
                "main_sem": row.get("main_sem"),
                "main_sem_status": row.get("main_sem_status"),
                "SI_mean": row.get("SI_mean"),
                "SI_sem": row.get("SI_sem"),
                "strong_at_1uM": row["strong_at_1uM"],
                "state_count": row["state_count"],
                "complete": row["complete"],
                "maximum_over_states": row["maximum_over_states"],
                "minimum_over_states": row["minimum_over_states"],
                "stored_positive_similarity": row["positive_similarity"],
                "nearest_positive_key": row["nearest_positive_key"],
                "rf_probability": row["rf_probability"],
                "state_ids": ";".join(
                    s["state_id"] for s in states if s["sample_id"] == row["sample_id"]
                ),
                "run_version": config["version"],
            }
            for row in samples
        ]
        state_rows = [
            {
                **ROLE,
                "sample_id": state["sample_id"],
                "state_id": state["state_id"],
                "smiles": state["smiles"],
                "ionization": state["ionization"],
                "stereo": state["stereo"],
                "interpretation": state["interpretation"],
                "status": result_index[state["state_id"]]["status"],
                "stored_CNNscore": result_index[state["state_id"]]["score"],
                "source_pose_rank": result_index[state["state_id"]].get("best_pose_rank"),
                "ligand_sdf": structure_index.get(state["state_id"], {}).get("ligand_sdf"),
                "visualization_complex_pdb": structure_index.get(state["state_id"], {}).get(
                    "visualization_complex_pdb"
                ),
                "run_version": config["version"],
            }
            for state in states
        ]
        _write_csv(output / "diagnostic_samples.csv", sample_rows)
        _write_csv(output / "diagnostic_states.csv", state_rows)
        _write_json(output / "diagnostic_samples.json", sample_rows)
        _write_json(output / "diagnostic_states.json", state_rows)
        _write_json(output / "structure_readback.json", structures)
        _write_json(output / "replayed_metrics.json", {**ROLE, "common_complete_metrics": metrics})
        _cards(output, manifest, source_meta, audit, samples, structure_sample)
        run_record.update(
            status="completed",
            sample_count=len(samples),
            state_count=len(states),
            structure_count=len(structures),
            stored_sample_aggregates_match=True,
            stored_metrics_match=True,
            source_files_modified=False,
        )
    except Exception as error:
        run_record.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(output / "export_manifest.json", run_record)
        raise
    _write_json(output / "export_manifest.json", run_record)
    return run_record
