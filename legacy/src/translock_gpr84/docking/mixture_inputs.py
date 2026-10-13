"""条件性混合物输入准备；不调用评分、不将样品标签复制给微观状态。"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from translock_gpr84.data.mixture_review import mirror_smiles
from translock_gpr84.docking.pilot import prepare_smiles, runtime, write_json


def ion_state(smiles: str, kind: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("无效结构")
    if kind == "author_neutral":
        if Chem.GetFormalCharge(molecule) != 0:
            raise ValueError("作者中性状态的电荷不为0")
        return Chem.MolToSmiles(molecule)
    patterns = {
        "secondary_amine_protonated": "[N;H1;+0;!$(N-C=O);!$(N-S)]",
        "sulfamic_OH_deprotonated": "[O;H1;+0]-S(=O)(=O)-N",
    }
    if kind not in patterns:
        raise ValueError("未登记的离子状态")
    hits = molecule.GetSubstructMatches(Chem.MolFromSmarts(patterns[kind]))
    if len(hits) != 1:
        raise ValueError("质子化位点不是唯一匹配")
    atom = molecule.GetAtomWithIdx(hits[0][0])
    atom.SetFormalCharge(1 if kind == "secondary_amine_protonated" else -1)
    atom.SetNumExplicitHs(2 if kind == "secondary_amine_protonated" else 0)
    atom.SetNoImplicit(True)
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule)


def states_for_sample(record: dict, policy: dict) -> list[dict]:
    if record["issues"] or not record["provisional_pair_review_candidate"]:
        raise ValueError("仍有逐条疑点的样品禁止进入输入准备")
    smiles = record["table_structure"]["smiles"]
    molecule = Chem.MolFromSmiles(smiles)
    centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    if len(centers) != 3 or any(c == "?" for _, c in centers):
        raise ValueError("此入口仅接受三个已指定核心中心的条件性镜像对")
    states = []
    for kind in policy.get(str(record["compound"]), ["author_neutral"]):
        representative = ion_state(smiles, kind)
        for letter, state in [("A", representative), ("B", mirror_smiles(representative))]:
            states.append({
                "state_id": f"c{record['compound']}_{kind}_{letter}",
                "sample_id": str(record["compound"]),
                "ionization": kind,
                "stereo": letter,
                "smiles": state,
                "interpretation": "conditional_not_experimental_E1_E2",
            })
    keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(s["smiles"])) for s in states]
    if len(keys) != len(set(keys)):
        raise ValueError("枚举状态重复")
    return states


def validate_coordinates(molecule: Chem.Mol, expected: str) -> dict:
    """独立清除二维标签后从坐标判型，不能只相信SDF原有的手性标签。"""
    if molecule is None or molecule.GetNumConformers() != 1:
        raise ValueError("SDF缺失有效的单一构象")
    coordinates = molecule.GetConformer().GetPositions()
    if not molecule.GetConformer().Is3D() or not np.isfinite(coordinates).all():
        raise ValueError("坐标不是有效三维有限数值")
    expected_mol = Chem.MolFromSmiles(expected)
    expected_key = Chem.MolToInchiKey(expected_mol)
    if Chem.MolToInchiKey(Chem.RemoveHs(molecule)) != expected_key:
        raise ValueError("SDF记录的分子图/手性与输入不同")
    independent = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(independent)
    Chem.AssignStereochemistryFrom3D(independent, replaceExistingTags=True)
    if Chem.MolToInchiKey(Chem.RemoveHs(independent)) != expected_key:
        raise ValueError("坐标推断手性与输入不同")
    heavy = Chem.RemoveHs(molecule)
    xyz = heavy.GetConformer().GetPositions()
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    graph_distances = Chem.GetDistanceMatrix(heavy)
    nonlocal_distances = distances[graph_distances > 2]
    bond_lengths = [distances[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in heavy.GetBonds()]
    if min(bond_lengths) < 0.8 or max(bond_lengths) > 2.2:
        raise ValueError("明显异常键长")
    if len(nonlocal_distances) and min(nonlocal_distances) < 0.8:
        raise ValueError("存在明显非邻接重原子重叠")
    return {
        "full_inchikey": expected_key,
        "formula": rdMolDescriptors.CalcMolFormula(heavy),
        "charge": Chem.GetFormalCharge(heavy),
        "coordinate_stereo_matches": True,
        "heavy_atom_count": heavy.GetNumAtoms(),
        "bond_length_range_A": [float(min(bond_lengths)), float(max(bond_lengths))],
        "min_nonlocal_heavy_distance_A": float(min(nonlocal_distances)),
        "scope": "gross_geometry_check_not_bound_pose_or_energy_validation",
    }


def run(config_path: Path, root: Path) -> dict:
    start = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["predict_or_dock"] or config["primary_eligible"]:
        raise ValueError("仅允许非主验证的无评分输入准备")
    records_path = root / config["source_review"]
    all_records = json.loads(records_path.read_text(encoding="utf-8"))
    records = [r for r in all_records if r["provisional_pair_review_candidate"]]
    expected = {str(i) for i in config["expected_ids"]}
    if len(records) != len(expected) or {str(r["compound"]) for r in records} != expected:
        raise ValueError("固定12条集合改变")
    evidence = {str(r["compound"]): r for r in config["spectral_review"]["records"]}
    if len(config["spectral_review"]["records"]) != len(expected) or set(evidence) != expected:
        raise ValueError("谱图证据集合不完整或重复")
    box_path = Path(config["fixed_docking_config"])
    fixed = yaml.safe_load(box_path.read_text(encoding="utf-8"))
    if fixed["engine"]["autobox_extend"]:
        raise ValueError("不得改变固定盒策略")
    receptor = root / fixed["paths"]["receptor_pdb"]
    sources = [config_path, records_path, box_path, receptor,
               Path(config["source_review_config"]), Path(config["source_manifest"])]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    # 在创建输出前完整检查选样与枚举，不在中途隐性改变集合。
    states = [s for r in records for s in states_for_sample(r, config["ionization_policy"])]
    output = (root / config["output"]).resolve()
    if not output.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("输出必须位于artifacts")
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    for source in sources:
        shutil.copyfile(source, snapshots / source.name)
    for source in [Path(__file__), Path(prepare_smiles.__code__.co_filename),
                   Path(mirror_smiles.__code__.co_filename)]:
        shutil.copyfile(source, snapshots / source.name)
    manifest = {
        "version": config["version"], "status": "started",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": runtime(), "seed": config["seed"],
        "inputs": [{"path": str(p), "bytes": p.stat().st_size} for p in sources],
        "fixed_box": fixed["engine"], "scores_computed": False, "primary_eligible": 0,
        "sample_count": len(records), "state_count": len(states),
    }
    write_json(output / "manifest.json", manifest)
    sample_rows = []
    for row in records:
        entry = evidence[row["compound"]]
        sample_rows.append({
            "sample_id": row["compound"], "source_doi": row["source_doi"],
            "sample_type": "author_racemate_conditional_pair_assumption",
            "main_pic50": row["main_pic50"], "main_sem": None,
            "main_sem_status": "not_reported_in_table",
            "SI_mean": entry.get("si_mean"), "SI_sem": entry.get("si_sem"),
            "strong_at_1uM": row["strong_at_1uM"],
            "threshold_caution": entry.get("threshold_caution"),
            "spectral_evidence": entry, "primary_eligible": False,
        })
    write_json(output / "sample_evidence.json", sample_rows)
    results = []
    for state in states:
        result = dict(state)
        path = output / "sdf" / f"{state['state_id']}.sdf"
        try:
            preparation = prepare_smiles(state["smiles"], path, config["seed"])
            molecule = Chem.SDMolSupplier(str(path), removeHs=False)[0]
            checks = validate_coordinates(molecule, state["smiles"])
            # 活性只存于样品表；SDF属性只记录来源关联与条件性假设。
            molecule.SetProp("_Name", state["state_id"])
            for key in ("sample_id", "ionization", "stereo", "interpretation"):
                molecule.SetProp(key, state[key])
            with Chem.SDWriter(str(path)) as writer:
                writer.write(molecule)
            reread = Chem.SDMolSupplier(str(path), removeHs=False)[0]
            validate_coordinates(reread, state["smiles"])
            # 上游函数的旧备注不适用于本入口的显式离子状态枚举。
            preparation["protonation"] = state["ionization"]
            result.update(status="prepared", sdf=str(path.relative_to(output)),
                          preparation=preparation, checks=checks)
        except (ValueError, RuntimeError) as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        results.append(result)
        write_json(output / "state_preparation.json", results)
    manifest.update(
        status="completed", prepared=sum(r["status"] == "prepared" for r in results),
        failed=[r["state_id"] for r in results if r["status"] != "prepared"],
        elapsed_seconds=round(time.perf_counter() - start, 3),
    )
    write_json(output / "manifest.json", manifest)
    return manifest
