"""混合物来源复核：保留人工证据，比较独立图转录；不调用模型或提升主准入。"""

from __future__ import annotations

import csv
import itertools
import json
import platform
from collections import Counter
from decimal import Decimal
from pathlib import Path

import yaml
from rdkit import Chem, rdBase

from translock_gpr84.data.literature_audit import structure_signature


def mirror_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("无效结构")
    for atom in molecule.GetAtoms():
        atom.InvertChirality()
    return Chem.MolToSmiles(molecule)


def graph_from_review(record: dict) -> str:
    """按表中 R 基拼接连接图，并以正文名称定位 3/3a/9b 相对构型。

    只用名称中的一个代表构型；外消旋比较允许三个核心中心同时反转。
    不为侧链猜构型，也不把生成的代表构型解释为真实受测单一异构体。
    """
    branch = f"({record['r2']})" if record["r2"] else ""
    bridge = {"O": "O", "N": "N", "NC": "N(C)"}[record["x"]]
    base = (
        f"N1{branch}C[CH:3](C(=O)N{record['r1']})"
        f"[CH:30]2C{bridge}c3ccccc3[CH:90]21"
    )
    molecule = Chem.MolFromSmiles(base)
    if molecule is None:
        raise ValueError("图转录不能解析")
    targets = {3: "R", 30: "R" if record["x"] == "O" else "S", 90: "S"}
    indices = {a.GetAtomMapNum(): a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomMapNum()}
    tags = [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
    for choices in itertools.product(tags, repeat=3):
        candidate = Chem.Mol(molecule)
        for key, tag in zip(targets, choices, strict=True):
            candidate.GetAtomWithIdx(indices[key]).SetChiralTag(tag)
        # 原子映射仅用于定位；移除后计算 CIP，避免影响同位素/排序处理。
        for atom in candidate.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
        if all(candidate.GetAtomWithIdx(indices[k]).GetProp("_CIPCode") == v
               for k, v in targets.items()):
            return Chem.MolToSmiles(candidate)
    raise ValueError("无法构造命名指定的核心构型")


def compare_racemate_graphs(csv_smiles: str, table_smiles: str) -> dict:
    left, right = structure_signature(csv_smiles), structure_signature(table_smiles)
    inverted = structure_signature(mirror_smiles(table_smiles))
    return {
        "csv_structure": left,
        "table_structure": right,
        "same_connectivity": left["connectivity"] == right["connectivity"],
        "same_representative_or_mirror": left["inchikey"] in {
            right["inchikey"], inverted["inchikey"]
        },
    }


def select_pending(previous: list[dict], expected: list[int]) -> dict[str, dict]:
    keys = [str(row["compound"]) for row in previous]
    if len(keys) != len(set(keys)):
        raise ValueError("旧审计 compound 重复")
    selected = {
        str(r["compound"]): r for r in previous
        if r["value_agrees"] and r["table_relation"] == "="
        and not r["structure_finding"] and not r["prose_table_conflict"]
    }
    if len(expected) != len(set(expected)) or set(selected) != {str(i) for i in expected}:
        raise ValueError("固定待核集合改变，需明确记录新的范围")
    return selected


def review_record(previous: dict, evidence: dict) -> dict:
    comparison = compare_racemate_graphs(previous["raw_smiles"], graph_from_review(evidence))
    issues = list(evidence.get("issues", []))
    if not comparison["same_connectivity"]:
        issues.append("connectivity_mismatch")
    elif not comparison["same_representative_or_mirror"]:
        issues.append("stereochemical_encoding_mismatch")
    if comparison["table_structure"]["unassigned_2d"]:
        issues.append("unassigned_sidechain_center")
    si = evidence.get("si_pic50")
    if si is not None and Decimal(str(si)) != Decimal(previous["table_raw"]):
        issues.append("main_SI_mean_disagreement")
    return {
        "compound": previous["compound"],
        "source_doi": previous["source_doi"],
        "csv_line": previous["csv_line"],
        "table_number": previous["table_number"],
        "main_pdf_pages": evidence["main_pages"],
        "SI_pdf_pages": evidence["si_pages"],
        "csv_pic50": previous["csv_raw"],
        "main_pic50": previous["table_raw"],
        "SI_pic50": si,
        "strong_at_1uM": Decimal(previous["table_raw"]) >= Decimal("6"),
        **comparison,
        "issues": sorted(set(issues)),
        "sample_type": "racemate_or_more_complex_mixture",
        "provisional_pair_review_candidate": not issues,
        "primary_eligible": False,
        "scoring_authorized_by_this_audit": False,
        "global_limitations": [
            "SI_stereo_text_inconsistent_absolute_pair_not_independently_proven",
            "not_a_full_raw_spectra_reassignment",
            "agonist_challenge_concentration_not_located",
            "labels_already_seen_external_retrospective_only",
        ],
    }


def run(config_path: Path, root: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["predict_or_dock"] or config["project_predictions_viewed"]:
        raise ValueError("此审计仅适用于未计算新分数的阶段")
    previous_path = root / config["previous_audit"]
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    selected = select_pending(previous, config["expected_ids"])
    csv_path = root / config["source_csv"]
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_id = {r["Compound "].strip(): r for r in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("来源 CSV compound 重复")
    entries = config["records"]
    if len(entries) != len(selected) or {str(r["compound"]) for r in entries} != set(selected):
        raise ValueError("证据条目重复或不完整")
    records = []
    for entry in entries:
        key = str(entry["compound"])
        old = selected[key]
        if (by_id[key]["Smiles"] != old["raw_smiles"]
                or by_id[key]["GTPyS (pIC50)"] != old["csv_raw"]):
            raise ValueError(f"来源 {key} 与旧快照改变")
        records.append(review_record(old, config["defaults"] | entry))
    candidates = [r["compound"] for r in records if r["provisional_pair_review_candidate"]]
    summary = {
        "version": config["version"],
        "reviewed": len(records),
        "graph_connectivity_agrees": sum(r["same_connectivity"] for r in records),
        "representative_or_mirror_agrees": sum(
            r["same_representative_or_mirror"] for r in records
        ),
        "provisional_pair_review_candidates": candidates,
        "held": [r["compound"] for r in records if not r["provisional_pair_review_candidate"]],
        "provisional_strong": sum(r["strong_at_1uM"] for r in records
                                  if r["provisional_pair_review_candidate"]),
        "issue_counts": dict(Counter(i for r in records for i in r["issues"])),
        "primary_eligible": 0,
        "scores_computed": False,
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "seed": None,
        "seed_reason": "确定性图与字段核对，无随机计算",
        "code_snapshot": "implementation_snapshot.py; 仓库尚无提交基线",
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size}
            for p in (config_path, previous_path, csv_path)
        ],
    }
    output = (root / config["output"]).resolve()
    if not output.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("输出必须位于 artifacts")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [("review_records.json", records), ("summary.json", summary)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    (output / "review_config.yaml").write_text(config_path.read_text(encoding="utf-8"),
                                              encoding="utf-8")
    (output / "implementation_snapshot.py").write_text(Path(__file__).read_text(encoding="utf-8"),
                                                       encoding="utf-8")
    return summary
