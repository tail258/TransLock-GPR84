"""保留测量证据，并对明确实验终点生成有限的二分类任务。"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd
from rdkit import Chem

UNIT_COLUMNS = [
    "document_id",
    "assay_id",
    "smiles",
    "connection_id",
    "activity_ids",
    "molecule_ids",
    "measurement_count",
    "relations",
    "document_year",
    "document_doi",
    "unit_id",
    "label",
]


def threshold_label(value: float, relation: str, threshold: float) -> int | None:
    """只有整个测量区间都在阈值同侧时才赋类；缺失不作阴性。"""
    if not math.isfinite(value) or value <= 0:
        return None
    relation = {"≤": "<=", "≥": ">="}.get(relation, relation)
    if relation == "=":
        return int(value <= threshold)
    if relation in {"<", "<="} and value <= threshold:
        return 1
    if relation == ">" and value >= threshold:
        return 0
    if relation == ">=" and value > threshold:
        return 0
    return None


def connection_smiles(smiles: str) -> str:
    """用无立体连接图作泄漏分组；不把该图当作已知实验样品身份。"""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("分子结构不能解析")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def admit_measurements(
    audit: pd.DataFrame,
    raw_activity_by_id: dict[int, dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """逐测量准入，不用旧include_primary拒绝所有删失值，不修补原始结构。"""
    accepted = []
    exclusions = []
    counters: Counter[str] = Counter()
    excluded_documents = set(policy["excluded_documents"])
    excluded_molecules = set(policy["excluded_molecule_ids"])
    excluded_connections = {connection_smiles(smiles) for smiles in policy["excluded_alias_smiles"]}
    for row in audit.to_dict("records"):
        reasons = []
        raw = raw_activity_by_id.get(int(row["activity_id"]), {})
        if raw.get("target_chembl_id") != policy["target_id"]:
            reasons.append("wrong_or_missing_target")
        if raw.get("target_organism") != "Homo sapiens":
            reasons.append("not_confirmed_human_target")
        if row["assay_type"] != "F" or row["standard_type"] != "IC50":
            reasons.append("not_functional_ic50")
        if row["standard_units"] != "nM" or row["mechanism_class"] != "antagonist":
            reasons.append("not_nm_antagonist_context")
        confidence = row["assay_confidence_score"]
        if pd.isna(confidence) or float(confidence) < policy["min_confidence"]:
            reasons.append("low_or_missing_confidence")
        if row["standard_flag"] != 1 or row["potential_duplicate"] == 1:
            reasons.append("standardization_or_duplicate_flag")
        if pd.notna(row["data_validity_comment"]):
            reasons.append("validity_flag")
        binned = row["assay_binned_like"]
        if pd.isna(binned) or binned is not False and binned != 0:
            reasons.append("binned_assay_not_reinterpreted")
        if row["document_chembl_id"] in excluded_documents:
            reasons.append("source_identity_audit_restriction")
        if row["molecule_chembl_id"] in excluded_molecules:
            reasons.append("quarantined_molecule")
        if row["structure_status"] != "valid":
            reasons.append("invalid_structure")
        if pd.isna(row["document_chembl_id"]) or pd.isna(row["assay_chembl_id"]):
            reasons.append("missing_source")
        try:
            label = threshold_label(
                float(row["standard_value"]),
                str(row["standard_relation"]),
                policy["threshold_nm"],
            )
        except (TypeError, ValueError):
            label = None
        if label is None:
            reasons.append("interval_not_classifiable")
        if not reasons:
            connection = connection_smiles(row["standardized_smiles"])
            if connection in excluded_connections:
                reasons.append("quarantined_identity_alias")
        if reasons:
            counters.update(reasons)
            exclusions.append({"activity_id": row["activity_id"], "reasons": ";".join(reasons)})
        else:
            accepted.append(
                {
                    **row,
                    "label": label,
                    "connection_id": connection,
                    "sample_identity_status": "database_measurement_not_resolved_author_sample",
                }
            )
    frame = pd.DataFrame(
        accepted, columns=[*audit.columns, "label", "connection_id", "sample_identity_status"]
    )
    summary = {
        "raw_measurements": len(audit),
        "admitted_measurements": len(frame),
        "excluded_measurements": len(exclusions),
        "exclusion_reason_counts_nonexclusive": dict(counters),
        "evidence_level": "database_record_retrospective_not_full_source_certification",
    }
    return frame, pd.DataFrame(exclusions, columns=["activity_id", "reasons"]), summary


def assay_graph_units(
    measurements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同assay同输入图只生成一个计权单位；强弱矛盾单列，保留全部测量ID。"""
    rows, conflicts = [], []
    if measurements.empty:
        return pd.DataFrame(columns=UNIT_COLUMNS), pd.DataFrame(columns=[*UNIT_COLUMNS, "reason"])
    keys = ["document_chembl_id", "assay_chembl_id", "standardized_smiles"]
    for (document, assay, smiles), group in measurements.groupby(keys, sort=True):
        row = {
            "document_id": document,
            "assay_id": assay,
            "smiles": smiles,
            "connection_id": group.iloc[0]["connection_id"],
            "activity_ids": ";".join(str(int(value)) for value in group["activity_id"]),
            "molecule_ids": ";".join(sorted(set(group["molecule_chembl_id"]))),
            "measurement_count": len(group),
            "relations": ";".join(sorted(set(group["standard_relation"]))),
            "document_year": (
                int(group["document_year"].min()) if group["document_year"].notna().any() else None
            ),
            "document_doi": str(group.iloc[0]["document_doi"]),
        }
        labels = sorted(set(group["label"]))
        if len(labels) != 1:
            conflicts.append({**row, "reason": "same_assay_graph_crosses_threshold"})
        else:
            rows.append({**row, "unit_id": f"{document}|{assay}|{smiles}", "label": labels[0]})
    return (
        pd.DataFrame(rows, columns=UNIT_COLUMNS),
        pd.DataFrame(conflicts, columns=[*UNIT_COLUMNS, "reason"]),
    )
