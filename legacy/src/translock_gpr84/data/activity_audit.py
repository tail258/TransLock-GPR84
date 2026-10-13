"""GPR84 活动记录的语境分类、结构标准化和纳入排除审计。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSON Lines 文件。"""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _curated_action_type(activity: dict[str, Any]) -> str | None:
    value = activity.get("action_type")
    if isinstance(value, dict):
        value = value.get("action_type")
    return value.upper() if isinstance(value, str) else None


def classify_mechanism(activity: dict[str, Any]) -> tuple[str, str]:
    """优先使用 ChEMBL 策展作用类型，否则保守解析 assay 描述。"""
    curated = _curated_action_type(activity)
    curated_mapping = {
        "ANTAGONIST": "antagonist",
        "AGONIST": "agonist",
        "INVERSE AGONIST": "inverse_agonist",
        "NEGATIVE ALLOSTERIC MODULATOR": "negative_allosteric_modulator",
        "POSITIVE ALLOSTERIC MODULATOR": "positive_allosteric_modulator",
    }
    if curated:
        return curated_mapping.get(curated, f"other_curated:{curated.lower()}"), (
            "curated_action_type"
        )

    description = (activity.get("assay_description") or "").lower()
    if re.search(r"\bantagoni", description):
        return "antagonist", "assay_description"
    if "inverse agonist" in description:
        return "inverse_agonist", "assay_description"
    if "negative allosteric" in description:
        return "negative_allosteric_modulator", "assay_description"
    if "positive allosteric" in description:
        return "positive_allosteric_modulator", "assay_description"
    if re.search(r"\bagonist", description):
        return "agonist", "assay_description"
    if activity.get("assay_type") == "B" or re.search(r"\bdisplacement\b", description):
        return "binding_unknown_mechanism", "assay_context"
    return "unknown", "unresolved"


def _standardize_smiles(smiles: str | None) -> tuple[str | None, str | None, str]:
    if not smiles:
        return None, None, "missing"
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None, None, "parse_failed"
        molecule = rdMolStandardize.Cleanup(molecule)
        molecule = rdMolStandardize.FragmentParent(molecule)
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        standardized_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(molecule)
        if not inchikey:
            return standardized_smiles, None, "inchikey_failed"
        return standardized_smiles, inchikey, "valid"
    except Exception as error:
        return None, None, f"standardization_failed:{type(error).__name__}"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def audit_records(
    activities: list[dict[str, Any]],
    assays: list[dict[str, Any]],
    molecules: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """生成逐条活动审计表，不删除任何原始记录。"""
    assay_by_id = {record["assay_chembl_id"]: record for record in assays}
    molecule_by_id = {record["molecule_chembl_id"]: record for record in molecules}
    document_by_id = {record["document_chembl_id"]: record for record in documents}
    source_by_id = {record["src_id"]: record for record in sources}
    structure_cache: dict[str | None, tuple[str | None, str | None, str]] = {}
    rows: list[dict[str, Any]] = []

    for activity in activities:
        assay = assay_by_id.get(activity.get("assay_chembl_id"), {})
        parent_id = activity.get("parent_molecule_chembl_id")
        molecule = molecule_by_id.get(parent_id) or molecule_by_id.get(
            activity.get("molecule_chembl_id"), {}
        )
        document = document_by_id.get(activity.get("document_chembl_id"), {})
        source = source_by_id.get(activity.get("src_id"), {})
        molecule_structures = molecule.get("molecule_structures") or {}
        parent_smiles = molecule_structures.get("canonical_smiles") or activity.get(
            "canonical_smiles"
        )
        if parent_smiles not in structure_cache:
            structure_cache[parent_smiles] = _standardize_smiles(parent_smiles)
        standardized_smiles, inchikey, structure_status = structure_cache[parent_smiles]

        mechanism_class, mechanism_evidence = classify_mechanism(activity)
        standard_value = _as_float(activity.get("standard_value"))
        exclusion_reasons: list[str] = []
        if mechanism_class != "antagonist":
            exclusion_reasons.append("not_antagonist")
        if structure_status != "valid":
            exclusion_reasons.append("structure_unavailable")
        if (assay.get("confidence_score") or 0) < config["min_assay_confidence"]:
            exclusion_reasons.append("assay_confidence_below_threshold")
        if activity.get("standard_type") not in config["allowed_endpoint_types"]:
            exclusion_reasons.append("endpoint_not_allowed")
        if activity.get("standard_units") != config["required_units"]:
            exclusion_reasons.append("units_not_allowed")
        if activity.get("standard_relation") not in config["exact_relations"]:
            exclusion_reasons.append("non_exact_or_missing_relation")
        if standard_value is None or standard_value <= 0:
            exclusion_reasons.append("invalid_standard_value")
        if activity.get("standard_flag") != 1:
            exclusion_reasons.append("not_standardized_by_chembl")
        if activity.get("potential_duplicate") == 1:
            exclusion_reasons.append("chembl_potential_duplicate")
        if activity.get("data_validity_comment") is not None:
            exclusion_reasons.append("data_validity_flag")

        include_primary = not exclusion_reasons
        assay_type = activity.get("assay_type") or assay.get("assay_type")
        if include_primary and assay_type == "F" and activity.get("standard_type") == "IC50":
            modeling_stratum = "functional_antagonist_ic50"
        elif include_primary and assay_type == "B":
            modeling_stratum = "binding_antagonist"
        elif include_primary:
            modeling_stratum = "other_antagonist_potency"
        else:
            modeling_stratum = "excluded_from_primary"

        rows.append(
            {
                "activity_id": activity.get("activity_id"),
                "record_id": activity.get("record_id"),
                "src_id": activity.get("src_id"),
                "source_short_name": source.get("src_short_name"),
                "molecule_chembl_id": activity.get("molecule_chembl_id"),
                "parent_molecule_chembl_id": parent_id,
                "assay_chembl_id": activity.get("assay_chembl_id"),
                "document_chembl_id": activity.get("document_chembl_id"),
                "document_year": activity.get("document_year") or document.get("year"),
                "document_doi": document.get("doi"),
                "document_patent_id": document.get("patent_id"),
                "raw_canonical_smiles": activity.get("canonical_smiles"),
                "parent_canonical_smiles": parent_smiles,
                "standardized_smiles": standardized_smiles,
                "standardized_inchikey": inchikey,
                "structure_status": structure_status,
                "assay_type": assay_type,
                "assay_description": activity.get("assay_description")
                or assay.get("description"),
                "assay_confidence_score": assay.get("confidence_score"),
                "action_type_curated": _curated_action_type(activity),
                "mechanism_class": mechanism_class,
                "mechanism_evidence": mechanism_evidence,
                "standard_type": activity.get("standard_type"),
                "standard_relation": activity.get("standard_relation"),
                "standard_value": standard_value,
                "standard_units": activity.get("standard_units"),
                "pchembl_value": _as_float(activity.get("pchembl_value")),
                "standard_flag": activity.get("standard_flag"),
                "potential_duplicate": activity.get("potential_duplicate"),
                "data_validity_comment": activity.get("data_validity_comment"),
                "include_primary": include_primary,
                "exclusion_reasons": ";".join(exclusion_reasons),
                "modeling_stratum": modeling_stratum,
            }
        )
    annotate_assay_value_profiles(
        rows,
        min_records=config.get("binned_assay_min_records", 8),
        max_unique_values=config.get("binned_assay_max_unique_values", 3),
    )
    return rows


def annotate_assay_value_profiles(
    rows: list[dict[str, Any]], *, min_records: int, max_unique_values: int
) -> None:
    """识别低数值多样性的分箱 assay，避免把档位编码用于连续回归。"""
    assay_values: dict[str, list[float]] = {}
    for row in rows:
        if (
            row["include_primary"]
            and row["modeling_stratum"] == "functional_antagonist_ic50"
        ):
            assay_values.setdefault(row["assay_chembl_id"], []).append(row["standard_value"])

    profiles = {
        assay_id: {
            "record_count": len(values),
            "unique_value_count": len(set(values)),
            "binned_like": len(values) >= min_records
            and len(set(values)) <= max_unique_values,
        }
        for assay_id, values in assay_values.items()
    }
    for row in rows:
        profile = profiles.get(row["assay_chembl_id"])
        row["assay_value_record_count"] = profile["record_count"] if profile else None
        row["assay_unique_value_count"] = profile["unique_value_count"] if profile else None
        row["assay_binned_like"] = profile["binned_like"] if profile else False
        row["regression_eligible"] = bool(
            row["include_primary"]
            and row["modeling_stratum"] == "functional_antagonist_ic50"
            and not row["assay_binned_like"]
        )


def _deduplicate_primary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in rows if row["include_primary"]]
    key_fields = (
        "standardized_inchikey",
        "assay_chembl_id",
        "document_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
    )
    retained: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in sorted(primary, key=lambda item: item["activity_id"]):
        key = tuple(row[field] for field in key_fields)
        retained.setdefault(key, row)
    return list(retained.values())


def build_audit_summary(
    rows: list[dict[str, Any]], thresholds: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """汇总审计结果并评估是否足以进入简单基线阶段。"""
    primary_before = [row for row in rows if row["include_primary"]]
    primary = _deduplicate_primary(rows)
    regression = [row for row in primary if row.get("regression_eligible", False)]
    structure_failures = sum(row["structure_status"] != "valid" for row in rows)
    years = sorted(
        {row["document_year"] for row in primary if isinstance(row["document_year"], int)}
    )
    unique_structures = {row["standardized_inchikey"] for row in primary}
    unique_assays = {row["assay_chembl_id"] for row in primary}
    unique_documents = {row["document_chembl_id"] for row in primary}
    regression_years = sorted(
        {
            row["document_year"]
            for row in regression
            if isinstance(row["document_year"], int)
        }
    )
    regression_structures = {row["standardized_inchikey"] for row in regression}
    regression_assays = {row["assay_chembl_id"] for row in regression}
    regression_documents = {row["document_chembl_id"] for row in regression}
    regression_year_span = (
        regression_years[-1] - regression_years[0] if regression_years else 0
    )
    structure_failure_fraction = structure_failures / len(rows) if rows else 1.0

    checks = {
        "unique_structures": len(regression_structures)
        >= thresholds["min_unique_structures"],
        "assays": len(regression_assays) >= thresholds["min_assays"],
        "documents": len(regression_documents) >= thresholds["min_documents"],
        "year_span": regression_year_span >= thresholds["min_year_span"],
        "structure_failure_fraction": structure_failure_fraction
        <= thresholds["max_structure_failure_fraction"],
    }
    exclusion_counter: Counter[str] = Counter()
    for row in rows:
        exclusion_counter.update(filter(None, row["exclusion_reasons"].split(";")))

    summary = {
        "schema_version": 1,
        "input": {
            "activity_records": len(rows),
            "unique_activity_ids": len({row["activity_id"] for row in rows}),
        },
        "mechanism_class_counts": dict(Counter(row["mechanism_class"] for row in rows)),
        "mechanism_evidence_counts": dict(Counter(row["mechanism_evidence"] for row in rows)),
        "endpoint_counts": dict(Counter(row["standard_type"] for row in rows)),
        "exclusion_reason_counts": dict(exclusion_counter),
        "structures": {
            "failure_count": structure_failures,
            "failure_fraction": structure_failure_fraction,
        },
        "primary": {
            "measurements_before_deduplication": len(primary_before),
            "measurements_after_deduplication": len(primary),
            "duplicate_measurements_removed": len(primary_before) - len(primary),
            "unique_structures": len(unique_structures),
            "unique_assays": len(unique_assays),
            "unique_documents": len(unique_documents),
            "years": years,
            "year_span": years[-1] - years[0] if years else 0,
            "modeling_stratum_counts": dict(
                Counter(row["modeling_stratum"] for row in primary)
            ),
        },
        "gate2_regression": {
            "measurements": len(regression),
            "unique_structures": len(regression_structures),
            "unique_assays": len(regression_assays),
            "unique_documents": len(regression_documents),
            "years": regression_years,
            "year_span": regression_year_span,
            "binned_assays_excluded": len(
                {
                    row["assay_chembl_id"]
                    for row in primary
                    if row.get("assay_binned_like", False)
                }
            ),
        },
        "thresholds": thresholds,
        "gate1": {"checks": checks, "passed": all(checks.values())},
        "random_seed": None,
    }
    return summary, primary


def write_audit_outputs(
    *,
    rows: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    summary: dict[str, Any],
    interim_directory: Path,
    processed_path: Path,
    regression_path: Path,
    summary_path: Path,
) -> None:
    """写入可重建的 Parquet 审计表与小型 JSON 摘要。"""
    interim_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(interim_directory / "activity_audit.parquet", index=False)
    pd.DataFrame(primary).to_parquet(
        interim_directory / "primary_measurements.parquet", index=False
    )
    functional_ic50 = [
        row for row in primary if row["modeling_stratum"] == "functional_antagonist_ic50"
    ]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(functional_ic50).to_parquet(processed_path, index=False)
    regression = [row for row in primary if row.get("regression_eligible", False)]
    regression_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(regression).to_parquet(regression_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
