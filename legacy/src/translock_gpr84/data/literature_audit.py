"""公开补充材料与数据库的结构、逐 assay 数值核对；不改写历史标签。"""

from __future__ import annotations

import math
import re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from translock_gpr84.data.activity_audit import _standardize_smiles


def structure_signature(smiles: str) -> dict:
    canonical, key, status = _standardize_smiles(smiles.strip())
    if status != "valid":
        raise ValueError(f"无效来源结构：{status}")
    molecule = Chem.MolFromSmiles(canonical)
    centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    return {
        "smiles": canonical,
        "inchikey": key,
        "connectivity": key.split("-")[0],
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "centers_2d": [list(item) for item in centers],
        "unassigned_2d": sum(label == "?" for _, label in centers),
    }


def compare_structures(first: str, second: str) -> dict:
    left, right = structure_signature(first), structure_signature(second)
    return {
        "same_full_key": left["inchikey"] == right["inchikey"],
        "same_connectivity": left["connectivity"] == right["connectivity"],
        "pilot": left,
        "source": right,
    }


def parse_pic50(value: str) -> dict:
    """保留不等号、无活性文本和空值，绝不把它们转为精确测量。"""
    value = value.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)", value)
    if not match:
        return {"text": value, "kind": "not_exact", "ic50_nm": None}
    mean, sem = map(float, match.groups())
    return {
        "text": value,
        "kind": "exact_mean_with_sem",
        "pic50": mean,
        "sem": sem,
        "ic50_nm": 10 ** (9 - mean),
        "threshold_within_one_sem": mean - sem <= 6 <= mean + sem,
        "note": "SEM 不是个体测量范围，也不是 95% 置信区间",
    }


def compare_value(source: str, database_nm: float) -> dict:
    parsed = parse_pic50(source)
    if parsed["ic50_nm"] is None:
        return {"source": parsed, "status": "source_not_exact"}
    expected = parsed["ic50_nm"]
    if math.isclose(database_nm, expected, abs_tol=0.0051, rel_tol=0):
        status = "matches_2_decimal_nm"
    elif math.isclose(database_nm, expected, abs_tol=0.5001, rel_tol=0):
        status = "compatible_integer_rounding_not_confirmed_duplicate"
    else:
        status = "numeric_disagreement"
    return {
        "source": parsed,
        "database_nm": database_nm,
        "absolute_error_nm": abs(database_nm - expected),
        "status": status,
    }
