"""核查现有测量是否含跨实验桥接与同体系类似物，不推断因果。"""

from __future__ import annotations

import json
import math
import re
from itertools import combinations

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def exact_mean_sem(text):
    """只接受明确的均值±SEM，不把删失、空值或inactive数值化。"""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)\s*", str(text))
    return tuple(map(float, match.groups())) if match else None


def graph_details(smiles: str) -> dict:
    """保留输入图与未指定四面体手性提示；不保证轴手性身份完整。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效分子图：{smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    return {
        "scaffold": scaffold,
        "unspecified_tetrahedral_centers": sum(
            label == "?" for _, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        ),
    }


def exact_single_value(rows: list[dict]) -> float | None:
    """只有单条精确测量可进入点差值；重复记录不静默平均。"""
    if len(rows) != 1 or rows[0]["standard_relation"] != "=":
        return None
    value = float(rows[0]["standard_value"])
    return value if math.isfinite(value) and value > 0 else None


def build_cause_audit(measurements: pd.DataFrame, units: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """沿用准入记录，只产生桥接候选与结构分组清单。"""
    if measurements["activity_id"].duplicated().any():
        raise ValueError("activity_id重复，不能把重复导入当测量")
    details = {s: graph_details(s) for s in sorted(measurements["standardized_smiles"].unique())}
    bridge_rows = []
    bridge_columns = [
        "smiles",
        "scaffold",
        "unspecified_tetrahedral_centers",
        "assay_left",
        "assay_right",
        "document_left",
        "document_right",
        "doi_left",
        "doi_right",
        "activity_ids_left",
        "activity_ids_right",
        "values_left",
        "values_right",
        "cross_document",
        "log10_ic50_right_over_left",
    ]
    for smiles, group in measurements.groupby("standardized_smiles", sort=True):
        assay_groups = list(group.groupby("assay_chembl_id", sort=True))
        for (left_id, left), (right_id, right) in combinations(assay_groups, 2):
            if (
                left["document_chembl_id"].nunique() != 1
                or right["document_chembl_id"].nunique() != 1
            ):
                raise ValueError("同assay关联多个来源，须另核文献映射")
            left_rows, right_rows = left.to_dict("records"), right.to_dict("records")
            left_value, right_value = exact_single_value(left_rows), exact_single_value(right_rows)
            row = {"smiles": smiles, **details[smiles]}
            for side, assay_id, records in [
                ("left", left_id, left_rows),
                ("right", right_id, right_rows),
            ]:
                row.update(
                    {
                        f"assay_{side}": assay_id,
                        f"document_{side}": records[0]["document_chembl_id"],
                        f"doi_{side}": records[0]["document_doi"],
                        f"activity_ids_{side}": ";".join(str(r["activity_id"]) for r in records),
                        f"values_{side}": json.dumps(
                            [
                                {
                                    "relation": r["standard_relation"],
                                    "value_nm": r["standard_value"],
                                    "molecule_id": r["molecule_chembl_id"],
                                }
                                for r in records
                            ],
                            ensure_ascii=False,
                        ),
                    }
                )
            row["cross_document"] = row["document_left"] != row["document_right"]
            row["log10_ic50_right_over_left"] = (
                math.log10(right_value / left_value)
                if left_value is not None and right_value is not None
                else None
            )
            bridge_rows.append(row)
    bridges = pd.DataFrame(bridge_rows, columns=bridge_columns)
    edge_rows = []
    for (left, right), group in bridges.groupby(["assay_left", "assay_right"], sort=True):
        edge_rows.append(
            {
                "assay_left": left,
                "assay_right": right,
                "shared_graphs": group["smiles"].nunique(),
                "cross_document": bool(group["cross_document"].iloc[0]),
                "single_exact_graph_pairs": int(group["log10_ic50_right_over_left"].notna().sum()),
                "shared_scaffolds": group.loc[group["scaffold"] != "", "scaffold"].nunique(),
            }
        )
    edges = pd.DataFrame(
        edge_rows,
        columns=[
            "assay_left",
            "assay_right",
            "shared_graphs",
            "cross_document",
            "single_exact_graph_pairs",
            "shared_scaffolds",
        ],
    )
    members = units.copy()
    members["scaffold"] = members["smiles"].map(lambda s: details[s]["scaffold"])
    members["unspecified_tetrahedral_centers"] = members["smiles"].map(
        lambda s: details[s]["unspecified_tetrahedral_centers"]
    )
    series_rows = []
    for (assay, scaffold), group in members.groupby(["assay_id", "scaffold"], sort=True):
        series_rows.append(
            {
                "assay_id": assay,
                "document_id": group.iloc[0]["document_id"],
                "document_doi": group.iloc[0]["document_doi"],
                "scaffold": scaffold,
                "graphs": group["smiles"].nunique(),
                "strong_units": int(group["label"].sum()),
                "weak_units": int((group["label"] == 0).sum()),
                "unspecified_stereo_units": int(
                    (group["unspecified_tetrahedral_centers"] > 0).sum()
                ),
                "usable_as_scaffold_group": bool(scaffold),
            }
        )
    return {
        "bridge_candidates": bridges,
        "assay_edges": edges,
        "scaffold_groups": pd.DataFrame(series_rows),
        "scaffold_members": members,
    }
