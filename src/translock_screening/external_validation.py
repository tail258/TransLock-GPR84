"""外部专利数据的保守身份审计和固定预算评价。"""

import math

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import average_precision_score, roc_auc_score

from .benchmark import expected_top_k_hits
from .screening_application import connection, normalize


def patent_label(bin_name):
    # 本专利Table10边界，不能推广到别的专利字母分箱。
    return {"A": 1, "B": 1, "C": 0}[bin_name]


def identity_keys(smiles):
    mol = normalize(smiles)
    stereo = Chem.MolToSmiles(mol)
    conn = connection(mol)
    broad = Chem.Mol(mol)
    Chem.RemoveStereochemistry(broad)
    for atom in broad.GetAtoms():
        atom.SetIsotope(0)
    broad = Chem.RemoveHs(broad)
    isotope_free = Chem.MolToSmiles(broad)
    tautomer = rdMolStandardize.TautomerEnumerator().Canonicalize(broad)
    return {
        "smiles": stereo,
        "connection": conn,
        "isotope_free": isotope_free,
        "broad_identity": Chem.MolToSmiles(tautomer),
    }


def fixed_metrics(labels, scores):
    y, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(y) or not set(y).issubset({0, 1}) or not np.isfinite(scores).all():
        raise ValueError("外部评价输入无效")
    n, positives = len(y), int(y.sum())
    budget = math.ceil(n * 0.2)
    hits = expected_top_k_hits(y, scores, budget)
    random_hits = budget * positives / n
    return {
        "n": n,
        "strong": positives,
        "weak": n - positives,
        "budget": budget,
        "hits": hits,
        "weak_selected": budget - hits,
        "random_hits": random_hits,
        "precision": hits / budget,
        "ef": hits / random_hits if positives else None,
        "ef_ceiling": min(budget, positives) / random_hits if positives else None,
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)) if positives else None,
    }


def collapse_connections(frame):
    """固定数字编号最小者代表同类连接组；矛盾组整体排除。"""
    ordered = frame.assign(_number=frame.sample_id.str.split("-").str[-1].astype(int))
    conflicts = ordered.groupby("isotope_free").label.nunique()
    good = set(conflicts[conflicts.eq(1)].index)
    return (
        ordered[ordered.isotope_free.isin(good)]
        .sort_values("_number")
        .drop_duplicates("isotope_free")
        .drop(columns="_number")
    )
