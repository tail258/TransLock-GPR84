"""限定已有短程轨迹的相互作用、波动、聚类和圆周角计算。"""

from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from translock_screening.md_diagnostics import minimum_image


def ligand_features(mol):
    factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    features = factory.GetFeaturesForMol(mol)
    roles = {}
    for family in ("Donor", "Acceptor"):
        roles[family] = sorted(
            {
                i
                for f in features
                if f.GetFamily() == family
                for i in f.GetAtomIds()
                if mol.GetAtomWithIdx(i).GetSymbol() in ("N", "O")
            }
        )
    roles["donor_hydrogens"] = [
        (i, h.GetIdx())
        for i in roles["Donor"]
        for h in mol.GetAtomWithIdx(i).GetNeighbors()
        if h.GetAtomicNum() == 1
    ]
    roles["nonpolar_carbons"] = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and a.GetFormalCharge() == 0
        and all(n.GetAtomicNum() in (1, 6) for n in a.GetNeighbors())
    ]
    roles["charged_atoms"] = [
        (a.GetIdx(), a.GetFormalCharge()) for a in mol.GetAtoms() if a.GetFormalCharge()
    ]
    return roles


def hbond_geometry(xyz, triples, box):
    triples = np.asarray(triples, dtype=int).reshape(-1, 3)
    if not len(triples):
        return np.zeros(0), np.zeros(0)
    d, h, a = (xyz[triples[:, i]] for i in range(3))
    hd = minimum_image(d - h, box)
    ha = minimum_image(a - h, box)
    norms = np.linalg.norm(hd, axis=1) * np.linalg.norm(ha, axis=1)
    if np.any(norms <= 1e-10):
        raise ValueError("氢键角存在重合原子")
    cosine = np.sum(hd * ha, axis=1) / norms
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    distances = np.linalg.norm(minimum_image(d - a, box), axis=1)
    return distances, angles


def rmsf(frames):
    xyz = np.asarray(frames, dtype=float)
    if xyz.ndim != 3 or not np.isfinite(xyz).all() or not len(xyz):
        raise ValueError("RMSF需要已对齐的有限坐标帧")
    return np.sqrt(np.mean(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=2), axis=0))


def pairwise_pose_distances(frames):
    xyz = np.asarray(frames, dtype=float)
    flat = xyz.reshape(len(xyz), -1)
    square = np.sum(flat * flat, axis=1)
    d2 = (square[:, None] + square[None, :] - 2 * flat @ flat.T) / xyz.shape[1]
    distances = np.sqrt(np.maximum(d2, 0))
    np.fill_diagonal(distances, 0)
    return distances


def gromos_clusters(distances, cutoff):
    distances = np.asarray(distances)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1] or cutoff <= 0:
        raise ValueError("聚类距离矩阵或半径无效")
    if not np.isfinite(distances).all() or not np.allclose(distances, distances.T):
        raise ValueError("距离矩阵不对称或非有限")
    available = np.ones(len(distances), dtype=bool)
    labels = np.full(len(distances), -1, dtype=int)
    neighbors = distances <= cutoff
    centers = []
    while available.any():
        counts = neighbors[:, available].sum(axis=1)
        counts[~available] = -1
        center = int(np.argmax(counts))
        members = available & neighbors[center]
        labels[members] = len(centers) + 1
        centers.append(center)
        available[members] = False
    return labels, centers


def torsion_definitions(mol, core):
    definitions = []
    core = set(core)
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        b, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # sp中心的邻接角接近180度，普通四原子二面角会把横向噪声放大。
        if any(
            mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP for i in (b, c)
        ):
            continue
        left = sorted(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(b).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() != c
        )
        right = sorted(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(c).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() != b
        )
        if left and right:
            definitions.append(
                {
                    "bond_index": bond.GetIdx(),
                    "atoms": [left[0], b, c, right[0]],
                    "core_boundary": (b in core) != (c in core),
                }
            )
    return sorted(definitions, key=lambda r: (not r["core_boundary"], r["bond_index"]))[:12]


def dihedrals(frames, quadruplets):
    q = np.asarray(quadruplets, dtype=int).reshape(-1, 4)
    xyz = np.asarray(frames)
    if not len(q):
        return np.zeros((len(xyz), 0))
    p = xyz[:, q]
    b0 = -(p[:, :, 1] - p[:, :, 0])
    b1 = p[:, :, 2] - p[:, :, 1]
    b2 = p[:, :, 3] - p[:, :, 2]
    b1 = b1 / np.linalg.norm(b1, axis=2, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=2, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=2, keepdims=True) * b1
    if np.any(np.linalg.norm(v, axis=2) < 1e-10) or np.any(np.linalg.norm(w, axis=2) < 1e-10):
        raise ValueError("共线原子的二面角未定义")
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, axis=2), np.sum(v * w, axis=2)))


def circular_statistics(degrees):
    values = np.asarray(degrees)
    vector = np.mean(np.exp(1j * np.radians(values)))
    return {
        "circular_mean_deg": float(np.degrees(np.angle(vector))),
        "resultant_length": float(abs(vector)),
        "minus_fraction": float(np.mean(values < -60)),
        "center_fraction": float(np.mean((values >= -60) & (values < 60))),
        "plus_fraction": float(np.mean(values >= 60)),
    }
