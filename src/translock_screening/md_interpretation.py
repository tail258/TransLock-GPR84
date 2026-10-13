"""已有轨迹的姿势分解与实际代表帧选择，不执行MD积分。"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from translock_screening.md_diagnostics import fit_to_reference


def rmsd(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or not len(a):
        raise ValueError("RMSD需要相同的非空N×3坐标")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("坐标必须有限")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def displacement_metrics(moving, reference, core):
    """输入均为受体对齐后的Å坐标；自身叠合残差另列，不作可加分量。"""
    moving, reference = np.asarray(moving), np.asarray(reference)
    total = rmsd(moving, reference)
    core = np.asarray(core, dtype=int)
    if len(core) < 3 or len(set(core.tolist())) != len(core):
        raise ValueError("核心至少三个不同原子")
    if core.min() < 0 or core.max() >= len(moving):
        raise ValueError("核心原子索引越界")
    shift = moving.mean(axis=0) - reference.mean(axis=0)
    centered = rmsd(moving - moving.mean(axis=0), reference - reference.mean(axis=0))
    _, _, _, internal = fit_to_reference(moving, reference)
    _, _, _, core_internal = fit_to_reference(moving[core], reference[core])
    periphery = np.setdiff1d(np.arange(len(moving)), core)
    translation = float(np.linalg.norm(shift))
    if not np.isclose(total**2, translation**2 + centered**2, atol=1e-10):
        raise ValueError("平移/中心化RMSD平方分解失败")
    return {
        "total_A": total,
        "centroid_shift_A": translation,
        "centered_A": centered,
        "self_fit_internal_A": float(internal),
        "core_A": rmsd(moving[core], reference[core]),
        "core_self_fit_internal_A": float(core_internal),
        "periphery_A": rmsd(moving[periphery], reference[periphery]) if len(periphery) else None,
    }


def murcko_atom_indices(mol):
    """核心只定义原子集合；对称匹配必须给出相同集合，避免选择偏差。"""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    matches = mol.GetSubstructMatches(scaffold, uniquify=False, maxMatches=10000)
    groups = {tuple(sorted(match)) for match in matches}
    if len(groups) != 1:
        raise ValueError("Murcko核心缺失或存在不同原子集合")
    return np.array(next(iter(groups)), dtype=int)


def actual_medoid_index(aligned_frames):
    """全部帧两两RMSD的平均值最小者；不对配体再次做独立叠合。"""
    frames = np.asarray(aligned_frames, dtype=float)
    if frames.ndim != 3 or frames.shape[2] != 3 or not np.isfinite(frames).all():
        raise ValueError("帧坐标格式错误")
    flat = frames.reshape(len(frames), -1)
    square = np.sum(flat**2, axis=1)
    d2 = (square[:, None] + square[None, :] - 2 * flat @ flat.T) / frames.shape[1]
    distance = np.sqrt(np.maximum(d2, 0))
    np.fill_diagonal(distance, 0)
    return int(np.argmin(distance.mean(axis=1)))


def coordinate_only_molecule(molecule, coordinates, name):
    """展示SDF只携带坐标/身份，不继承显式氢参数的整分子属性数组。"""
    result = Chem.Mol(molecule)
    for property_name in list(result.GetPropNames()):
        result.ClearProp(property_name)
    if result.GetNumAtoms() != len(coordinates):
        raise ValueError("坐标原子数不匹配")
    for index, point in enumerate(coordinates):
        result.GetConformer().SetAtomPosition(index, point)
    result.SetProp("_Name", name)
    result.SetProp("purpose", "Visualization heavy-atom coordinates; not force-field parameters")
    return result
