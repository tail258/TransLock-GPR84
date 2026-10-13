"""重对接姿势的对称校正 RMSD 与残基接触恢复评估。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import rdMolAlign


class AtomMapping(dict[int, int]):
    """Meeko SMILES 序号到 PDBQT 序号；携带 SMILES，避免误用参考 SDF 的原子序。"""

    smiles: str = ""


def read_atom_mapping(path: Path) -> AtomMapping:
    """读取 Meeko 的 SMILES 分子图及 1-based 原子映射。"""
    mapping = AtomMapping()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("REMARK SMILES ") and not line.startswith("REMARK SMILES IDX"):
            mapping.smiles = line.removeprefix("REMARK SMILES ")
        if not line.startswith("REMARK SMILES IDX"):
            continue
        values = [int(value) for value in line.split()[3:]]
        mapping.update(dict(zip(values[::2], values[1::2], strict=True)))
    if not mapping or not mapping.smiles:
        raise RuntimeError("PDBQT 中缺少 Meeko 原子映射")
    return mapping


def read_pose_coordinates(path: Path) -> list[np.ndarray]:
    """读取多模型 PDBQT 中每个姿势的重原子坐标。"""
    poses: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith(("ATOM  ", "HETATM")):
            current.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        elif line.startswith("ENDMDL"):
            poses.append(current)
    return [np.asarray(pose, dtype=float) for pose in poses]


def coordinates_in_reference_order(
    mapping: dict[int, int], pose_coordinates: np.ndarray, atom_count: int
) -> np.ndarray:
    """将 PDBQT 坐标恢复为 Meeko SMILES 序（历史函数名保留，不是参考 SDF 序）。"""
    if set(mapping) != set(range(1, atom_count + 1)) or any(
        serial < 1 or serial > len(pose_coordinates) for serial in mapping.values()
    ):
        raise RuntimeError("对接姿势缺失重原子映射")
    reordered = np.empty((atom_count, 3), dtype=float)
    for original_index, pdbqt_index in mapping.items():
        reordered[original_index - 1] = pose_coordinates[pdbqt_index - 1]
    return reordered


def calculate_rmsds(
    reference_sdf: Path,
    mapping: AtomMapping,
    pose_coordinates: list[np.ndarray],
) -> list[float]:
    """计算所有姿势相对参考构象的对称校正重原子 RMSD。"""
    reference_with_h = Chem.SDMolSupplier(str(reference_sdf), removeHs=False)[0]
    if reference_with_h is None:
        raise RuntimeError("无法读取参考配体")
    reference = Chem.RemoveHs(reference_with_h)
    if not isinstance(mapping, AtomMapping) or not mapping.smiles:
        raise ValueError("RMSD 需要 Meeko SMILES 分子图，不能只用原子序号字典")
    template = Chem.MolFromSmiles(mapping.smiles)
    if template is None or template.GetNumAtoms() != reference.GetNumAtoms():
        raise ValueError("Meeko SMILES 与参考分子重原子数不匹配")
    rmsds: list[float] = []
    for coordinates in pose_coordinates:
        reordered = coordinates_in_reference_order(
            mapping, coordinates, reference.GetNumAtoms()
        )
        pose = Chem.Mol(template)
        conformer = Chem.Conformer(pose.GetNumAtoms())
        for atom_index, xyz in enumerate(reordered):
            conformer.SetAtomPosition(atom_index, xyz)
        pose.RemoveAllConformers()
        pose.AddConformer(conformer, assignId=True)
        # 对接姿势处在固定受体坐标系中，只允许原子对称映射，不能再做刚体叠合。
        rmsds.append(float(rdMolAlign.CalcRMS(pose, reference)))
    return rmsds


def calculate_sdf_pose_rmsds(reference_sdf: Path, poses_sdf: Path) -> list[float]:
    """计算 SDF 多姿势相对单个参考分子的对称校正重原子 RMSD。"""
    reference_with_h = Chem.SDMolSupplier(str(reference_sdf), removeHs=False)[0]
    if reference_with_h is None:
        raise RuntimeError("无法读取参考配体")
    reference = Chem.RemoveHs(reference_with_h)
    rmsds: list[float] = []
    for pose_with_h in Chem.SDMolSupplier(str(poses_sdf), removeHs=False):
        if pose_with_h is None:
            continue
        pose = Chem.RemoveHs(pose_with_h)
        if pose.GetNumAtoms() != reference.GetNumAtoms():
            raise RuntimeError("SDF 姿势的重原子数与参考配体不一致")
        # CalcRMS 会枚举对称映射，但不会把预测姿势旋转/平移到参考姿势。
        rmsds.append(float(rdMolAlign.CalcRMS(pose, reference)))
    if not rmsds:
        raise RuntimeError("SDF 中没有可读取的姿势")
    return rmsds


def read_reference_coordinates(reference_sdf: Path) -> np.ndarray:
    """读取参考 SDF 的重原子坐标。"""
    molecule_with_h = Chem.SDMolSupplier(str(reference_sdf), removeHs=False)[0]
    if molecule_with_h is None:
        raise RuntimeError("无法读取参考配体")
    molecule = Chem.RemoveHs(molecule_with_h)
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=float)


def read_receptor_heavy_atoms(receptor_pdb: Path) -> tuple[np.ndarray, list[str]]:
    """读取受体重原子坐标及可追溯残基标签。"""
    structure = PDBParser(QUIET=True).get_structure("receptor", str(receptor_pdb))
    coordinates: list[np.ndarray] = []
    residue_labels: list[str] = []
    for atom in structure.get_atoms():
        if atom.element.strip().upper() == "H":
            continue
        residue = atom.get_parent()
        chain = residue.get_parent()
        coordinates.append(np.asarray(atom.coord, dtype=float))
        residue_labels.append(f"{chain.id}:{residue.resname}{residue.id[1]}")
    return np.asarray(coordinates), residue_labels


def residue_contacts(
    receptor_coordinates: np.ndarray,
    residue_labels: list[str],
    ligand_coordinates: np.ndarray,
    threshold_angstrom: float = 4.0,
) -> dict[str, float]:
    """返回阈值内的残基及其与配体的最短重原子距离。"""
    if len(receptor_coordinates) != len(residue_labels):
        raise ValueError("受体坐标数与残基标签数不一致")
    distances = np.linalg.norm(
        receptor_coordinates[:, None, :] - ligand_coordinates[None, :, :], axis=2
    )
    atom_minima = distances.min(axis=1)
    contacts: dict[str, float] = {}
    for residue_label, distance in zip(residue_labels, atom_minima, strict=True):
        if distance <= threshold_angstrom:
            contacts[residue_label] = min(contacts.get(residue_label, float("inf")), distance)
    return contacts


def contact_recovery(
    reference_contacts: set[str], predicted_contacts: set[str]
) -> dict[str, float | int]:
    """计算残基接触的精确率、召回率与 Jaccard。"""
    intersection = reference_contacts & predicted_contacts
    union = reference_contacts | predicted_contacts
    return {
        "reference_count": len(reference_contacts),
        "predicted_count": len(predicted_contacts),
        "recovered_count": len(intersection),
        "precision": len(intersection) / len(predicted_contacts) if predicted_contacts else 0.0,
        "recall": len(intersection) / len(reference_contacts) if reference_contacts else 0.0,
        "jaccard": len(intersection) / len(union) if union else 1.0,
    }
