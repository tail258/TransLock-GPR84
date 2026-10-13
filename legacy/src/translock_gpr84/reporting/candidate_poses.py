"""候选姿势的有限导出；保持坐标，排除明显受体冲突，不推断活性。"""

import numpy as np
from rdkit import Chem

from translock_gpr84.reporting.diagnostic_export import _complex_pdb


def receptor_heavy_coordinates(text: str) -> np.ndarray:
    rows = [
        line
        for line in text.splitlines()
        if line.startswith(("ATOM  ", "HETATM")) and line[76:78].strip().upper() not in ("H", "D")
    ]
    if not rows or any(not line[76:78].strip() for line in rows):
        raise ValueError("受体原子元素缺失，不能猜测重原子")
    return np.array(
        [[float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))] for line in rows]
    )


def minimum_distance(molecule: Chem.Mol, receptor_xyz: np.ndarray) -> float:
    ligand = Chem.RemoveHs(molecule).GetConformer().GetPositions()
    distances = np.linalg.norm(ligand[:, None, :] - receptor_xyz[None, :, :], axis=2)
    if not np.isfinite(distances).all():
        raise ValueError("非有限原子距离")
    return float(distances.min())


def complex_text(receptor: str, molecule: Chem.Mol, identifier: str) -> tuple[str, str]:
    text, chain = _complex_pdb(receptor, molecule)
    # 沿用经过测试的坐标写入，仅替换不适用于本批的历史诊断角色说明。
    body = [line for line in text.splitlines() if not line.startswith("REMARK 900")]
    remarks = [
        f"REMARK 900 ID {identifier}",
        "REMARK 900 COMPUTATIONAL POSE HYPOTHESIS; NOT EXPERIMENTAL COMPLEX",
        "REMARK 900 NO POTENCY OR FUNCTIONAL DIRECTION INFERENCE",
        "REMARK 900 LIGAND SDF IS CHEMICAL IDENTITY SOURCE; ANGSTROM",
    ]
    result = "\n".join(remarks + body) + "\n"
    original = [line for line in receptor.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    written = [line for line in result.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    if written[: len(original)] != original:
        raise ValueError("复合物改变受体原子记录")
    xyz = np.array(
        [
            [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            for line in written[len(original) :]
        ]
    )
    if not np.allclose(xyz, molecule.GetConformer().GetPositions(), atol=0.000501, rtol=0):
        raise ValueError("复合物改变配体坐标")
    return result, chain
