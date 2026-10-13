"""小规模固定盒对接的三维准备、日志、断点与排序评价。"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import average_precision_score, roc_auc_score


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def runtime() -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "code_state": "工作树代码，尚无提交基线；未执行哈希校验",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("rdkit", "vina", "meeko", "numpy", "pandas", "scikit-learn")
        },
    }


def gnina_env() -> dict:
    environment = os.environ.copy()
    import sys

    nvidia = Path(sys.prefix) / "lib/python3.11/site-packages/nvidia"
    libraries = [str(path) for path in sorted(nvidia.glob("*/lib"))]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    return environment


def prepare_smiles(smiles: str, output: Path, seed: int) -> dict:
    """只从二维图生成构象，不接受实验坐标或参考构象。"""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("SMILES 无法解析或包含多个片段")
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    parameters.maxIterations = 1000
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise ValueError("ETKDGv3 嵌入失败；不自动更换种子")
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("缺少 MMFF 参数；不静默切换力场")
    result = AllChem.MMFFOptimizeMolecule(molecule, mmffVariant="MMFF94s", maxIters=1000)
    if result != 0:
        raise ValueError(f"MMFF 未收敛：{result}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(output)) as writer:
        writer.write(molecule)
    return {
        "smiles": smiles,
        "seed": seed,
        "method": "ETKDGv3/MMFF94s",
        "mmff_status": result,
        "formal_charge": Chem.GetFormalCharge(molecule),
        "unassigned_stereocenters": len(
            [c for c in Chem.FindMolChiralCenters(molecule, includeUnassigned=True) if c[1] == "?"]
        ),
        "protonation": "沿用标准化结构，不枚举 pH/互变异构体；仅供小样",
    }


def fixed_command(
    gnina: Path, receptor: Path, ligand: Path, output: Path, config: dict
) -> list[str]:
    if config.get("autobox_extend", False):
        raise ValueError("固定盒流程禁止按测试分子自动扩盒")
    if (
        len(config["center"]) != 3
        or len(config["size"]) != 3
        or any(not np.isfinite(value) or value <= 0 for value in config["size"])
    ):
        raise ValueError("固定盒尺寸必须是三个有限正数")
    command = [
        str(gnina),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--out",
        str(output),
        "--autobox_extend",
        "0",
        "--scoring",
        "vina",
        "--cnn_scoring",
        "rescore",
        "--pose_sort_order",
        "CNNscore",
        "--verbosity",
        "2",
    ]
    for name in ("seed", "exhaustiveness", "num_modes", "cpu", "device"):
        command += ["--" + name, str(config[name])]
    for axis, center, size in zip("xyz", config["center"], config["size"], strict=True):
        command += [f"--center_{axis}", str(center), f"--size_{axis}", str(size)]
    return command


def run_command(command: list[str], directory: Path, timeout: int) -> dict:
    """先记 started；已完成/失败任务保留，进程中断后不静默覆盖输出。"""
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "execution.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous["command"] != command:
            raise ValueError("断点命令不匹配，请使用新输出目录")
        return previous
    result = {
        "status": "started",
        "command": command,
        "timeout_seconds": timeout,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(result_path, result)
    start = time.perf_counter()
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=gnina_env(),
                timeout=timeout,
                check=False,
            )
            result.update(
                status="completed" if completed.returncode == 0 else "failed",
                return_code=completed.returncode,
            )
        except subprocess.TimeoutExpired:
            result.update(status="timeout", return_code=None)
    result["run_seconds"] = round(time.perf_counter() - start, 3)
    write_json(result_path, result)
    return result


def read_scores(path: Path) -> list[dict]:
    scores = []
    for rank, molecule in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), 1):
        if molecule is None:
            raise ValueError(f"输出姿势 {rank} 无法解析")
        values = {
            key: float(molecule.GetProp(key))
            for key in ("minimizedAffinity", "CNNscore", "CNNaffinity", "CNN_VS")
            if molecule.HasProp(key)
        }
        if "CNNscore" not in values or not all(np.isfinite(list(values.values()))):
            raise ValueError(f"输出姿势 {rank} 缺失或非有限分数")
        scores.append({"pose_rank": rank, **values})
    if not scores:
        raise ValueError("输出未包含姿势")
    return scores


def ranking_metrics(labels: list[int], scores: list[float], k: int = 5) -> dict:
    """并列分数按层平均占位，top-k 允许分数层的期望检出数。"""
    y, s = np.asarray(labels), np.asarray(scores)
    if len(y) != len(s) or not len(y) or not np.isfinite(s).all():
        raise ValueError("标签/分数长度不匹配或为空/非有限值")
    if not set(y).issubset({0, 1}) or len(np.unique(y)) < 2:
        return {"n": len(y), "not_evaluable": "需要明确的两个实验阈值类别"}
    k = min(k, len(y))
    boundary = np.sort(s)[::-1][k - 1]
    above, tied = s > boundary, s == boundary
    hits = float(y[above].sum() + (k - above.sum()) * y[tied].mean())
    return {
        "n": len(y),
        "positive_count": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "top_k": k,
        "expected_hits_at_k_tie_averaged": hits,
        "recall_at_k": hits / y.sum(),
        "enrichment_at_k": (hits / k) / y.mean(),
    }
