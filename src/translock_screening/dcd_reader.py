"""有界回读OpenMM写出的无固定原子、带盒DCD；时间按实际分段步长计算。"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _record(handle):
    prefix = handle.read(4)
    if not prefix:
        return None
    if len(prefix) != 4:
        raise ValueError("DCD记录长度被截断")
    size = struct.unpack("<i", prefix)[0]
    if size < 0 or size > 100_000_000:
        raise ValueError("DCD记录长度异常")
    data = handle.read(size)
    tail = handle.read(4)
    if len(data) != size or len(tail) != 4 or struct.unpack("<i", tail)[0] != size:
        raise ValueError("DCD记录被截断或前后长度不一致")
    return data


@dataclass
class DCDFrame:
    step: int
    positions_nm: np.ndarray
    box_vectors_nm: np.ndarray


def read_openmm_dcd(path: Path):
    with path.open("rb") as handle:
        header = _record(handle)
        if header is None or len(header) != 84 or header[:4] != b"CORD":
            raise ValueError("不是支持的OpenMM小端DCD")
        count, first, interval = struct.unpack_from("<3i", header, 4)
        if struct.unpack_from("<i", header, 36)[0] != 0:
            raise ValueError("不支持固定原子DCD")
        if struct.unpack_from("<i", header, 44)[0] != 1:
            raise ValueError("本检查需要周期盒DCD")
        title = _record(handle)
        atom_record = _record(handle)
        if title is None or atom_record is None or len(atom_record) != 4:
            raise ValueError("DCD头被截断")
        atoms = struct.unpack("<i", atom_record)[0]
        if atoms <= 0 or interval <= 0 or count < 0:
            raise ValueError("DCD原子或帧数量异常")
        observed = 0
        while (cell := _record(handle)) is not None:
            if len(cell) != 48:
                raise ValueError("DCD盒记录不支持")
            a, cg, b, cb, ca, c = struct.unpack("<6d", cell)
            if not np.isfinite([a, cg, b, cb, ca, c]).all() or min(a, b, c) <= 0:
                raise ValueError("DCD盒无效")
            sg = np.sqrt(1 - cg * cg)
            if sg < 1e-8 or abs(cb) >= 1 or abs(ca) >= 1:
                raise ValueError("DCD盒角无效")
            cy = (ca - cb * cg) / sg
            cz2 = 1 - cb * cb - cy * cy
            if cz2 <= 0:
                raise ValueError("DCD盒体积非正")
            vectors = (
                np.array([[a, 0, 0], [b * cg, b * sg, 0], [c * cb, c * cy, c * np.sqrt(cz2)]]) / 10
            )
            coords = []
            for _ in range(3):
                data = _record(handle)
                if data is None or len(data) != 4 * atoms:
                    raise ValueError("DCD坐标被截断或原子数不匹配")
                coords.append(np.frombuffer(data, dtype="<f4"))
            xyz = np.stack(coords, axis=1).astype(float) / 10
            if not np.isfinite(xyz).all():
                raise ValueError("DCD坐标含非有限值")
            observed += 1
            # firstStep是第一份报告的步数；与DCDFile写入的NSTEP字段一致。
            yield DCDFrame(first + (observed - 1) * interval, xyz, vectors)
        if observed != count:
            raise ValueError(f"DCD头声明{count}帧，实际{observed}帧")


def time_for_step(step, stages):
    """DCD头只有首段dt，不能用它给变步长轨迹计算时间。"""
    remaining, time_ps = int(step), 0.0
    if remaining < 0:
        raise ValueError("负步数")
    for stage in stages:
        steps = round(stage["time_ps"] * 1000 / stage["timestep_fs"])
        used = min(remaining, steps)
        time_ps += used * stage["timestep_fs"] / 1000
        remaining -= used
        if remaining == 0:
            return time_ps
    raise ValueError("轨迹步数超过固定分段协议")
