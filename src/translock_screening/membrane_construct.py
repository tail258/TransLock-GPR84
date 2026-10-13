"""膜体系的构建体映射与刚体取向；保留合成连接肽标识。"""

import numpy as np


def alanine_linker_map(native_ids, replace_start, replace_end, linker_length):
    if linker_length < 1 or replace_start >= replace_end:
        raise ValueError("连接肽范围或长度无效")
    native_ids = sorted(native_ids)
    left = [n for n in native_ids if n < replace_start]
    right = [n for n in native_ids if n > replace_end]
    if not left or not right or left[-1] != replace_start - 1 or right[0] != replace_end + 1:
        raise ValueError("连接端点缺少实验坐标")
    if len(set(native_ids)) != len(native_ids):
        raise ValueError("原生编号重复")
    for part in (left, right):
        if part != list(range(part[0], part[-1] + 1)):
            raise ValueError("保留片段另有缺口")
    records = [{"native_id": n, "role": "retained_native"} for n in left]
    records += [{"native_id": None, "role": "synthetic_ALA_linker"}] * linker_length
    records += [{"native_id": n, "role": "retained_native"} for n in right]
    return [dict(r, model_id=i + 1) for i, r in enumerate(records)]


def rotation_to_z(axis):
    axis = np.asarray(axis, dtype=float)
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-10:
        raise ValueError("取向轴无效")
    axis /= np.linalg.norm(axis)
    target = np.array([0.0, 0.0, 1.0])
    cosine = float(np.dot(axis, target))
    if cosine > 1 - 1e-12:
        return np.eye(3)
    if cosine < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    cross = np.cross(axis, target)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def l_alanine_cb(n, ca, c):
    """按N/CA/C重建L-ALA的CB；防止软核补环出现镜像或平面中心。"""
    u, v = np.asarray(n) - ca, np.asarray(c) - ca
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    normal = np.cross(u, v)
    if np.linalg.norm(normal) < 0.1:
        raise ValueError("连接肽CA周围骨架近共线")
    normal /= np.linalg.norm(normal)
    planar = -0.55 * (u + v)
    d = planar + np.sqrt(1 - np.dot(planar, planar)) * normal
    return np.asarray(ca) + 1.53 * d


def alpha_hydrogen_from_heavy(n, ca, c, cb):
    """从三个重原子方向放置第四个四面体取代基；长度单位为Å。"""
    directions = np.asarray([n, c, cb], dtype=float) - np.asarray(ca)
    lengths = np.linalg.norm(directions, axis=1)
    if not np.isfinite(directions).all() or np.any(lengths < 0.1):
        raise ValueError("CA取代基坐标无效")
    direction = -(directions / lengths[:, None]).sum(axis=0)
    if np.linalg.norm(direction) < 0.1:
        raise ValueError("CA周围接近平面，无法唯一放置氢")
    direction /= np.linalg.norm(direction)
    if np.dot(np.cross(directions[0], directions[1]), direction) >= 0:
        raise ValueError("重原子不是可用于L-氨基酸的四面体")
    return np.asarray(ca) + 1.09 * direction


def chiral_minimization_barrier(atom_quartets):
    """补环最小化时保护L-氨基酸CA立体；该力不进入膜MD系统。"""
    import openmm

    expression = (
        "0.5*kc*min(0,V-0.002)^2;"
        "V=(x1-x2)*((y3-y2)*(z4-z2)-(z3-z2)*(y4-y2))"
        "-(y1-y2)*((x3-x2)*(z4-z2)-(z3-z2)*(x4-x2))"
        "+(z1-z2)*((x3-x2)*(y4-y2)-(y3-y2)*(x4-x2))"
    )
    force = openmm.CustomCompoundBondForce(4, expression)
    force.addGlobalParameter("kc", 1e9)
    for quartet in atom_quartets:
        force.addBond(list(quartet), [])
    return force


def tm_geometry_transform(ca_by_native, tm_ranges):
    """以七段跨膜端点作几何取向近似：奇数段C→N朝胞外。"""
    directions, centers, helices = [], [], []
    for i, (start, end) in enumerate(tm_ranges):
        xyz = np.array([ca_by_native[n] for n in range(start, end + 1)])
        outward = (xyz[:3].mean(axis=0) - xyz[-3:].mean(axis=0)) * (-1 if i % 2 else 1)
        directions.append(outward / np.linalg.norm(outward))
        centers.append(xyz.mean(axis=0))
        helices.append(xyz)
    axis = np.mean(directions, axis=0)
    rotation = rotation_to_z(axis)
    origin = np.mean(centers, axis=0)
    summaries = []
    for bounds, xyz in zip(tm_ranges, helices, strict=True):
        moved = (xyz - origin) @ rotation.T
        summaries.append(
            {
                "native_range": bounds,
                "z_range_A": [float(moved[:, 2].min()), float(moved[:, 2].max())],
            }
        )
    return rotation, origin, summaries
