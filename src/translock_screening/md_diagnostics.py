"""膜MD小样的周期边界与受体对齐诊断。"""

import numpy as np


def minimum_image(delta, box_lengths):
    box = np.asarray(box_lengths, dtype=float)
    if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0):
        raise ValueError("盒尺寸必须有限且为正")
    delta = np.asarray(delta, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("坐标差含NaN/无穷")
    return delta - box * np.round(delta / box)


def fit_to_reference(moving, reference):
    moving, reference = np.asarray(moving), np.asarray(reference)
    if moving.shape != reference.shape or moving.shape[1] != 3:
        raise ValueError("对齐坐标维度不匹配")
    a, b = moving.mean(axis=0), reference.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - a).T @ (reference - b))
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    rmsd = np.sqrt(np.mean(np.sum(((moving - a) @ rotation - (reference - b)) ** 2, axis=1)))
    return rotation, a, b, float(rmsd)


def pair_distances(positions, pairs, box_lengths):
    pairs = np.asarray(pairs, dtype=int)
    return np.linalg.norm(
        minimum_image(positions[pairs[:, 0]] - positions[pairs[:, 1]], box_lengths), axis=1
    )
