"""参数组步长交叉对照与逐来源配对差，不搜索最优格子。"""

import numpy as np
import pandas as pd


def split_optimizer(model, head_lr, encoder_lr, weight_decay):
    import torch

    if not np.isfinite([head_lr, encoder_lr]).all() or min(head_lr, encoder_lr) <= 0:
        raise ValueError("步长必须有限且为正")
    groups = []
    for name, rate in [("encoder", encoder_lr), ("head", head_lr)]:
        parameters = [p for p in getattr(model, name).parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("交叉对照要求两个模块都可训练")
        groups.append({"params": parameters, "lr": rate, "name": name})
    included = [id(p) for group in groups for p in group["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    if len(included) != len(set(included)) or set(included) != set(expected):
        raise ValueError("参数组有遗漏或重复")
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def paired_grid(metrics):
    primary = metrics[metrics.primary_eligible]
    keys = ["document_id", "assay_id", "seed"]
    pivot = primary.pivot(index=keys, columns="condition", values="enrichment")
    cells = [
        "head_low_encoder_low",
        "head_high_encoder_low",
        "head_low_encoder_high",
        "head_high_encoder_high",
    ]
    if not set(cells).issubset(pivot.columns) or pivot[cells].isna().any().any():
        raise ValueError("交叉四格不齐")
    result = pivot.copy()
    result["head_effect_encoder_low"] = pivot[cells[1]] - pivot[cells[0]]
    result["head_effect_encoder_high"] = pivot[cells[3]] - pivot[cells[2]]
    result["encoder_effect_head_low"] = pivot[cells[2]] - pivot[cells[0]]
    result["encoder_effect_head_high"] = pivot[cells[3]] - pivot[cells[1]]
    result["interaction"] = pivot[cells[3]] - pivot[cells[1]] - pivot[cells[2]] + pivot[cells[0]]
    return result.reset_index()


def join_reference(new_metrics, old_metrics):
    reference = old_metrics[old_metrics.method.eq("pretrained_trained")].copy()
    if set(reference.learning_rate.unique()) != {0.0001, 0.001}:
        raise ValueError("旧对角步长不符")
    reference["condition"] = reference.learning_rate.map(
        {
            0.0001: "head_low_encoder_low",
            0.001: "head_high_encoder_high",
        }
    )
    reference["head_lr"] = reference.learning_rate
    reference["encoder_lr"] = reference.learning_rate
    reference["result_origin"] = "previous_factor_run_no_refit"
    added = new_metrics.copy()
    added["result_origin"] = "new_cross_run"
    return pd.concat([reference, added], ignore_index=True)
