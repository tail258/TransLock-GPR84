"""检查模块步长没有互换、同速优化一致和四格差分方向。"""

import numpy as np
import pandas as pd
import pytest

from translock_screening.training_cross import paired_grid, split_optimizer


def test_group_learning_rates_and_equal_rate_adam_match():
    import copy

    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 3)
            self.head = torch.nn.Linear(3, 1)

        def forward(self, x):
            return self.head(self.encoder(x))

    torch.manual_seed(17)
    model = Toy()
    optimizer = split_optimizer(model, 0.001, 0.0001, 0)
    group_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert group_rates == {"head": 0.001, "encoder": 0.0001}
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        expected = 0.001 if name.startswith("head.") else 0.0001
        torch.testing.assert_close(
            before[name] - parameter, torch.full_like(parameter, expected), rtol=1e-4, atol=1e-8
        )
    first, second = Toy(), None
    second = copy.deepcopy(first)
    original = torch.optim.Adam(first.parameters(), lr=0.0001, weight_decay=0)
    grouped = split_optimizer(second, 0.0001, 0.0001, 0)
    x = torch.tensor([[0.2, 0.7], [0.9, -0.2]])
    for _ in range(5):
        for candidate, opt in [(first, original), (second, grouped)]:
            opt.zero_grad()
            candidate(x).square().mean().backward()
            opt.step()
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="遗漏或重复"):
        split_optimizer(model, 0.001, 0.0001, 0)


def test_cross_grid_signs_and_incomplete_cells():
    names = [
        "head_low_encoder_low",
        "head_high_encoder_low",
        "head_low_encoder_high",
        "head_high_encoder_high",
    ]
    frame = pd.DataFrame(
        [
            {
                "document_id": source,
                "assay_id": source,
                "seed": 7,
                "condition": name,
                "enrichment": value + offset,
                "primary_eligible": True,
            }
            for source, offset in [("A", 0), ("B", 0.2)]
            for name, value in zip(names, [1.1, 1.0, 0.8, 0.9], strict=True)
        ]
    )
    result = paired_grid(frame)
    np.testing.assert_allclose(result.head_effect_encoder_low, -0.1)
    np.testing.assert_allclose(result.head_effect_encoder_high, 0.1)
    np.testing.assert_allclose(result.encoder_effect_head_low, -0.3)
    np.testing.assert_allclose(result.encoder_effect_head_high, -0.1)
    np.testing.assert_allclose(result.interaction, 0.2)
    with pytest.raises(ValueError, match="四格不齐"):
        paired_grid(frame.iloc[:-1])
