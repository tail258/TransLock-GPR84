"""概率饱和/舍入会改变排序并列，稳定决策分数也不保证更好命中。"""

import numpy as np
import pytest
from scipy.special import expit

from translock_screening.benchmark import expected_top_k_hits


def test_sigmoid_saturation_erases_distinct_positive_margins():
    margins = np.array([40.0, 41.0, 42.0])
    probabilities = expit(margins)
    assert np.unique(probabilities).tolist() == [1.0]
    labels = np.array([0, 1, 1])
    assert expected_top_k_hits(labels, probabilities, 1) == pytest.approx(2 / 3)
    assert expected_top_k_hits(labels, margins, 1) == 1


def test_probability_rounding_erases_negative_margins_without_guaranteeing_gain():
    margins = np.array([-60.0, -50.0, -40.0])
    probabilities = expit(margins)
    assert len(np.unique(probabilities)) == 3
    assert np.unique(np.round(probabilities, 12)).tolist() == [0.0]
    labels = np.array([1, 1, 0])
    assert expected_top_k_hits(labels, probabilities, 1) == pytest.approx(2 / 3)
    assert expected_top_k_hits(labels, margins, 1) == 0
