import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.costs import QuadraticCost
from src.enums import ControlVariate, DistributionType
from src.gradient.gradient import Gradient
from src.score_gradient import ScoreGradient

x1: NDArray[np.float64] = np.array([1.0], dtype=np.float64)
costparams1: NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
distparams1: NDArray[np.float64] = np.array([1.0, 1.0], dtype=np.float64)
dist: DistributionType = DistributionType.NORMAL
cost: QuadraticCost = QuadraticCost(costparams1)
score_grad: ScoreGradient = ScoreGradient(cost, dist)


def test_score_gradient() -> None:
    score: NDArray[np.float64] = score_grad.eval_integrand(x1, distparams1)
    # True value computed analytically
    true_score: NDArray[np.float64] = np.array([0.0, -0.5 / np.sqrt(2 * np.pi)], dtype=np.float64)
    assert_allclose(score, true_score)


def test_score_gradient_monte_carlo() -> None:
    mc_gradient: Gradient = score_grad.mc_grad_estimate_from_dist(100000, distparams1)
    assert mc_gradient.gradient is not None
    assert mc_gradient.gradient.size == 2
    assert mc_gradient.n_samples == 100000


def test_score_gradient_monte_carlo_cv() -> None:
    cv_gradient: Gradient = score_grad.mc_grad_estimate_from_dist(1000, distparams1, 10.0)
    assert cv_gradient.gradient is not None
    assert cv_gradient.gradient.size == 2

    cv_gradient1: Gradient = score_grad.mc_grad_estimate_from_dist(1000, distparams1, ControlVariate.AVERAGE)
    assert cv_gradient1.gradient is not None
    assert cv_gradient1.gradient.size == 2
