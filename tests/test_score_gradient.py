import unittest

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose

from src.cost.quadratic_cost import QuadraticCost
from src.enums import DistributionType
from src.score_gradient import ScoreGradient
from src.enums import ControlVariate
from src.gradient.gradient import Gradient


class TestScoreGradient(unittest.TestCase):
    x1: NDArray[np.float64]
    costparams1: NDArray[np.float64]
    distparams1: NDArray[np.float64]
    dist: DistributionType
    cost: QuadraticCost
    score_grad: ScoreGradient

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([1.0], dtype=np.float64)
        cls.costparams1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cls.distparams1 = np.array([1.0, 1.0], dtype=np.float64)
        cls.dist = DistributionType.NORMAL
        cls.cost = QuadraticCost(cls.costparams1)
        cls.score_grad = ScoreGradient(cls.cost, cls.dist)

    def test_score_gradient(self) -> None:
        score: NDArray[np.float64] = self.score_grad.eval_integrand(self.x1, self.distparams1)
        # True value computed analytically
        true_score: NDArray[np.float64] = np.array([0.0, -0.5 / np.sqrt(2 * np.pi)], dtype=np.float64)
        assert_allclose(score, true_score)

    def test_score_gradient_monte_carlo(self) -> None:
        mc_gradient: Gradient = self.score_grad.mc_grad_estimate_from_dist(100000, self.distparams1)
        self.assertIsNotNone(mc_gradient.gradient)
        self.assertEqual(mc_gradient.gradient.size, 2)

    def test_score_gradient_monte_carlo_cv(self) -> None:
        cv_gradient: Gradient = self.score_grad.mc_grad_estimate_from_dist(1000, self.distparams1, 10.0)
        self.assertIsNotNone(cv_gradient.gradient)
        self.assertEqual(cv_gradient.gradient.size, 2)

        cv_gradient1: Gradient = self.score_grad.mc_grad_estimate_from_dist(
            1000, self.distparams1, ControlVariate.AVERAGE
        )
        self.assertIsNotNone(cv_gradient1.gradient)
        self.assertEqual(cv_gradient1.gradient.size, 2)
