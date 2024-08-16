import unittest

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose

from src.cost.linear_cost import LinearCost
from src.distributions.univariate_normal import UnivariateNormalDistribution
from src.score_gradient import ScoreGradient


class TestScoreGradient(unittest.TestCase):
    x1: NDArray[np.float64]
    costparams1: NDArray[np.float64]
    distparams1: NDArray[np.float64]
    dist: UnivariateNormalDistribution
    cost: LinearCost
    score_grad: ScoreGradient

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([1.], dtype=np.float64)
        cls.costparams1 = np.array([1., 0.], dtype=np.float64)
        cls.distparams1 = np.array([0., 1.], dtype=np.float64)
        cls.dist = UnivariateNormalDistribution()
        cls.cost = LinearCost(cls.costparams1)
        cls.score_grad = ScoreGradient(cls.cost, cls.dist)

    def test_score_gradient(self) -> None:
        score: NDArray[np.float64] = self.score_grad.eval_integrand(self.x1, self.distparams1)
        # True value computed analytically
        true_score: NDArray[np.float64] = np.array([np.exp(-0.5) / np.sqrt(2*np.pi), 0.], dtype=np.float64)
        assert_allclose(score, true_score)
