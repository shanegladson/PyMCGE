import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.costs import QuadraticCost
from src.enums import DistributionType
from src.gradient.gradient import Gradient
from src.measure_valued_gradient import MeasureValuedGradient


class TestMeasureValuedGradient(unittest.TestCase):
    x1: NDArray[np.float64]
    costparams1: NDArray[np.float64]
    distparams1: NDArray[np.float64]
    dist: DistributionType
    cost: QuadraticCost
    measure_grad: MeasureValuedGradient

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([1.0], dtype=np.float64)
        cls.costparams1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cls.distparams1 = np.array([1.0, 1.0], dtype=np.float64)
        cls.dist = DistributionType.NORMAL
        cls.cost = QuadraticCost(cls.costparams1)
        cls.measure_grad = MeasureValuedGradient(cls.cost, cls.dist)

    def test_measure_valued_gradient_samples(self) -> None:
        pos_samp_mu, neg_samp_mu, pos_samp_sigma_sq, neg_samp_sigma_sq = self.measure_grad._generate_mvg_samples(
            1, self.distparams1
        )

        self.assertIsNotNone(pos_samp_mu)
        self.assertEqual(pos_samp_mu.size, 1)

        self.assertIsNotNone(neg_samp_mu)
        self.assertEqual(neg_samp_mu.size, 1)

        self.assertIsNotNone(pos_samp_sigma_sq)
        self.assertEqual(pos_samp_sigma_sq.size, 1)

        self.assertIsNotNone(neg_samp_sigma_sq)
        self.assertEqual(neg_samp_sigma_sq.size, 1)

    def test_measure_valued_gradient_constant(self) -> None:
        norm_mvg_const = MeasureValuedGradient(self.cost, DistributionType.NORMAL)._get_mvg_constant(
            np.array([1.0, 2.0], dtype=np.float64)
        )
        assert_allclose(norm_mvg_const, np.array([1.0 / np.sqrt(4 * np.pi), 1 / np.sqrt(2.0)]))

        weib_mvg_const = MeasureValuedGradient(self.cost, DistributionType.WEIBULL)._get_mvg_constant(
            np.array([1.0, 2.0], dtype=np.float64)
        )
        assert_allclose(weib_mvg_const[0], np.array([0.5]))

        gamma_mvg_const = MeasureValuedGradient(self.cost, DistributionType.GAMMA)._get_mvg_constant(
            np.array([1.0, 2.0], dtype=np.float64)
        )
        assert_allclose(gamma_mvg_const[0], np.array(0.5))

        pois_mvg_const = MeasureValuedGradient(self.cost, DistributionType.POISSON)._get_mvg_constant(
            np.array([2.0], dtype=np.float64)
        )
        assert_allclose(pois_mvg_const[0], np.array(1.0))

    def test_measure_valued_gradient_mc(self) -> None:
        mc_gradient: Gradient = self.measure_grad.mc_grad_estimate_from_dist(10, self.distparams1)
        self.assertIsNotNone(mc_gradient)
        self.assertEqual(mc_gradient.gradient.size, 2)
        self.assertEqual(mc_gradient.variance.size, 2)
        self.assertEqual(mc_gradient.n_samples, 10)
