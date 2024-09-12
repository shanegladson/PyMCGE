import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.costs import QuadraticCost
from src.enums import DistributionType
from src.gradient.gradient import Gradient
from src.measure_valued_gradient import MeasureValuedGradient

x1: NDArray[np.float64] = np.array([1.0], dtype=np.float64)
costparams1: NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
distparams1: NDArray[np.float64] = np.array([1.0, 1.0], dtype=np.float64)
dist: DistributionType = DistributionType.NORMAL
cost: QuadraticCost = QuadraticCost(costparams1)
measure_grad: MeasureValuedGradient = MeasureValuedGradient(cost, dist)


def test_measure_valued_gradient_samples() -> None:
    pos_samp_mu, neg_samp_mu, pos_samp_sigma_sq, neg_samp_sigma_sq = measure_grad._generate_mvg_samples(1, distparams1)

    assert pos_samp_mu is not None
    assert pos_samp_mu.size == 1

    assert neg_samp_mu is not None
    assert neg_samp_mu.size == 1

    assert pos_samp_sigma_sq is not None
    assert pos_samp_sigma_sq.size == 1

    assert neg_samp_sigma_sq is not None
    assert neg_samp_sigma_sq.size == 1


def test_measure_valued_gradient_constant() -> None:
    norm_mvg_const = MeasureValuedGradient(cost, DistributionType.NORMAL)._get_mvg_constant(
        np.array([1.0, 2.0], dtype=np.float64)
    )
    assert_allclose(norm_mvg_const, np.array([1.0 / np.sqrt(4 * np.pi), 1 / np.sqrt(2.0)]))

    weib_mvg_const = MeasureValuedGradient(cost, DistributionType.WEIBULL)._get_mvg_constant(
        np.array([1.0, 2.0], dtype=np.float64)
    )
    assert_allclose(weib_mvg_const[0], np.array([0.5]))

    gamma_mvg_const = MeasureValuedGradient(cost, DistributionType.GAMMA)._get_mvg_constant(
        np.array([1.0, 2.0], dtype=np.float64)
    )
    assert_allclose(gamma_mvg_const[0], np.array(0.5))

    pois_mvg_const = MeasureValuedGradient(cost, DistributionType.POISSON)._get_mvg_constant(
        np.array([2.0], dtype=np.float64)
    )
    assert_allclose(pois_mvg_const[0], np.array(1.0))


def test_measure_valued_gradient_mc() -> None:
    mc_gradient: Gradient = measure_grad.mc_grad_estimate_from_dist(10, distparams1)
    assert mc_gradient is not None
    assert mc_gradient.gradient.size == 2
    assert mc_gradient.variance.size == 2
    assert mc_gradient.n_samples == 10
