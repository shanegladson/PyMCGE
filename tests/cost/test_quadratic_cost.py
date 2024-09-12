import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.costs import QuadraticCost

x1: NDArray[np.float64] = np.array([5.0], dtype=np.float64)
params1: NDArray[np.float64] = np.array([2.0, 3.0, 4.0], dtype=np.float64)
cost: QuadraticCost = QuadraticCost(params1)


def test_quadratic_cost() -> None:
    comp_cost: np.float64 = cost.eval_cost(x1)
    true_cost: np.float64 = np.float64(2.0 * 25.0 + 3.0 * 5.0 + 4.0)
    assert_allclose(comp_cost, true_cost)


def test_quadratic_cost_grad() -> None:
    cost_grad: NDArray[np.float64] = cost.eval_grad(x1)
    true_cost_grad: NDArray[np.float64] = np.array([2.0 * 2.0 * 5.0 + 3.0], dtype=np.float64)
    assert_allclose(cost_grad, true_cost_grad)
