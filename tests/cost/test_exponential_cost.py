import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.costs import ExponentialCost

x1: NDArray[np.float64] = np.array([1.0], dtype=np.float64)
params1: NDArray[np.float64] = np.array([3.0], dtype=np.float64)
cost: ExponentialCost = ExponentialCost(params1)


def test_exponential_cost() -> None:
    comp_cost: np.float64 = cost.eval_cost(x1)
    true_cost: np.float64 = np.float64(np.exp(3.0))
    assert_allclose(comp_cost, true_cost)


def test_exponential_cost_grad() -> None:
    cost_grad: NDArray[np.float64] = cost.eval_grad(x1)
    true_cost_grad: NDArray[np.float64] = np.array([3.0 * np.exp(3.0)], dtype=np.float64)
    assert_allclose(cost_grad, true_cost_grad)
