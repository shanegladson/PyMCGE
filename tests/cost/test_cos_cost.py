import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.costs import CosineCost

x1: NDArray[np.float64] = np.array([0.0], dtype=np.float64)
x2: NDArray[np.float64] = np.array([np.pi], dtype=np.float64)
params1: NDArray[np.float64] = np.array([1.0, 0.0], dtype=np.float64)
cost: CosineCost = CosineCost(params1)


def test_cos_cost() -> None:
    cost1: np.float64 = cost.eval_cost(x1)
    true_cost1: np.float64 = np.float64(1.0)
    assert_allclose(cost1, true_cost1)

    cost2: np.float64 = cost.eval_cost(x2)
    true_cost2: np.float64 = np.float64(-1.0)
    assert_allclose(cost2, true_cost2)
