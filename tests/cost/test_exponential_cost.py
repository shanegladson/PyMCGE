import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.cost.exponential_cost import ExponentialCost


class TestExponentialCost(unittest.TestCase):
    x1: NDArray[np.float64]
    params1: NDArray[np.float64]
    cost: ExponentialCost

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([1.0], dtype=np.float64)
        cls.params1 = np.array([3.0], dtype=np.float64)
        cls.cost = ExponentialCost(cls.params1)

    def test_exponential_cost(self) -> None:
        cost: np.float64 = self.cost.eval_cost(self.x1)
        true_cost: np.float64 = np.float64(np.exp(3.0))
        assert_allclose(cost, true_cost)

    def test_exponential_cost_grad(self) -> None:
        cost_grad: NDArray[np.float64] = self.cost.eval_grad(self.x1)
        true_cost_grad: NDArray[np.float64] = np.array([3.0 * np.exp(3.0)], dtype=np.float64)
        assert_allclose(cost_grad, true_cost_grad)


if __name__ == "__main__":
    unittest.main()
