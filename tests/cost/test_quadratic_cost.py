import unittest

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose

from src.cost.quadratic_cost import QuadraticCost


class TestQuadraticCost(unittest.TestCase):
    x1: NDArray[np.float64]
    params1: NDArray[np.float64]
    cost: QuadraticCost

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([5.0], dtype=np.float64)
        cls.params1 = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        cls.cost = QuadraticCost(cls.params1)

    def test_quadratic_cost(self) -> None:
        cost: np.float64 = self.cost.eval_cost(self.x1)
        true_cost: np.float64 = np.float64(2.0 * 25.0 + 3.0 * 5.0 + 4.0)
        assert_allclose(cost, true_cost)

    def test_quadratic_cost_grad(self) -> None:
        cost_grad: NDArray[np.float64] = self.cost.eval_grad(self.x1)
        true_cost_grad: NDArray[np.float64] = np.array([2.0 * 2.0 * 5.0 + 3.0], dtype=np.float64)
        assert_allclose(cost_grad, true_cost_grad)


if __name__ == "__main__":
    unittest.main()
