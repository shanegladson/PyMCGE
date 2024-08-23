import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.cost.cos_cost import CosineCost


class TestCosineCost(unittest.TestCase):
    x1: NDArray[np.float64]
    x2: NDArray[np.float64]
    params1: NDArray[np.float64]
    cost: CosineCost

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([0.0], dtype=np.float64)
        cls.x2 = np.array([np.pi], dtype=np.float64)
        cls.params1 = np.array([1.0, 0.0], dtype=np.float64)
        cls.cost = CosineCost(cls.params1)

    def test_cos_cost(self) -> None:
        cost1: np.float64 = self.cost.eval_cost(self.x1)
        true_cost1: np.float64 = np.float64(1.0)
        assert_allclose(cost1, true_cost1)

        cost2: np.float64 = self.cost.eval_cost(self.x2)
        true_cost2: np.float64 = np.float64(-1.0)
        assert_allclose(cost2, true_cost2)
