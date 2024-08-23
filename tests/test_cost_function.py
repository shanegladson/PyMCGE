import unittest

import numpy as np

from src._costs._cost_function import _CostFunction


class BlankCostFunction(_CostFunction):
    pass


class TestCostFunction(unittest.TestCase):
    cost: BlankCostFunction

    @classmethod
    def setUpClass(cls) -> None:
        cls.cost = BlankCostFunction(np.zeros(1))

    def test_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.cost.eval_cost(np.zeros(1))

        with self.assertRaises(NotImplementedError):
            self.cost.eval_grad(np.zeros(1))


if __name__ == "__main__":
    unittest.main()
