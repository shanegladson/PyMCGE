import unittest

import numpy as np

from src.cost_function import CostFunction


class BlankCostFunction(CostFunction):
    pass


class TestCostFunction(unittest.TestCase):
    def test_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            BlankCostFunction.eval_cost(np.zeros(1), np.zeros(1))

        with self.assertRaises(NotImplementedError):
            BlankCostFunction.eval_grad(np.zeros(1), np.zeros(1))


if __name__ == '__main__':
    unittest.main()