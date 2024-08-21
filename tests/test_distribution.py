import unittest

import numpy as np

from src._distributions._distribution import _Distribution


class BlankDistribution(_Distribution):
    pass


class TestDistribution(unittest.TestCase):
    def test_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            BlankDistribution.eval_density(np.zeros(1), np.zeros(1))

        with self.assertRaises(NotImplementedError):
            BlankDistribution.eval_grad(np.zeros(1), np.zeros(1))

        with self.assertRaises(NotImplementedError):
            BlankDistribution.eval_grad_log(np.zeros(1), np.zeros(1))


if __name__ == '__main__':
    unittest.main()
