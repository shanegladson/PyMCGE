import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore

from src.distributions import UnivariateNormalDistribution


class TestUnivariateNormal(unittest.TestCase):
    x1: NDArray[np.float64]
    params1: NDArray[np.float64]
    params2: NDArray[np.float64]

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([0.0], dtype=np.float64)
        cls.params1 = np.array([0.0, 1.0], dtype=np.float64)
        cls.params2 = np.array([1.0, 2.0], dtype=np.float64)

    def test_normal_density(self) -> None:
        density1: np.float64 = UnivariateNormalDistribution.eval_density(self.x1, self.params1)
        true_density1 = norm.pdf(self.x1, loc=self.params1[0], scale=np.sqrt(self.params1[1]))
        assert_allclose(density1, true_density1)

        density2: np.float64 = UnivariateNormalDistribution.eval_density(self.x1, self.params2)
        true_density2 = norm.pdf(self.x1, loc=self.params2[0], scale=np.sqrt(self.params2[1]))
        assert_allclose(density2, true_density2)

    def test_normal_log_grad(self) -> None:
        log_grad1: NDArray[np.float64] = UnivariateNormalDistribution.eval_grad_log(self.x1, self.params1)
        # This is computed analytically
        true_log_grad1: NDArray[np.float64] = np.array([0.0, -0.5], dtype=np.float64)
        assert_allclose(log_grad1, true_log_grad1)

        log_grad2: NDArray[np.float64] = UnivariateNormalDistribution.eval_grad_log(self.x1, self.params2)
        # This is computed analytically
        true_log_grad2: NDArray[np.float64] = np.array([-0.5, -0.125], dtype=np.float64)
        assert_allclose(log_grad2, true_log_grad2)

    def test_normal_sample(self) -> None:
        sample1: NDArray[np.float64] = UnivariateNormalDistribution.generate_samples([1], self.params1)
        self.assertIsNotNone(sample1)

        sample2: NDArray[np.float64] = UnivariateNormalDistribution.generate_samples([1], self.params2)
        self.assertIsNotNone(sample2)


if __name__ == "__main__":
    unittest.main()
