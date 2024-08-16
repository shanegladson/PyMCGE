import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore

from src.distributions.univariate_uniform import UnivariateUniformDistribution


class TestUnivariateUniform(unittest.TestCase):
    x1: NDArray[np.float64]
    x2: NDArray[np.float64]
    params1: NDArray[np.float64]
    params2: NDArray[np.float64]

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([1.], dtype=np.float64)
        cls.x2 = np.array([-1.], dtype=np.float64)
        cls.params1 = np.array([0., 1.], dtype=np.float64)
        cls.params2 = np.array([1., 10.], dtype=np.float64)

    def test_uniform_density(self) -> None:
        density1: np.float64 = UnivariateUniformDistribution.eval_density(self.x1, self.params1)
        true_density1: np.float64 = np.float64(1.)
        assert_allclose(density1, true_density1)

        density2: np.float64 = UnivariateUniformDistribution.eval_density(self.x1, self.params2)
        true_density2: np.float64 = np.float64(1. / 9.)
        assert_allclose(density2, true_density2)

        density3: np.float64 = UnivariateUniformDistribution.eval_density(self.x2, self.params1)
        true_density3: np.float64 = np.float64(0.)
        assert_allclose(density3, true_density3)

        density4: np.float64 = UnivariateUniformDistribution.eval_density(self.x2, self.params2)
        true_density4: np.float64 = np.float64(0.)
        assert_allclose(density4, true_density4)

    def test_uniform_grad(self) -> None:
        grad1: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(self.x1, self.params1)
        true_grad1: NDArray[np.float64] = np.array([1., -1.], dtype=np.float64)
        assert_allclose(grad1, true_grad1)

        grad2: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(self.x1, self.params2)
        true_grad2: NDArray[np.float64] = np.array([1. / 81., -1. / 81.], dtype=np.float64)
        assert_allclose(grad2, true_grad2)

        grad3: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(self.x2, self.params1)
        true_grad3: NDArray[np.float64] = np.array([0., 0.], dtype=np.float64)
        assert_allclose(grad3, true_grad3)

        grad4: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(self.x2, self.params2)
        true_grad4: NDArray[np.float64] = np.array([0., 0.], dtype=np.float64)
        assert_allclose(grad4, true_grad4)

    def test_uniform_log_grad(self) -> None:
        log_grad1: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(self.x1, self.params1)
        true_log_grad1: NDArray[np.float64] = np.array([1., -1.], dtype=np.float64)
        assert_allclose(log_grad1, true_log_grad1)

        log_grad2: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(self.x1, self.params2)
        true_log_grad2: NDArray[np.float64] = np.array([1. / 9., -1. / 9.], dtype=np.float64)
        assert_allclose(log_grad2, true_log_grad2)

        log_grad3: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(self.x2, self.params1)
        true_log_grad3: NDArray[np.float64] = np.array([np.NAN, np.NAN], dtype=np.float64)
        assert_allclose(log_grad3, true_log_grad3)

        log_grad4: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(self.x2, self.params2)
        true_log_grad4: NDArray[np.float64] = np.array([np.NAN, np.NAN], dtype=np.float64)
        assert_allclose(log_grad4, true_log_grad4)

    def test_uniform_sample(self) -> None:
        sample1: NDArray[np.float64] = UnivariateUniformDistribution.generate_samples([1], self.params1)
        self.assertGreaterEqual(sample1, self.params1[0])
        self.assertLessEqual(sample1, self.params1[1])

        sample2: NDArray[np.float64] = UnivariateUniformDistribution.generate_samples([1], self.params2)
        self.assertGreaterEqual(sample2, self.params2[0])
        self.assertLessEqual(sample2, self.params2[1])


if __name__ == '__main__':
    unittest.main()
