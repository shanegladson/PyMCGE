import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.stats import gamma  # type: ignore

from src.distributions import UnivariateGammaDistribution


class TestUnivariateGamma(unittest.TestCase):
    x1: NDArray[np.float64]
    params1: NDArray[np.float64]
    params2: NDArray[np.float64]

    @classmethod
    def setUpClass(cls) -> None:
        cls.x1 = np.array([1.0], dtype=np.float64)
        cls.params1 = np.array([2.0, 1.0], dtype=np.float64)
        cls.params2 = np.array([1.0, 2.0], dtype=np.float64)

    def test_gamma_density(self) -> None:
        density1: np.float64 = UnivariateGammaDistribution.eval_density(self.x1, self.params1)
        true_density1 = gamma.pdf(self.x1, self.params1[0], scale=1.0 / self.params1[1])
        assert_allclose(density1, true_density1)

        density2: np.float64 = UnivariateGammaDistribution.eval_density(self.x1, self.params2)
        true_density2 = gamma.pdf(self.x1, self.params2[0], scale=1.0 / self.params2[1])
        assert_allclose(density2, true_density2)

    def test_gamma_parameters(self) -> None:
        assert_allclose(self.params1, [2.0, 1.0])
        assert_allclose(self.params2, [1.0, 2.0])
