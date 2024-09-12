import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.stats import gamma  # type: ignore

from src.distributions import UnivariateGammaDistribution

x1: NDArray[np.float64] = np.array([1.0], dtype=np.float64)
params1: NDArray[np.float64] = np.array([2.0, 1.0], dtype=np.float64)
params2: NDArray[np.float64] = np.array([1.0, 2.0], dtype=np.float64)


def test_gamma_density() -> None:
    density1: np.float64 = UnivariateGammaDistribution.eval_density(x1, params1)
    true_density1 = gamma.pdf(x1, params1[0], scale=1.0 / params1[1])
    assert_allclose(density1, true_density1)

    density2: np.float64 = UnivariateGammaDistribution.eval_density(x1, params2)
    true_density2 = gamma.pdf(x1, params2[0], scale=1.0 / params2[1])
    assert_allclose(density2, true_density2)


def test_gamma_parameters() -> None:
    assert_allclose(params1, [2.0, 1.0])
    assert_allclose(params2, [1.0, 2.0])
