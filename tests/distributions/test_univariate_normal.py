import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore

from src.distributions import UnivariateNormalDistribution

x1: NDArray[np.float64] = np.array([0.0], dtype=np.float64)
params1: NDArray[np.float64] = np.array([0.0, 1.0], dtype=np.float64)
params2: NDArray[np.float64] = np.array([1.0, 2.0], dtype=np.float64)


def test_normal_density() -> None:
    density1: np.float64 = UnivariateNormalDistribution.eval_density(x1, params1)
    true_density1 = norm.pdf(x1, loc=params1[0], scale=np.sqrt(params1[1]))
    assert_allclose(density1, true_density1)

    density2: np.float64 = UnivariateNormalDistribution.eval_density(x1, params2)
    true_density2 = norm.pdf(x1, loc=params2[0], scale=np.sqrt(params2[1]))
    assert_allclose(density2, true_density2)


def test_normal_log_grad() -> None:
    log_grad1: NDArray[np.float64] = UnivariateNormalDistribution.eval_grad_log(x1, params1)
    # This is computed analytically
    true_log_grad1: NDArray[np.float64] = np.array([0.0, -0.5], dtype=np.float64)
    assert_allclose(log_grad1, true_log_grad1)

    log_grad2: NDArray[np.float64] = UnivariateNormalDistribution.eval_grad_log(x1, params2)
    # This is computed analytically
    true_log_grad2: NDArray[np.float64] = np.array([-0.5, -0.125], dtype=np.float64)
    assert_allclose(log_grad2, true_log_grad2)


def test_normal_sample() -> None:
    sample1: NDArray[np.float64] = UnivariateNormalDistribution.generate_samples([1], params1)
    assert sample1 is not None

    sample2: NDArray[np.float64] = UnivariateNormalDistribution.generate_samples([1], params2)
    assert sample2 is not None
