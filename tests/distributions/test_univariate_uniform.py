import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.distributions import UnivariateUniformDistribution

x1: NDArray[np.float64] = np.array([1.0], dtype=np.float64)
x2: NDArray[np.float64] = np.array([-1.0], dtype=np.float64)
params1: NDArray[np.float64] = np.array([0.0, 1.0], dtype=np.float64)
params2: NDArray[np.float64] = np.array([1.0, 10.0], dtype=np.float64)


def test_uniform_density() -> None:
    density1: np.float64 = UnivariateUniformDistribution.eval_density(x1, params1)
    true_density1: np.float64 = np.float64(1.0)
    assert_allclose(density1, true_density1)

    density2: np.float64 = UnivariateUniformDistribution.eval_density(x1, params2)
    true_density2: np.float64 = np.float64(1.0 / 9.0)
    assert_allclose(density2, true_density2)

    density3: np.float64 = UnivariateUniformDistribution.eval_density(x2, params1)
    true_density3: np.float64 = np.float64(0.0)
    assert_allclose(density3, true_density3)

    density4: np.float64 = UnivariateUniformDistribution.eval_density(x2, params2)
    true_density4: np.float64 = np.float64(0.0)
    assert_allclose(density4, true_density4)


def test_uniform_grad() -> None:
    grad1: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(x1, params1)
    true_grad1: NDArray[np.float64] = np.array([1.0, -1.0], dtype=np.float64)
    assert_allclose(grad1, true_grad1)

    grad2: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(x1, params2)
    true_grad2: NDArray[np.float64] = np.array([1.0 / 81.0, -1.0 / 81.0], dtype=np.float64)
    assert_allclose(grad2, true_grad2)

    grad3: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(x2, params1)
    true_grad3: NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)
    assert_allclose(grad3, true_grad3)

    grad4: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad(x2, params2)
    true_grad4: NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)
    assert_allclose(grad4, true_grad4)


def test_uniform_log_grad() -> None:
    log_grad1: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(x1, params1)
    true_log_grad1: NDArray[np.float64] = np.array([1.0, -1.0], dtype=np.float64)
    assert_allclose(log_grad1, true_log_grad1)

    log_grad2: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(x1, params2)
    true_log_grad2: NDArray[np.float64] = np.array([1.0 / 9.0, -1.0 / 9.0], dtype=np.float64)
    assert_allclose(log_grad2, true_log_grad2)

    log_grad3: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(x2, params1)
    true_log_grad3: NDArray[np.float64] = np.array([np.NAN, np.NAN], dtype=np.float64)
    assert_allclose(log_grad3, true_log_grad3)

    log_grad4: NDArray[np.float64] = UnivariateUniformDistribution.eval_grad_log(x2, params2)
    true_log_grad4: NDArray[np.float64] = np.array([np.NAN, np.NAN], dtype=np.float64)
    assert_allclose(log_grad4, true_log_grad4)


def test_uniform_sample() -> None:
    sample1: NDArray[np.float64] = UnivariateUniformDistribution.generate_samples([1], params1)
    assert sample1 >= params1[0]
    assert sample1 <= params1[1]

    sample2: NDArray[np.float64] = UnivariateUniformDistribution.generate_samples([1], params2)
    assert sample2 >= params2[0]
    assert sample2 <= params2[1]
