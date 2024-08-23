import numpy as np
from numpy.typing import ArrayLike, NDArray

from src._distributions._distribution import _Distribution
from src._distributions._univariate_gamma import _UnivariateGammaDistribution


class _UnivariateExponentialDistribution(_Distribution):
    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        """
        Computes the exponential density function where input parameters
        are given as (lambda) at point x. Returns a numpy float.
        :param NDArray x: Scalar of point to evaluate the density
        :param NDArray struct_params: Numpy array of (lambda)
        :return: Density of exponential distribution at point x
        """
        lmda = _UnivariateExponentialDistribution.get_parameters(struct_params)
        gamma_params: NDArray[np.float64] = np.array([1.0, lmda], dtype=np.float64)
        density: np.float64 = _UnivariateGammaDistribution.eval_density(x, gamma_params)
        return density

    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        lmda = _UnivariateExponentialDistribution.get_parameters(struct_params)
        gamma_params: NDArray[np.float64] = np.array([1.0, lmda], dtype=np.float64)
        samples: NDArray[np.float64] = _UnivariateGammaDistribution.generate_samples(shape, gamma_params)

        return np.asarray(samples, dtype=np.float64)

    @staticmethod
    def get_parameters(struct_params: NDArray[np.float64]) -> tuple[np.float64, ...]:
        lmda: np.float64 = np.float64(struct_params[0])
        return (lmda,)
