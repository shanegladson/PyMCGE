import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import factorial  # type: ignore

from src._distributions._distribution import _Distribution


class _UnivariatePoissonDistribution(_Distribution):
    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        """
        Computes the Poisson density function where input parameter
        is given as (theta) at point x. Returns a numpy float.
        :param NDArray x: Scalar of point to evaluate the density
        :param NDArray struct_params: Numpy array of (theta)
        :return: Density of Poissondistribution at point x
        """
        k: np.float64 = np.float64(np.floor(x[0]))
        if k < 0.0:
            return np.float64(0.0)
        theta = _UnivariatePoissonDistribution.get_parameters(struct_params)[0]
        density = np.exp(-theta) * np.power(theta, k) / factorial(k)
        return np.float64(density)

    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        theta: np.float64 = _UnivariatePoissonDistribution.get_parameters(struct_params)[0]

        shape = np.asarray(shape, dtype=int)
        samples = _UnivariatePoissonDistribution.rng.poisson(theta, size=shape)

        return np.asarray(samples, dtype=np.float64)

    @staticmethod
    def get_parameters(struct_params: NDArray[np.float64]) -> tuple[np.float64, ...]:
        theta: np.float64 = np.float64(struct_params[0])
        return (theta,)
