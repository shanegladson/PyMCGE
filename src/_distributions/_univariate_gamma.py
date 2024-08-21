import numpy as np 
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma  # type: ignore

from src._distributions._distribution import _Distribution


class _UnivariateGammaDistribution(_Distribution):

    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        """
        Computes the gamma density function where input parameters are 
        given as (alpha, beta) at point x. Returns a numpy float.
        :param NDArray x: Scalar of point to evaluate the density
        :param NDArray struct_params: Numpy array of (alpha, beta)
        :return: Density of gamma distribution at point x
        """
        x_i: np.float64 = np.float64(x[0])
        if x_i >= 0.:
            alpha, beta = _UnivariateGammaDistribution.get_parameters(struct_params)
            density = np.power(beta, alpha) * np.power(x_i, alpha - 1) * np.exp(-beta * x_i) / gamma(alpha)
            return density
        else:
            return np.float64(0.)
    
    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        alpha, beta = _UnivariateGammaDistribution.get_parameters(struct_params)
        
        k: np.float64 = alpha
        theta: np.float64 = 1. / beta
        shape = np.asarray(shape, dtype=int)
        samples: NDArray[np.float64] = _UnivariateGammaDistribution.rng.gamma(k, theta, shape)

        return np.asarray(samples, dtype=np.float64)

    @staticmethod
    def get_parameters(struct_params: NDArray[np.float64]) -> tuple[np.float64, ...]:
        alpha: np.float64 = np.float64(struct_params[0])
        beta: np.float64 = np.float64(struct_params[1])
        return alpha, beta

