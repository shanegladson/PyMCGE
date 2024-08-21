import numpy as np
from numpy.typing import NDArray, ArrayLike

from src.distribution import Distribution


class UnivariateWeibullDistribution(Distribution):
    
    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        """
        Computes the density function of the Weibull distribution
        where input parameters are given as (alpha, beta)
        at point x. Returns a numpy float.
        :param NDArray x: Scalar of point to evaluate the density
        :param NDArray struct_params: Numpy array of (alpha, beta)
        :return: Density of Weibull distribution at point x
        """
        x_i: np.float64 = np.float64(x[0])
        alpha, beta = UnivariateWeibullDistribution.get_parameters(struct_params)

        exp_term: np.float64 = np.exp(-beta * np.power(x_i, alpha))
        density: np.float64 = alpha * beta * np.power(x_i, alpha - 1) * exp_term

        return np.float64(density)
    
    @staticmethod
    def eval_grad_log(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the gradient at point x with structural parameters
        (alpha, beta). Returns the gradient in a tuple.
        :param NDArray x: Scalar of point at which to evaluate the gradient
        :param NDArray struct_params: Tuple of (alpha, beta)
        :return: Tuple as grad(alpha, beta)
        """
        x_i: np.float64 = np.float64(x[0])
        alpha, beta = UnivariateWeibullDistribution.get_parameters(struct_params)

        dpdalpha = 1. / alpha + np.log(x_i) - alpha * beta * np.power(x_i, alpha - 1.)
        dpdbeta = 1. / beta - np.power(x_i, alpha)

        return np.array([dpdalpha, dpdbeta], dtype=np.float64)

    @staticmethod
    def generate_initial_guess() -> NDArray[np.float64]:
        return np.array([1., 1.], dtype=np.float64)

    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        alpha, beta = UnivariateWeibullDistribution.get_parameters(struct_params)
        shape = np.asarray(shape, dtype=int)

        samples: NDArray[np.float64] = UnivariateWeibullDistribution.rng.weibull(alpha, size=shape)
        samples /= beta

        return np.asarray(samples, dtype=np.float64)

    @staticmethod
    def get_parameters(struct_params: NDArray[np.float64]) -> tuple[np.float64, ...]:
        alpha: np.float64 = np.float64(struct_params[0])
        beta: np.float64 = np.float64(struct_params[1])
        return alpha, beta

