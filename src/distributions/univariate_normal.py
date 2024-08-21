import numpy as np
from numpy.typing import NDArray, ArrayLike

from src.distribution import Distribution


class UnivariateNormalDistribution(Distribution):
    
    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        """
        Computes the normal density function where input parameters
        are given as (mu, sigma_sq) at point x. Returns a numpy float.
        :param NDArray x: Scalar of point to evaluate the density
        :param NDArray struct_params: Numpy array of (mu, sigma_sq)
        :return: Density of normal distribution at point x
        """
        x_i: np.float64 = np.float64(x[0])
        mu = struct_params[0]
        sigma_sq = struct_params[1]
        density = np.power(2 * np.pi * sigma_sq, -0.5) * np.exp(-0.5 * ((x_i - mu) ** 2) / sigma_sq)
        return density

    @staticmethod
    def eval_grad_log(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the gradient at point x with structural parameters
        (mu, sigma_sq). Returns the gradient in a tuple.
        :param NDArray x: Scalar of point at which to evaluate gradient
        :param NDArray struct_params: Tuple of (mu, sigma_sq)
        :return: Tuple as grad(mu, sigma_sq)
        """
        x_i: np.float64 = np.float64(x[0])
        mu, sigma_sq = UnivariateNormalDistribution.get_parameters(struct_params)
        
        dpdmu = (x_i - mu) / sigma_sq
        dpdsigma_sq = -0.5 / sigma_sq + 0.5 / (sigma_sq ** 2) * (x_i - mu) ** 2

        return np.array([dpdmu, dpdsigma_sq], dtype=np.float64)

    @staticmethod
    def generate_initial_guess() -> NDArray[np.float64]:
        return np.array([0., 1.], dtype=np.float64)

    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        mu, sigma_sq = UnivariateNormalDistribution.get_parameters(struct_params)

        sigma: np.float64 = np.sqrt(sigma_sq, dtype=np.float64)
        shape = np.asarray(shape, dtype=int)
        samples: NDArray[np.float64] = UnivariateNormalDistribution.rng.normal(mu, sigma, shape)

        return np.asarray(samples, dtype=np.float64)

    @staticmethod
    def get_parameters(struct_params: NDArray[np.float64]) -> tuple[np.float64, ...]:
        mu: np.float64 = np.float64(struct_params[0])
        sigma_sq: np.float64 = np.float64(struct_params[1])
        return mu, sigma_sq

