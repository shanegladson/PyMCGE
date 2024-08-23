import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import maxwell  # type: ignore

from src._distributions._distribution import _Distribution


class _UnivariateDSMaxwellDistribution(_Distribution):
    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        """
        Computes the density function of the double-sided Maxwell
        distribution where input parameters are given as (mu, sigma_sq)
        at point x. Returns a numpy float.
        :param NDArray x: Scalar of point to evaluate the density
        :param NDArray struct_params: Numpy array of (mu, sigma_sq)
        :return: Density of Maxwell distribution at point x
        """
        x_i: np.float64 = np.float64(x[0])
        mu, sigma_sq = _UnivariateDSMaxwellDistribution.get_parameters(struct_params)
        sigma = np.sqrt(sigma_sq, dtype=np.float64)

        diff_sq: np.float64 = np.power(x_i - mu, 2.0, dtype=np.float64)
        density = diff_sq * np.exp(-0.5 * diff_sq / sigma_sq) / (np.power(sigma, 3.0) * np.sqrt(2 * np.pi))

        return np.float64(density)

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
        mu, sigma_sq = _UnivariateDSMaxwellDistribution.get_parameters(struct_params)

        dpdmu = (x_i - mu) / sigma_sq - 2 / (x_i - mu)
        dpdsigma_sq = 0.5 * np.power(x_i - mu, 2.0) / np.power(sigma_sq, 2.0) - 0.5 * 3.0 / sigma_sq

        return np.array([dpdmu, dpdsigma_sq], dtype=np.float64)

    @staticmethod
    def generate_initial_guess() -> NDArray[np.float64]:
        return np.array([0.0, 1.0], dtype=np.float64)

    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        mu, sigma_sq = _UnivariateDSMaxwellDistribution.get_parameters(struct_params)
        sigma: np.float64 = np.sqrt(sigma_sq, dtype=np.float64)
        shape = np.asarray(shape, dtype=int)

        samples: NDArray[np.float64] = maxwell.rvs(scale=sigma, size=shape)
        signs: NDArray = np.asarray(np.random.choice([1, -1], size=shape), dtype=int)
        samples *= signs
        samples += mu

        return samples

    @staticmethod
    def get_parameters(struct_params: NDArray[np.float64]) -> tuple[np.float64, ...]:
        mu: np.float64 = np.float64(struct_params[0])
        sigma_sq: np.float64 = np.float64(struct_params[1])
        return mu, sigma_sq
