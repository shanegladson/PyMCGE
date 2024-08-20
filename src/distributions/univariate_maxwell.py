import numpy as np 
from numpy.typing import NDArray, ArrayLike

from src.distribution import Distribution

class UnivariateDSMaxwellDistribution(Distribution):
    
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
        x = x[0]
        mu = struct_params[0]
        sigma_sq = struct_params[1]
        sigma = np.sqrt(sigma_sq, dtype=np.float64)
        diff_sq: np.float64 = np.power(x - mu, 2., dtype=np.float64)
        density = diff_sq * np.exp(-0.5 * diff_sq / sigma_sq) / (np.power(sigma, 3.) * np.sqrt(2 * np.pi))
        return np.float64(density)

    @staticmethod
    def eval_grad_log(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates teh gradient at point x with structural parameters
        (mu, sigma_sq). Returns the gradient in a tuple.
        :param NDArray x: Scalar of point at which to evaluate gradient
        :param NDArray struct_params: Tuple of (mu, sigma_sq)
        :return Tuple as grad(mu, sigma_sq)
        """
        x = x[0]
        mu = struct_params[0]
        sigma_sq = struct_params[1]
        dpdmu = (x - mu) / sigma_sq - 2 / (x - mu)
        dpdsigma_sq = 0.5 * np.power(x - mu, 2.) / np.power(sigma_sq, 2.) - 0.5 * 3. / sigma_sq
        return np.array([dpdmu, dpdsigma_sq], dtype=np.float64)

    @staticmethod
    def generate_initial_guess() -> NDArray[np.float64]:
        return np.array([0., 1.], dtype=np.float64)

    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("No current support for generating samples from the Maxwell" 
                                  "distribution!")
