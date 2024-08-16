from functools import partial

import numpy as np
from numpy.typing import NDArray

from src.cost_function import CostFunction
from src.distribution import Distribution


class ScoreGradient:
    def __init__(self, cost: CostFunction, dist: Distribution) -> None:
        """
        Creates the score gradient object. Lightweight class used to
        evaluate gradients of an objective function using the score
        gradient method.
        :param CostFunction cost: Derived class with parent CostFunction
        :param dist: Derived class with parent Distribution
        """
        self.cost: CostFunction = cost
        self.dist: Distribution = dist

    def eval_integrand(self, x: NDArray[np.float64], dist_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the score function integrand.
        :param NDArray x: Randomly sampled value
        :param NDArray dist_params: Structural parameters for distribution function
        :return: Product of cost, density, and gradient of log-density
        """
        return (self.cost.eval_cost(x) *
                self.dist.eval_density(x, dist_params) *
                self.dist.eval_grad_log(x, dist_params))

    def mc_grad_estimate(self, n_samp: int, lb: NDArray[np.float64], ub: NDArray[np.float64],
                         dist_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Uses traditional Monte Carlo to estimate the gradient of the
        objective function.
        :param int n_samp: Number of samples
        :param NDArray[np.float64] lb: Lower bound for uniform samples
        :param NDArray[np.float64] ub: Upper bound for uniform samples
        :param NDArray dist_params: Structural parameters for distribution function
        :return: Array of estimated gradients
        """
        n_col = dist_params.size
        gradient_estimates: NDArray[np.float64]
        random_data: NDArray[np.float64] = np.random.uniform(lb, ub, size=(n_samp, n_col))
        gradient_estimates = np.apply_along_axis(partial(self.eval_integrand, dist_params=dist_params), axis=1, arr=random_data)

        return np.mean(gradient_estimates, axis=1)

