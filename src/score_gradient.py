import numpy as np

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

    def eval_integrand(self, x: np.ndarray, cost_params: np.ndarray, dist_params: np.ndarray) -> np.ndarray:
        """
        Returns the score function integrand.
        :param np.ndarray x: Randomly sampled value
        :param np.ndarray cost_params: Structural parameters for cost function
        :param np.ndarray dist_params: Structural parameters for distribution function
        :return: Product of cost, density, and gradient of log-density
        """
        return (self.cost.eval_cost(x, cost_params) *
                self.dist.eval_dist(x, dist_params) *
                self.dist.eval_grad_log(x, dist_params))
