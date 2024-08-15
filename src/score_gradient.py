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

    def eval_integrand(self, x: NDArray[np.float64], cost_params: NDArray[np.float64],
                       dist_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the score function integrand.
        :param NDArray x: Randomly sampled value
        :param NDArray cost_params: Structural parameters for cost function
        :param NDArray dist_params: Structural parameters for distribution function
        :return: Product of cost, density, and gradient of log-density
        """
        return (self.cost.eval_cost(x, cost_params) *
                self.dist.eval_density(x, dist_params) *
                self.dist.eval_grad_log(x, dist_params))
