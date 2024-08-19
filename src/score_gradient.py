from functools import partial
from typing import Union

import numpy as np
from numpy.typing import NDArray

from src.cost_function import CostFunction
from src.distribution import Distribution
from src.enums import ControlVariate

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

    def mc_grad_expectation(self, n_samp: int, lb: NDArray[np.float64], ub: NDArray[np.float64],
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
        random_data: NDArray[np.float64] = np.random.uniform(lb, ub, size=(n_samp, n_col))
        gradient_estimates = np.apply_along_axis(partial(self.eval_integrand, dist_params=dist_params),
                                                 axis=1, arr=random_data)

        return np.mean(gradient_estimates, axis=1)

    def mc_grad_estimate_from_dist(self, n_samp: int, dist_params: NDArray[np.float64], beta: Union[ControlVariate, float] = ControlVariate.NONE) -> NDArray[np.float64]:
        """
        Uses traditional Monte Carlo to estimate the gradient of the
        objective function. 
        :param int n_samp: Number of samples
        :param NDArray dist_params: Structural parameters for distribution function
        :param Union[ControlVariate | float] beta: Control variate parameter to be used
        :return: Array of estimated gradients
        """
        # Sample from the given distribution
        samples: NDArray[np.float64] = self.dist.generate_samples([n_samp], dist_params)
        if len(samples.shape) == 1:
            samples = samples[:, np.newaxis]
        # Compute the gradient of the log-density for each observation
        grad_log: NDArray[np.float64] = np.apply_along_axis(partial(self.dist.eval_grad_log, struct_params=dist_params),
                                                            axis=1, arr=samples)
        # Compute the cost for each observation
        cost: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=samples)
        adjusted_cost: NDArray[np.float64] = self.adjust_for_cv(cost, beta)
        # Combine the gradient with the cost
        gradient_estimates = grad_log * adjusted_cost[:, np.newaxis]
        return np.mean(gradient_estimates, axis=0)

    def adjust_for_cv(self, cost: NDArray[np.float64], beta: Union[ControlVariate, float]) -> NDArray[np.float64]:
        """
        Subtracts the control variate parameter to give the adjusted cost.
        :param NDArray[np.float64] cost: Numpy array of costs for each sample
        :param Union[ControlVariate | float] beta: Parameter to be used in control variate adjustment
        :return: Numpy array of adjusted costs
        """
        cv_param: float
        if isinstance(beta, ControlVariate):
            if beta == ControlVariate.NONE:
                cv_param = 0
            elif beta == ControlVariate.AVERAGE:
                cv_param = cost.mean()
            else:
                raise NotImplementedError('Unknown control variate enum used!')
        else:
            cv_param = beta
        adjusted_cost: NDArray[np.float64] = cost - cv_param
        return adjusted_cost

