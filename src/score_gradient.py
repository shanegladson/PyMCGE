from functools import partial

import numpy as np
from numpy.typing import NDArray

from src._costs._cost_function import _CostFunction
from src.enums import ControlVariate, DistributionType
from src.gradient.gradient import Gradient
from src.gradient_estimator import GradientEstimator


class ScoreGradient(GradientEstimator):
    def __init__(self, cost: _CostFunction, dist_type: DistributionType) -> None:
        """
        Creates the score gradient object. Lightweight class used to
        evaluate gradients of an objective function using the score
        gradient method.
        :param CostFunction cost: Derived class with parent CostFunction
        :param dist: Derived class with parent Distribution
        """
        super().__init__(cost, dist_type)

    def eval_integrand(self, x: NDArray[np.float64], dist_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the score function integrand.
        :param NDArray x: Randomly sampled value
        :param NDArray dist_params: Structural parameters for distribution function
        :return: Product of cost, density, and gradient of log-density
        """
        return self.cost.eval_cost(x) * self.dist.eval_density(x, dist_params) * self.dist.eval_grad_log(x, dist_params)

    def mc_grad_estimate_from_dist(
        self, n_samp: int, dist_params: NDArray[np.float64], beta: ControlVariate | float = ControlVariate.NONE
    ) -> Gradient:
        """
        Uses traditional Monte Carlo to estimate the gradient of the
        objective function.
        :param int n_samp: Number of samples
        :param NDArray dist_params: Structural parameters for distribution function
        :param ControlVariate | float beta: Control variate parameter to be used
        :return: Array of estimated gradients
        """
        samples: NDArray[np.float64] = self.dist.generate_samples([n_samp], dist_params)
        if len(samples.shape) == 1:
            samples = samples[:, np.newaxis]

        grad_log: NDArray[np.float64] = np.apply_along_axis(
            partial(self.dist.eval_grad_log, struct_params=dist_params), axis=1, arr=samples
        )

        cost: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=samples)
        adjusted_cost: NDArray[np.float64] = self.adjust_for_cv(cost, beta)

        gradient_estimates = grad_log * adjusted_cost[:, np.newaxis]
        gradient_estimates_mean = np.mean(gradient_estimates, axis=0)

        variance_estimate = np.mean(np.power(gradient_estimates, 2), axis=0) - np.power(gradient_estimates_mean, 2)
        variance_estimate = np.divide(variance_estimate, n_samp, dtype=np.float64)

        gradient: Gradient = Gradient(gradient_estimates_mean, variance_estimate, n_samp)
        return gradient

    def adjust_for_cv(self, cost: NDArray[np.float64], beta: ControlVariate | float) -> NDArray[np.float64]:
        """
        Subtracts the control variate parameter to give the adjusted cost.
        :param NDArray[np.float64] cost: Numpy array of costs for each sample
        :param ControlVariate | float beta: Parameter to be used in control variate adjustment
        :return: Numpy array of adjusted costs
        """
        cv_param: float
        if isinstance(beta, ControlVariate):
            if beta == ControlVariate.NONE:
                cv_param = 0
            elif beta == ControlVariate.AVERAGE:
                cv_param = cost.mean()
            else:
                raise NotImplementedError("Unknown control variate enum used!")
        else:
            cv_param = beta
        adjusted_cost: NDArray[np.float64] = cost - cv_param
        return adjusted_cost
