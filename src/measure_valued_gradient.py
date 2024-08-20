from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.cost_function import CostFunction
from src.distribution import Distribution
from src.distributions.univariate_maxwell import UnivariateDSMaxwellDistribution
from src.distributions.univariate_weibull import UnivariateWeibullDistribution
from src.enums import DistributionType
from src.gradient.gradient import Gradient
from src.helper_functions import get_distribution_from_type

class MeasureValuedGradient:
    def __init__(self, cost: CostFunction, dist_type: DistributionType) -> None:
        """
        Creates the measure-valued gradient object. Lightweight class
        used to evaluate gradients of an objective function using
        the measure-valued gradient method.
        :param CostFunction cost: Derived class with parent CostFunction
        :param DistributionType dist: Derived class with parent Distribution
        """

        self.cost: CostFunction = cost
        self.dist_type: DistributionType = dist_type
        # TODO: This member variable may not be needed
        self.dist: Distribution = get_distribution_from_type(self.dist_type)

    def mc_grad_estimate_from_dist(self, n_samp: int, dist_params: NDArray[np.float64]) -> Gradient:
        """
        Uses traditional Monte Carlo to estimate the gradient of the objective function.
        :param int n_samp: Number of samples
        :param NDArray dist_params: Structural parameters for distribution function
        """
        pos_samp, neg_samp = self._generate_mvg_samples(n_samp, dist_params)
        const: np.float64 = self._get_mvg_constant(dist_params)

        pos_cost: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=pos_samp)
        neg_cost: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=neg_samp)

        gradient_estimate = const * np.mean(pos_cost - neg_cost, axis=0)

        return Gradient(gradient_estimate)

    def _generate_mvg_samples(self, n_samp: int, dist_params: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generates samples for use in the measure-valued gradient method
        according to the input distribution. Samples are returned as
        (positive measure, negative_measure).
        :param int n_samp: Number of samples to be generated
        :param NDArray[np.float64] dist_params: Parameters to be used in the distribution
        """
        pos_samp: NDArray[np.float64]
        neg_samp: NDArray[np.float64]
        match self.dist_type:
            case DistributionType.NORMAL:
                mu = dist_params[0]
                sigma_sq = dist_params[1]
                sigma = np.sqrt(sigma_sq)
                maxwell_params: NDArray[np.float64] = np.array([2., 0.5], dtype=np.float64)
                pos_samp = mu + sigma * UnivariateWeibullDistribution.generate_samples([n_samp], maxwell_params)
                neg_samp = mu - sigma * UnivariateWeibullDistribution.generate_samples([n_samp], maxwell_params)
                return (pos_samp, neg_samp)
            case _:
                raise NotImplementedError('Not currently supported!')
        
    def _get_mvg_constant(self, dist_params: NDArray[np.float64]) -> np.float64:
        """
        Gets the constant associated with the measure-valued gradient according
        to the input parameters and underlying distribution.
        :param NDArray[np.float64] dist_params: Parameter distribution
        """
        const: np.float64
        match self.dist_type:
            case DistributionType.NORMAL:
                sigma_sq: np.float64 = dist_params[0]
                const = 1. / np.sqrt(2. * sigma_sq * np.pi, dtype=np.float64)
            case _:
                raise NotImplementedError('Distribution type not supported!')
        return const

