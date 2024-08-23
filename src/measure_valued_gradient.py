import numpy as np
from numpy.typing import NDArray

from src._costs._cost_function import _CostFunction
from src._distributions._distribution import _Distribution
from src.distributions import *
from src.enums import DistributionType
from src.gradient.gradient import Gradient
from src.helper_functions import get_distribution_from_type


class MeasureValuedGradient:
    def __init__(self, cost: _CostFunction, dist_type: DistributionType) -> None:
        """
        Creates the measure-valued gradient object. Lightweight class
        used to evaluate gradients of an objective function using
        the measure-valued gradient method.
        :param CostFunction cost: Derived class with parent CostFunction
        :param DistributionType dist: Derived class with parent Distribution
        """
        self.cost: _CostFunction = cost
        self.dist_type: DistributionType = dist_type
        # TODO: This member variable may not be needed
        self.dist: _Distribution = get_distribution_from_type(self.dist_type)

    def mc_grad_estimate_from_dist(self, n_samp: int, dist_params: NDArray[np.float64]) -> Gradient:
        """
        Uses traditional Monte Carlo to estimate the gradient of the objective function.
        :param int n_samp: Number of samples
        :param NDArray dist_params: Structural parameters for distribution function
        """
        if self.dist_type != DistributionType.NORMAL:
            pos_samp, neg_samp = self._generate_mvg_samples(n_samp, dist_params)
            const: np.float64 = self._get_mvg_constant(dist_params)[0]

            pos_cost: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=pos_samp)
            neg_cost: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=neg_samp)

            gradient_estimate = const * np.mean(pos_cost - neg_cost, axis=0)
        else:
            pos_samp_mu, neg_samp_mu, pos_samp_sigma_sq, neg_samp_sigma_sq = self._generate_mvg_samples(
                n_samp, dist_params
            )
            const_mu, const_sigma_sq = self._get_mvg_constant(dist_params)

            pos_cost_mu: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=pos_samp_mu)
            neg_cost_mu: NDArray[np.float64] = np.apply_along_axis(self.cost.eval_cost, axis=1, arr=neg_samp_mu)

            pos_cost_sigma_sq: NDArray[np.float64] = np.apply_along_axis(
                self.cost.eval_cost, axis=1, arr=pos_samp_sigma_sq
            )
            neg_cost_sigma_sq: NDArray[np.float64] = np.apply_along_axis(
                self.cost.eval_cost, axis=1, arr=neg_samp_sigma_sq
            )

            gradient_estimate_mu = const_mu * np.mean(pos_cost_mu - neg_cost_mu, axis=0)
            gradient_estimate_sigma_sq = const_sigma_sq * np.mean(pos_cost_sigma_sq - neg_cost_sigma_sq, axis=0)
            gradient_estimate = np.array([gradient_estimate_mu, gradient_estimate_sigma_sq], dtype=np.float64)

        return Gradient(gradient_estimate)

    def _generate_mvg_samples(self, n_samp: int, dist_params: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        """
        Generates samples for use in the measure-valued gradient method
        according to the input distribution. Samples are returned as
        (positive measure, negative_measure).
        :param int n_samp: Number of samples to be generated
        :param NDArray[np.float64] dist_params: Parameters to be used in the distribution
        """
        match self.dist_type:
            case DistributionType.NORMAL:
                mu, sigma_sq = UnivariateNormalDistribution.get_parameters(dist_params)
                sigma = np.sqrt(sigma_sq)

                weibull_params = np.array([2.0, 0.5], dtype=np.float64)
                pos_samp_mu = mu + sigma * UnivariateWeibullDistribution.generate_samples([n_samp], weibull_params)
                neg_samp_mu = mu - sigma * UnivariateWeibullDistribution.generate_samples([n_samp], weibull_params)

                sigma_sq_params = np.array([mu, sigma_sq], dtype=np.float64)
                pos_samp_sigma_sq = UnivariateDSMaxwellDistribution.generate_samples([n_samp], sigma_sq_params)
                neg_samp_sigma_sq = UnivariateNormalDistribution.generate_samples([n_samp], sigma_sq_params)

                return pos_samp_mu, neg_samp_mu, pos_samp_sigma_sq, neg_samp_sigma_sq
            case DistributionType.WEIBULL:
                alpha, beta = UnivariateWeibullDistribution.get_parameters(dist_params)

                weibull_params = np.array([alpha, beta], dtype=np.float64)
                pos_samp = UnivariateWeibullDistribution.generate_samples([n_samp], weibull_params)

                gamma_params: NDArray[np.float64] = np.array([2.0, beta], dtype=np.float64)
                neg_samp = UnivariateGammaDistribution.generate_samples([n_samp], gamma_params)
                neg_samp = np.power(neg_samp, 1.0 / alpha, dtype=np.float64)

                return pos_samp, neg_samp
            case DistributionType.GAMMA:
                alpha, beta = UnivariateGammaDistribution.get_parameters(dist_params)

                pos_gamma_params = np.array([alpha, beta], dtype=np.float64)
                pos_samp = UnivariateGammaDistribution.generate_samples([n_samp], pos_gamma_params)

                neg_gamma_params = np.array([alpha + 1, beta], dtype=np.float64)
                neg_samp = UnivariateGammaDistribution.generate_samples([n_samp], neg_gamma_params)

                return pos_samp, neg_samp
            case _:
                raise NotImplementedError("Not currently supported!")

    def _get_mvg_constant(self, dist_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Gets the constant associated with the measure-valued gradient according
        to the input parameters and underlying distribution.
        :param NDArray[np.float64] dist_params: Parameter distribution
        :return: Array of constants
        """
        const: NDArray[np.float64] = np.empty(2, dtype=np.float64)
        match self.dist_type:
            case DistributionType.NORMAL:
                _, sigma_sq = UnivariateNormalDistribution.get_parameters(dist_params)
                sigma = np.sqrt(sigma_sq, dtype=np.float64)
                const[0] = 1.0 / np.sqrt(2.0 * sigma_sq * np.pi, dtype=np.float64)
                const[1] = 1.0 / sigma
            case DistributionType.POISSON:
                const[0] = np.float64(1.0)
            case DistributionType.WEIBULL:
                _, beta = UnivariateWeibullDistribution.get_parameters(dist_params)
                const[0] = np.float64(1.0 / beta)
            case DistributionType.GAMMA:
                alpha, beta = UnivariateGammaDistribution.get_parameters(dist_params)
                const[0] = np.float64(alpha / beta)
            case _:
                raise NotImplementedError("Distribution type not supported!")
        return const
