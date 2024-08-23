import unittest

from src.distributions import *
from src.enums import DistributionType
from src.helper_functions import get_distribution_from_type


class TestDistributionFromType(unittest.TestCase):
    def test_distribution_from_type(self) -> None:
        # Uniform distribution
        unif_dist = get_distribution_from_type(DistributionType.UNIFORM)
        self.assertIsInstance(unif_dist, UnivariateUniformDistribution)

        # Uniform distribution
        norm_dist = get_distribution_from_type(DistributionType.NORMAL)
        self.assertIsInstance(norm_dist, UnivariateNormalDistribution)

        # Uniform distribution
        maxw_dist = get_distribution_from_type(DistributionType.MAXWELL)
        self.assertIsInstance(maxw_dist, UnivariateDSMaxwellDistribution)

        # Weibull distribution
        weib_dist = get_distribution_from_type(DistributionType.WEIBULL)
        self.assertIsInstance(weib_dist, UnivariateWeibullDistribution)
