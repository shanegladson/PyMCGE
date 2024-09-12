import pytest

from src.distributions import *
from src.enums import DistributionType
from src.helper_functions import get_distribution_from_type


def test_distribution_from_type() -> None:
    # Uniform distribution
    unif_dist = get_distribution_from_type(DistributionType.UNIFORM)
    assert isinstance(unif_dist, UnivariateUniformDistribution)

    # Uniform distribution
    norm_dist = get_distribution_from_type(DistributionType.NORMAL)
    assert isinstance(norm_dist, UnivariateNormalDistribution)

    # Uniform distribution
    maxw_dist = get_distribution_from_type(DistributionType.MAXWELL)
    assert isinstance(maxw_dist, UnivariateDSMaxwellDistribution)

    # Weibull distribution
    weib_dist = get_distribution_from_type(DistributionType.WEIBULL)
    assert isinstance(weib_dist, UnivariateWeibullDistribution)
