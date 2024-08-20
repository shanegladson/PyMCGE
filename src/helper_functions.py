from src.distribution import Distribution
from src.distributions.univariate_normal import UnivariateNormalDistribution
from src.distributions.univariate_uniform import UnivariateUniformDistribution
from src.distributions.univariate_maxwell import UnivariateDSMaxwellDistribution
from src.enums import DistributionType

def get_distribution_from_type(dist_type: DistributionType) -> Distribution:
    match dist_type:
        case DistributionType.UNIFORM:
            return UnivariateUniformDistribution()
        case DistributionType.NORMAL:
            return UnivariateNormalDistribution()
        case DistributionType.MAXWELL:
            return UnivariateDSMaxwellDistribution()
        case _:
            raise NotImplementedError('Distribution not yet supported!')
