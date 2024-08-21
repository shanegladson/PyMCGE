from src._distributions._distribution import _Distribution
from src.distributions import UnivariateNormalDistribution
from src.distributions import UnivariateUniformDistribution
from src.distributions import UnivariateDSMaxwellDistribution
from src.enums import DistributionType

def get_distribution_from_type(dist_type: DistributionType) -> _Distribution:
    match dist_type:
        case DistributionType.UNIFORM:
            return UnivariateUniformDistribution()
        case DistributionType.NORMAL:
            return UnivariateNormalDistribution()
        case DistributionType.MAXWELL:
            return UnivariateDSMaxwellDistribution()
        case _:
            raise NotImplementedError('Distribution not yet supported!')
