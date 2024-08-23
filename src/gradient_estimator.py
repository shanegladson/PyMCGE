from abc import ABC

from src._costs._cost_function import _CostFunction
from src._distributions._distribution import _Distribution
from src.enums import DistributionType
from src.helper_functions import get_distribution_from_type


class GradientEstimator(ABC):
    def __init__(self, cost: _CostFunction, dist_type: DistributionType) -> None:
        self.__cost: _CostFunction = cost
        self.__dist_type: DistributionType = dist_type
        self.__dist: _Distribution = get_distribution_from_type(self.__dist_type)

    @property
    def cost(self) -> _CostFunction:
        return self.__cost

    @property
    def dist_type(self) -> DistributionType:
        return self.__dist_type

    @property
    def dist(self) -> _Distribution:
        return self.__dist
