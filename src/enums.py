from enum import Enum


class ControlVariate(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2


class DistributionType(Enum):
    UNIFORM = 0
    NORMAL = 1
    MAXWELL = 2
    EXPONENTIAL = 3
    WEIBULL = 4
    POISSON = 5
    GAMMA = 6
