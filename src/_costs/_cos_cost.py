from typing import Final

import numpy as np
from numpy.typing import NDArray

from src._costs._cost_function import _CostFunction


class _CosineCost(_CostFunction):
    def __init__(self, struct_params: NDArray[np.float64]) -> None:
        """
        Takes as input the coefficients and offsets for the cost.
        :param NDArray[np.float64] struct_params: (n, 2) array of structural parameters
        """
        super().__init__(struct_params)
        self.coeff: Final[NDArray[np.float64]] = struct_params[0]
        self.offset: Final[NDArray[np.float64]] = struct_params[1]

    def eval_cost(self, x: NDArray[np.float64]) -> np.float64:
        """
        Evaluates the cost as cos(ax+b). If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :return: Scalar cost
        """
        return np.sum(np.cos(self.coeff * x + self.offset), dtype=np.float64)

    def eval_grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the gradient of cost. If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :return: Gradient of cost for each element in x
        """
        return np.asarray(-self.coeff * np.sin(self.coeff * x + self.offset), dtype=np.float64)
