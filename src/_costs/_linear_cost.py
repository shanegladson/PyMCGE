import numpy as np
from numpy.typing import NDArray

from src._costs._cost_function import _CostFunction


class _LinearCost(_CostFunction):
    def __init__(self, struct_params: NDArray[np.float64]) -> None:
        """
        Takes as input the coefficients and intercepts for the cost.
        :param struct_params: (n, 2) array of structural parameters
        """
        super().__init__(struct_params)
        self.coeff: NDArray[np.float64] = struct_params[0]
        self.intercept: NDArray[np.float64] = struct_params[1]

    def eval_cost(self, x: NDArray[np.float64]) -> np.float64:
        """
        Evaluates the cost as ax+b. If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :return: Scalar cost
        """
        return np.sum(self.coeff * x + self.intercept, dtype=np.float64)

    def eval_grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the gradient of cost. If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :return: Gradient of cost for each element in x
        """
        return self.coeff
