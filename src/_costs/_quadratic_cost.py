import numpy as np
from numpy.typing import NDArray

from src._costs._cost_function import _CostFunction


class _QuadraticCost(_CostFunction):
    def __init__(self, struct_params: NDArray) -> None:
        """
        Takes as input the (n, 3) array of structural parameters
        :params NDArray struct_params: (3, n) or (3, 1) array of structural parameters
        """
        self.a = struct_params[0]
        self.b = struct_params[1]
        self.c = struct_params[2]

    def eval_cost(self, x: NDArray) -> np.float64:
        """
        Evaluates the cost as ax^2+bx+c. If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :return: Scalar cost
        """
        return np.sum(self.a * x**2 + self.b * x + self.c, dtype=np.float64)

    def eval_grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the gradient of cost. If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :return: Gradient of cost for each element in x
        """
        return 2 * self.a * x + self.b
