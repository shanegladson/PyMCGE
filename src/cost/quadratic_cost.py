import numpy as np
from numpy.typing import NDArray

from src.cost_function import CostFunction


class QuadraticCost(CostFunction):
    @staticmethod
    def eval_cost(x: NDArray, struct_params: np.ndarray) -> np.float64:
        """
        Evaluates the cost as ax^2+bx+c. If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :param struct_params: (n, 3) array of structural parameters
        :return: Scalar cost
        """
        a = struct_params[:, 0]
        b = struct_params[:, 1]
        c = struct_params[:, 2]
        return np.sum(a * x ** 2 + b * x + c, dtype=np.float64)

    @staticmethod
    def eval_grad(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the gradient of cost. If x is an array, the sum of each
        cost is returned.
        :param NDArray x: 1D array of input parameters (length n)
        :param struct_params: (n, 2) array of structural parameters
        :return: Gradient of cost for each element in x
        """
        a = struct_params[:, 0]
        b = struct_params[:, 1]
        return 2 * a * x + b
