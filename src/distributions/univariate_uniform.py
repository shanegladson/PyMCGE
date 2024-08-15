import numpy as np
from numpy.typing import NDArray

from src.distribution import Distribution


class UnivariateUniformDistribution(Distribution):
    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        """
        Returns the density where the structural parameters are a
        tuple of (a, b) where a and b correspond to lower/upper
        boudns respectively.
        :param NDArray x: Point at which to evaluate the density
        :param NDArray struct_params: Tuple of bounds given as (a,b)
        :return: Density
        """
        x = x[0]
        a = struct_params[0]
        b = struct_params[1]
        if a <= x <= b:
            return np.float64(1 / (b - a))
        else:
            return np.float64(0.)

    @staticmethod
    def eval_grad(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the gradient with respect to the structural parameters.
        :param NDArray x: Point at which to evaluate the gradient
        :param NDArray struct_params: Structural parameters
        :return: Tuple of grad(a, b)
        """
        x = x[0]
        a = struct_params[0]
        b = struct_params[1]
        if a <= x <= b:
            dpda = np.power(b - a, -2)
            dpdb = -np.power(b - a, -2)
            return np.array([dpda, dpdb], dtype=np.float64)
        else:
            return np.array([0., 0.], dtype=np.float64)

    @staticmethod
    def eval_grad_log(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the gradient of the log-density.
        :param x: Point at which to evaluate the log-gradient
        :param struct_params: Structural parameters
        :return: Tuple of grad log (a, b)
        """
        x = x[0]
        a = struct_params[0]
        b = struct_params[1]
        if a <= x <= b:
            dlogpda = np.power(b - a, -1)
            dlogpdb = -np.power(b - a, -1)
            return np.array([dlogpda, dlogpdb], dtype=np.float64)
        else:
            return np.array([np.NAN, np.NAN], dtype=np.float64)