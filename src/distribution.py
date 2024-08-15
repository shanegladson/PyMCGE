from abc import ABC

import numpy as np
from numpy.typing import NDArray


class Distribution(ABC):
    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        raise NotImplementedError('Analytic distribution must be provided!')

    @staticmethod
    def eval_grad(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError('Analytic gradient must be provided!')

    @staticmethod
    def eval_grad_log(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError('Analytic gradient of log-distribution must be provided!')
