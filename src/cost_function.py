from abc import ABC

import numpy as np
from numpy.typing import NDArray


class CostFunction(ABC):
    @staticmethod
    def eval_cost(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        raise NotImplementedError('Analytic cost function must be provided!')

    @staticmethod
    def eval_grad(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError('Analytic gradient must be provided!')
