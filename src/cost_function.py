from abc import ABC

import numpy as np


class CostFunction(ABC):
    @staticmethod
    def eval_cost(x: np.ndarray, struct_params: np.ndarray) -> float:
        raise NotImplementedError('Analytic cost function must be provided!')

    @staticmethod
    def eval_grad(x: np.ndarray, struct_params: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Analytic gradient must be provided!')
