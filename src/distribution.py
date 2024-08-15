from abc import ABC

import numpy as np


class Distribution(ABC):
    @staticmethod
    def eval_dist(x: np.ndarray, struct_params: np.ndarray) -> float:
        raise NotImplementedError('Analytic distribution must be provided!')

    @staticmethod
    def eval_grad(x: np.ndarray, struct_params: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Analytic gradient must be provided!')

    @staticmethod
    def eval_grad_log(x: np.ndarray, struct_params: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Analytic gradient of log-distribution must be provided!')
