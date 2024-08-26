from abc import ABC
from typing import Final

import numpy as np
from numpy.typing import NDArray


class _CostFunction(ABC):
    def __init__(self, struct_params: NDArray[np.float64]) -> None:
        self.struct_params: Final[NDArray[np.float64]] = struct_params

    def eval_cost(self, x: NDArray[np.float64]) -> np.float64:
        raise NotImplementedError("Analytic cost function must be provided!")

    def eval_grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Analytic gradient must be provided!")
