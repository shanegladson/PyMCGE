from abc import ABC

import numpy as np
from numpy.typing import NDArray, ArrayLike


class _Distribution(ABC):
    rng: np.random.Generator = np.random.default_rng()

    @staticmethod
    def eval_density(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> np.float64:
        raise NotImplementedError("Analytic distribution must be provided!")

    @staticmethod
    def eval_grad(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Analytic gradient must be provided!")

    @staticmethod
    def eval_grad_log(x: NDArray[np.float64], struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Analytic gradient of log-distribution must be provided!")

    @staticmethod
    def generate_initial_guess() -> NDArray[np.float64]:
        raise NotImplementedError("Must return a valid initial guess with the correct dimension!")

    @staticmethod
    def generate_samples(shape: ArrayLike, struct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("The distribution must be able to provide its own samples of arbitrary shape!")

    @staticmethod
    def get_parameters(struct_params: NDArray[np.float64]) -> tuple[np.float64, ...]:
        raise NotImplementedError("The method must be implemented to return all relevant parameters!")
