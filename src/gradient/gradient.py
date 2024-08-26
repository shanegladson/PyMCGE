import numpy as np
from numpy.typing import NDArray


class Gradient:
    def __init__(
        self, gradient: NDArray[np.float64], variance: NDArray[np.float64] | None = None, n_samples: int | None = None
    ) -> None:
        self.__gradient: NDArray[np.float64]
        self.__variance: NDArray[np.float64]
        self.__n_samples: int | None

        self.__gradient = np.asarray(gradient, dtype=np.float64)

        if variance is None:
            self.__variance = np.full_like(self.__gradient, np.nan)
        else:
            self.__variance = np.asarray(variance, dtype=np.float64)

        self.__n_samples = n_samples

    @property
    def gradient(self) -> NDArray[np.float64]:
        return self.__gradient

    @property
    def variance(self) -> NDArray[np.float64]:
        return self.__variance

    @property
    def n_samples(self) -> int | None:
        return self.__n_samples
