from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Gradient:
    gradient: NDArray[np.float64]
    variance: NDArray[np.float64] | None = None
    n_samples: int | np.int64 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gradient, np.ndarray):
            raise TypeError("Gradient must be of type NDArray[np.float64]!")

        if self.variance is not None and not isinstance(self.variance, np.ndarray):
            raise TypeError("Variance must be of type NDArray[np.float64]!")
