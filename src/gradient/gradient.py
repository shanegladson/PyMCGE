from dataclasses import dataclass
from typing import Union

import numpy as np 
from numpy.typing import NDArray


@dataclass
class Gradient:
    gradient: NDArray[np.float64]
    variance: Union[NDArray[np.float64], None] = None
    n_samples: Union[int, np.int64, None] = None
