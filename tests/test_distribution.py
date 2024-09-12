import numpy as np
import pytest

from src._distributions._distribution import _Distribution


class BlankDistribution(_Distribution):
    pass


def test_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BlankDistribution.eval_density(np.zeros(1), np.zeros(1))

    with pytest.raises(NotImplementedError):
        BlankDistribution.eval_grad(np.zeros(1), np.zeros(1))

    with pytest.raises(NotImplementedError):
        BlankDistribution.eval_grad_log(np.zeros(1), np.zeros(1))
