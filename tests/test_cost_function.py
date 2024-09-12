import unittest

import numpy as np
import pytest

from src._costs._cost_function import _CostFunction


class BlankCostFunction(_CostFunction):
    pass


cost: BlankCostFunction = BlankCostFunction(np.zeros(1))


def test_not_implemented() -> None:

    with pytest.raises(NotImplementedError):
        cost.eval_cost(np.zeros(1))

    with pytest.raises(NotImplementedError):
        cost.eval_grad(np.zeros(1))
