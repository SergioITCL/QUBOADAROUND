from math import frexp
from typing import Generic, TypeVar

import numpy as np

E = TypeVar("E", bound=int | np.ndarray)


class FloatingIntegerApprox(Generic[E]):
    def __init__(self, floating_value: float, integer_only: bool = True) -> None:
        self._fp = floating_value
        self._integer_only = integer_only
        q, shift = frexp(floating_value)
        quantized_multiplier = round(q * (1 << 31))

        assert quantized_multiplier <= (1 << 31)

        if quantized_multiplier == (1 << 31):
            quantized_multiplier //= 2
            shift += 1

        if shift < -31:
            self._shift = 0
            quantized_multiplier = 0

        self._q = quantized_multiplier

        self._shift = 31 - shift
        self._round = 1 << (self._shift - 1)

    def __mul__(self, other: E) -> E:

        if not self._integer_only:
            return other * self._fp  # type: ignore

        res = other * self._q + self._round
        return res >> self._shift  # type: ignore
