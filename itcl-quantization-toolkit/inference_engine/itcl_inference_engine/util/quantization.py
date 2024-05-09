from typing import Callable

import numpy as np
from itcl_quantization import Quantization

max_int32 = (1 << 32) - 1
min_int32 = -(1 << 32)

int8_quantization = Quantization(np.int8)
uint8_quantization = Quantization(np.uint8)


def to_int8(real_value: np.ndarray, zero_point: int, scale: float):
    """
    Given a real value, returns the quantized value in int8

    Args:
        real_value (np.ndarray): An array of real values (typically floats)
        zero_point (int8): Zero point of the quantization (between -128 and 127)
        scale (float): The scale of the quantization (between .0 and .1)
        round (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: An array of quantized values in int8
    """
    return int8_quantization.quantize(real_value, zero_point, scale)


def from_int8(int8_val: np.ndarray, zero_point: int, scale: float):
    """Given a quantized value in int8, returns an approximate real value

    Args:
        int8_val (np.ndarray): Quantized value in int8
        zero_point (int8): Zero point of the quantization (between -128 and 127)
        scale (float32): The scale of the quantization (between .0 and .1)

    Returns:
        Numpy array of real values
    """
    return int8_quantization.dequantize(int8_val, zero_point, scale)


int8_multiplication: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int16(np.int16(x) * np.int16(y))  # type: ignore
int8_addition: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int16(np.int16(x) + np.int16(y))  # type: ignore
int16_multiplication: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int32(np.int32(x) * np.int32(y))  # type: ignore
int16_addition: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int32(np.int32(x) + np.int32(y))  # type: ignore
int32_multiplication: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int64(np.int64(x) * np.int64(y))  # type: ignore
int32_addition: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int64(np.int64(x) + np.int64(y))  # type: ignore
