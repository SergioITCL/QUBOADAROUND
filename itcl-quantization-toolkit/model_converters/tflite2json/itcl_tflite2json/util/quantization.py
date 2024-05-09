import numpy as np

def quantize_uint8(data: np.ndarray, zero_point: int, scale: float) -> np.ndarray:
    """
    Quantize the given data using the given zero point and scale.
    """
    return np.clip((data * scale + zero_point), -127, 128).astype(np.int8)


def dequantize_uint8(data: np.ndarray, zero_point: int, scale: float) -> np.ndarray:
    """
    Dequantize the given data using the given zero point and scale.
    """
    return (data.astype(np.float32) - zero_point) / scale