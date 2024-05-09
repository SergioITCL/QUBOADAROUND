import numpy as np
from itcl_quantization import Quantization


__Q = Quantization(np.uint8)

def quantize(input, scale: float, zero_point: int) -> np.ndarray:
	return __Q.quantize(input, zero_point, scale)

def dequantize(input, scale: float, zero_point: int) -> np.ndarray:
	return __Q.dequantize(input, zero_point, scale)

create_lut = __Q.create_LUT