import unittest

import numpy as np
from itcl_quantization.json.specification import Operator
from itcl_quantization.quantization.operators import Quantization

from itcl_inference_engine.layers.common.layer import ILayer


class Dequantize(ILayer):
    """
    (DE)QUANTIZE (Requantization)
    Input 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
    Output 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor

    """

    def __init__(self, input_scale, input_zp, dtype: str) -> None:
        """Base Constructor

        Builds a layer that dequantizes an int8 quantized value

        Args:
            input_scale (_type_): _description_
            input_zp (_type_): _description_
        """
        super().__init__()
        self.input_scale = input_scale
        self.input_zp = input_zp
        self.Q = Quantization(dtype)

    @classmethod
    def from_model(cls, operator: Operator):
        """Builds a layer from a json operator
            This operator only has input Scale an ZP
        Args:
            layer (Operator): JSON OPERATOR

        Returns:
            Instance of a dequantization layer
        """

        input_ = operator["inputs"][0]

        if input_ is None or input_["tensor"] is None:
            raise ValueError("Dequantize layer input is None or there is no tensor")

        return cls(input_["scale"], input_["zero_point"], input_["tensor"]["dtype"])

    def infer(self, input_: np.ndarray) -> np.ndarray:
        """Inference Method.
            Dequantized an int8 tensor
        Args:
            input (int8 np.ndarray): Input int8 tensor

        Returns:
            np.ndarray: Float32 dequantized tensor
        """
        return self.Q.dequantize(input_, self.input_zp, self.input_scale)


class TestDequantize(unittest.TestCase):
    def test_op(self):
        x = np.array([0, 3, 128, -127]).astype(np.int8)
        x_scale = float(2)
        x_zero_point = 128

        y = np.array([-256, -250, -512, -510], dtype=np.float32)

        res = Dequantize(x_scale, x_zero_point, "int8").infer(x)
        np.testing.assert_array_almost_equal(res, y)
