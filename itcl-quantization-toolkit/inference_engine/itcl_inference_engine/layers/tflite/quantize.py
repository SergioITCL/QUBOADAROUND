import unittest

import numpy as np
from itcl_quantization import Quantization
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.layer import ILayer
from itcl_inference_engine.util.quantization import to_int8


class Quantize(ILayer):
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

    def __init__(self, output_scale, output_zp, dtype: str) -> None:
        """
        Builds a layer that quantizes a float32 tensor
        """
        super().__init__()

        self.output_scale = output_scale
        self.output_zp = output_zp
        self.Q = Quantization(dtype)

    @classmethod
    def from_model(cls, operator: Operator):
        """Builds the layer from a json operator

        Args:
            operator (Operator): JSON OPERATOR

        Returns:
            Instance of a quantization layer
        """
        output = operator["outputs"][0]

        if output is None or output["tensor"] is None:
            raise ValueError("Quantize layer output is None or there is no tensor")

        return cls(output["scale"], output["zero_point"], output["tensor"]["dtype"])

    def infer(self, input_: np.ndarray) -> np.ndarray:
        """Given an input tensor, quantizes it in int8

        Args:
            input (np.ndarray): Input tensor (Usually FP32)

        Returns:
            np.ndarray: Quantized tensor (int8)
        """
        return self.Q.quantize(input_, self.output_zp, self.output_scale)


class TestQuantize(unittest.TestCase):
    def test_op(self):
        x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
        y_scale = 0.8
        y_zero_point = 64
        y = np.array([64, 67, 68, 127, -128, -128]).astype(np.int8)

        res = Quantize(y_scale, y_zero_point, "int8").infer(x)
        np.testing.assert_array_almost_equal(res, y)
