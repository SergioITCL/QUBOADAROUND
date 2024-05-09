import unittest

import numpy as np
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.operator import IOperator
from itcl_inference_engine.layers.onnx.dequantize_linear import \
    DequantizeLinear
from itcl_inference_engine.layers.onnx.quantize_linear import QuantizeLinear


class QLinearSigmoid(IOperator):
    """Add two int quantized uint8 tensors and return a uint8 tensor

    https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.QLinearSigmoid

    """

    def __init__(self, input_scale, input_zerop, output_scale, output_zerop) -> None:
        """

        Args:
            input_scale (fp32): Input Scale
            input_zerop (uint8): Input Zero Point

            output_scale (fp32): Output Scale
            output_zerop (uint8): Output Zero Point
        """
        super().__init__()

        self.__i_s = input_scale
        self.__i_zp = input_zerop

        self.__o_s = output_scale
        self.__o_zp = output_zerop

    @classmethod
    def from_model(cls, operator: Operator):
        """Buils the operator from a given node

        Args:
            operator (Operator): Sigmoid Operator

        Returns:
            QLinearSigmoid Instance
        """

        i = operator["inputs"]

        input_s = i[0]["scale"]
        input_zp = i[0]["zero_point"]

        o_s = i[1]["scale"]
        o_zp = i[1]["zero_point"]

        return cls(input_s, input_zp, o_s, o_zp)

    def infer(self, input_: np.ndarray):
        """Adds input an input tensor with another weight tensor
            C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point

        Args:
            input (np.ndarray): unit8 array

        Returns:
            uint8: Output Tensor
        """
        i_scale = self.__i_s
        i_zp = self.__i_zp

        real_value = DequantizeLinear(i_scale, i_zp).infer(input_)

        sig = _sigmoid(real_value)
        o_scale = self.__o_s
        o_zp = self.__o_zp
        return QuantizeLinear(o_scale, o_zp).infer(sig)


def _sigmoid(arr: np.ndarray) -> np.ndarray:
    """Applies the sigmoid function to an array
    Sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        arr (np.ndarray): numpy array

    Returns:
        np.ndarray: Numpy Array
    """
    return 1 / (1 + np.exp(-arr))  # type: ignore
