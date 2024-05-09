import unittest

import numpy as np
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.operator import IOperator
from itcl_inference_engine.util.quantization import uint8_quantization


class QuantizeLinear(IOperator):
    def __init__(self, scale, zerop) -> None:
        """Transforms a float32 value into int8

        Args:
            scale (fp32): Node Scale
            zerop (uint8): Node Zero Point
        """
        super().__init__()

        self.__scale = scale
        self.__zerop = zerop

    @classmethod
    def from_model(cls, operator: Operator):
        """Builds a QuantizeLinear operator from a given model

        Args:
            operator (ProtoBuf Operator): ONNX ProtoBuf Operator
            model (ProtoBuf Model): A complete ONNX IR Model in protobuf format.

        Returns:
            DequantizeLinear: An instance of the class
        """

        scale = operator["inputs"][0]["scale"]
        zerop = operator["inputs"][0]["zero_point"]
        return cls(scale, zerop)

    def infer(self, input_: np.ndarray) -> np.ndarray:
        """Cast a 32 bit tensor into a 8 uint bit one.

        y = saturate ((x / y_scale) + y_zero_point)
        Args:
            input (np.ndarray): Input Tensor

        Returns:
            np.ndarray: int8 tensor
        """
        return uint8_quantization.quantize(input_, self.__zerop, self.__scale)
