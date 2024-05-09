import numpy as np
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.operator import IOperator
from itcl_inference_engine.util.quantization import uint8_quantization


class DequantizeLinear(IOperator):
    """
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear
    Dequantize Operator

    Transforms a uint8 tensor to fp32
    """

    def __init__(self, scale, zero_point):
        """

        Args:
            scale (float32): Tensor Scale
            zero_point (unit8): Tensor Zero Point
        """
        super().__init__()

        self.__scale = scale
        self.__zerop = zero_point

    @classmethod
    def from_model(cls, operator: Operator):
        """Builds a DequantizeLinear operator from a given model

        Args:
            operator (ProtoBuf Operator): ONNX ProtoBuf Operator
            model (ProtoBuf Model): A complete ONNX IR Model in protobuf format.

        Returns:
            DequantizeLinear: An instance of the class
        """

        scale = operator["inputs"][0]["scale"]
        zerop = operator["inputs"][0]["zero_point"]
        return cls(scale, zerop)

    def infer(self, input_: np.ndarray):
        return uint8_quantization.dequantize(input_, self.__zerop, self.__scale)
