import numpy as np
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.operator import IOperator


class QLinearAdd(IOperator):
    """Add two int quantized uint8 tensors and return a uint8 tensor

    https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.QLinearAdd

    """

    def __init__(
        self,
        input_scale,
        input_zerop,
        b_tensor,
        b_scale,
        b_zerop,
        output_scale,
        output_zerop,
    ) -> None:
        """

        Args:
            input_scale (fp32): Input Scale
            input_zerop (uint8): Input Zero Point
            b_tensor (np.ndarray uint8): For matmul: the bias tensor
            b_scale (fp32): For matmul, the bias scale
            b_zerop (uint8): For matmul, the bias zerop
            output_scale (fp32): Output Scale
            output_zerop (uint8): Output Zero Point
        """
        super().__init__()

        self.__i_s = input_scale
        self.__i_zp = input_zerop
        self.__b = b_tensor
        self.__b_s = b_scale
        self.__b_zp = b_zerop
        self.__o_s = output_scale
        self.__o_zp = output_zerop

    @classmethod
    def from_model(cls, operator: Operator):
        """Buils the operator from a given node

        Args:
            operator (Operator): Json Operator Dict

        Returns:
            QLinearAdd Instance
        """

        i = operator["inputs"]

        input_s = i[0]["scale"]
        input_zp = i[0]["zero_point"]

        b_tensor_def = i[1]["tensor"]

        if b_tensor_def is None:
            raise ValueError(
                f"Weight Tensor for QLinearAdd ({operator['name']}) is not defined"
            )

        b_tensor = np.array(b_tensor_def["tensor"], dtype=b_tensor_def["dtype"])

        assert list(b_tensor.shape) == b_tensor_def["shape"]

        b_s = i[1]["scale"]
        b_zp = i[1]["zero_point"]

        o_s = i[2]["scale"]
        o_zp = i[2]["zero_point"]

        return cls(input_s, input_zp, b_tensor, b_s, b_zp, o_s, o_zp)

    def infer(self, input_: np.ndarray):
        """Adds input an input tensor with another weight tensor
            C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point

        Args:
            input (np.ndarray): unit8 array

        Returns:
            uint8: Output Tensor
        """
        i_scale = self.__i_s
        b_scale = self.__b_s
        o_scale = self.__o_s
        o_zp = self.__o_zp

        i = input_.astype(np.int32) - self.__i_zp  # i32
        b = self.__b.astype(np.int32) - self.__b_zp  # i32

        C = np.round((i_scale * i + b_scale * b) / o_scale) + o_zp

        return np.clip(
            C,
            0,
            255,
        ).astype(np.uint8)
