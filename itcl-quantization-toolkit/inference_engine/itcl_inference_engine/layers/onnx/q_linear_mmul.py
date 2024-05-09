import numpy as np
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.operator import IOperator


class QLinearMatMul(IOperator):
    """Matrix Multiplication with quantized values

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul

    M: Downscale Parameter

    """

    def __init__(
        self,
        input_scale,
        input_zerop,
        weights_tensor,
        w_scale,
        w_zerop,
        out_scale,
        out_zerop,
    ) -> None:
        """_summary_

        Args:
            input_scale (fp32): Input Node Scale value
            input_zerop (uint8): Input ZeroP
            weights_tensor (np.ndarray uint8): Tensor/Matrix to multiply the input with.
            w_scale (fp32): Weight Scale
            w_zerop (uint8): Weight Zerop
            out_scale (fp32): Output Scale
            out_zerop (uint8): Output Zerop
        """

        # fp32 downscale parameter
        self.__M = input_scale * w_scale / out_scale

        self.__w = np.transpose(np.array(weights_tensor))
        self.__i_zp = input_zerop
        self.__w_zp = w_zerop
        self.__o_zp = out_zerop

        super().__init__()

    @classmethod
    def from_model(cls, operator: Operator):
        """_summary_

        Args:
            operator (Operator): A Json Operator

        Returns:
            QLinearMatMul Instance
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
        """Numpy Matrix Multiplication with quantized tensors

        Multiplies $input with $weight tensor matrix


        Args:
            input (np.ndarray uint8): Uint8 Numpy Array / Matrix

        Returns:
            np.ndarray uint8: Matrix Multiplication Result.
        """
        i = input_.astype(np.int32) - self.__i_zp  # As int32
        w = self.__w.astype(np.int32) - self.__w_zp  # As int32

        int32mmul = np.round(self.__M * np.matmul(i, w)) + self.__o_zp

        return np.clip(int32mmul, 0, 255).astype(np.uint8)
