import unittest
from math import exp

import numpy as np
from itcl_quantization.json.specification import Operator
from tensorflow.keras.activations import sigmoid

import itcl_inference_engine.util.quantization as quant
from itcl_inference_engine.layers.common.layer import ILayer


def _sigmoid(x):
    return 1 / (1 + exp(-x))


sigmoidMap = np.vectorize(_sigmoid)


class Sigmoid(ILayer):
    """
    LOGISTIC / SIGMOID

    Input 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
    Output 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor

        restriction: (scale, zero_point) = (1.0 / 256.0, -128)
    """

    def __init__(self, input_scale, input_zerop) -> None:
        """Sigmoid / Logistic constructor.

        Args:
            input_scale (float32): Input scale (between 0 and 1)
            input_zerop (int8): Input Zero Point
        """
        super().__init__()
        self.__input_scale = input_scale
        self.__input_zerop = input_zerop
        self.__output_scale = 1.0 / 256.0
        self.__output_zero_p = -128

    @classmethod
    def from_model(cls, operator: Operator):
        """Constructor from Json Operator

        Args:
            layer (Operator): Json Operator with all the input and output tensors.

        Returns:
            Logistic Layer: Logistic Layer Instance
        """
        return cls(operator["inputs"][0]["scale"], operator["inputs"][0]["zero_point"])

    def infer(self, input_: np.ndarray):
        """Infer method.

        Args:
            sigmoid_input (int8 ndarray): Input Tensor to be infered

        Returns:
            (int8): Quantized input tensor after applying the sigmoid activation function
        """
        int8_input = quant.from_int8(input_, self.__input_zerop, self.__input_scale)

        res = np.array(sigmoid(int8_input))
        return quant.to_int8(res, self.__output_zero_p, self.__output_scale)


class TestSigmoid(unittest.TestCase):
    def test_op(self):

        x = np.array([42, 49, 28, 42, -2])
        y = np.array([65, 82, 19, 65, -80])

        sigmoid_ = Sigmoid(0.0588817298412323, 23)

        np.testing.assert_almost_equal(y, sigmoid_.infer(x))
