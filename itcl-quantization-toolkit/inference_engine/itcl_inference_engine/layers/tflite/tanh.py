import unittest

import numpy as np
from itcl_quantization.json.specification import Node, Operator

import itcl_inference_engine.util.quantization as quant
from itcl_inference_engine.layers.common.layer import ILayer


class Tanh(ILayer):
    """
        TANH
    Input 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
    Output 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
        restriction: (scale, zero_point) = (1.0 / 128.0, 0)
    """

    def __init__(self, input_scale, input_zp) -> None:
        """Method Constructor

        This Layer does not require any output quantization parameters as they are constants.


        Args:
            input_scale (np.fp32): Input Node Scale
            input_zp (np.int8): Input Node Zero Point
        """
        super().__init__()
        self.__input_scale = input_scale
        self.__input_zp = input_zp

    @classmethod
    def from_model(cls, operator: Operator):
        """Constructor from Json Operator

        Args:
            operator (Operator): Json Operator

        Returns:
            Instance of the Tanh Layer
        """
        return cls(operator["inputs"][0]["scale"], operator["inputs"][0]["zero_point"])

    @classmethod
    def from_node(cls, node: Node):
        """Constructor from Json Node

        Args:
            node (Node): Json Node

        Returns:
            Instance of the Tanh Layer
        """
        return cls(node["scale"], node["zero_point"])

    def infer(self, input_: np.ndarray) -> np.ndarray:
        """
        Method to infer the output of the layer.
        Args:
            tanh_input (np.int8): Data to be processed by the layer.

        Returns:
            np.int8: Input data after applying Tanh.
        """
        res = quant.from_int8(input_, self.__input_zp, self.__input_scale)

        return quant.to_int8(np.array(np.tanh(res)), 0, 1 / 128)


class TestTanh(unittest.TestCase):
    def test_op(self):

        x = np.array([42, 49, 28, 42, -2])
        y = np.array([103, 117, 37, 103, -115])

        tanh = Tanh(0.0588817298412323, 23)

        np.testing.assert_almost_equal(y, tanh.infer(x))

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = (np.tanh(x) * 128).astype(np.int8)

        tanh = Tanh(1, 0)

        np.testing.assert_almost_equal(y, tanh.infer(x))
