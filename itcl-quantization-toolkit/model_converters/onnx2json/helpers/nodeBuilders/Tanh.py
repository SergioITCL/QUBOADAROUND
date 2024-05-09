import math
import numpy as np
from helpers.nodeBuilders.DequantizeLinear import DequantizeLinear
from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.QuantizeLinear import QuantizeLinear
from helpers.nodeBuilders.buildBaseNode import build_base_node
from itcl_quantization.json.specification import Node
from util.quantization import create_lut

from itcl_quantization.quantization.lut import ReducedLUT


class Tanh(IBuilder):
    def build(self, node, model):
        """
        It converts a float value into uint8
        
        :param node: The node object that is being processed
        :param model: The model object
        :return: A tuple of three elements.
        """
        input_key = node.input[0]
        output_key = node.output[0]

        return (
            [build_base_node(model, input_key)],
            [build_base_node(model, output_key)],
            "Casts a float value into uint8",
        )


class TanhLUT(IBuilder):
    def __init__(self, depth: int = 3, min_reduce: int = 10):
        self.__depth = depth
        self.__min_reduce = min_reduce

    def build(self, node, model):
        """
        The function takes in the input and output scale and zero point values, and creates a LUT for the
        tanh look up table operator
        
        :param node: The node that is being processed
        :param model: The model object
        :return: The return is a tuple of 3 elements.
        """
        node_list: list = node
        input_node = DequantizeLinear().build(node_list[0], model)
        output_node = QuantizeLinear().build(node_list[2], model)

        self.__input_s = input_node[0][0]["scale"] or 1.0
        self.__input_zp = input_node[0][0]["zero_point"] or 0

        self.__output_s = output_node[0][0]["scale"] or 1.0
        self.__output_zp = output_node[0][0]["zero_point"] or 0

        LUT = create_lut(
            math.tanh,
            self.__input_s,
            self.__input_zp,
            self.__output_s,
            self.__output_zp,
        )

        input_node_lut: Node = {
            "scale": None,
            "tensor": None,
            "zero_point": None,
            "LUT": {
                "LUT": LUT,
                "offset": 0,
                "reduced_LUT": ReducedLUT(LUT, self.__depth, self.__min_reduce).serialize(),
            },
        }

        return (
            [input_node_lut],
            [],
            "Includes a LUT for tanh",
        )
