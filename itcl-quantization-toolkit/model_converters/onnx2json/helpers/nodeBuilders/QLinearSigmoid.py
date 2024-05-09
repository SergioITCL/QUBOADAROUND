import math
from typing import List, Optional, Tuple
from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.buildBaseNode import build_quantized_node
from itcl_quantization.json.specification import Node
from itcl_quantization.quantization.lut import ReducedLUT
from util.quantization import create_lut


class QLinearSigmoid(IBuilder):
    def build(self, node, graph) -> Tuple[List[Node], List[Node], Optional[str]]:
        input_idx, i_scale_idx, i_zp_idx, out_scale_idx, out_zp_idx = node.input

        output = node.output[0]
        return (
            [
                build_quantized_node(graph, None, i_scale_idx, i_zp_idx, "uint8"),
                build_quantized_node(graph, None, out_scale_idx, out_zp_idx, "uint8"),
            ],
            [],
            "",
        )


sigmoid = lambda x: 1 / (1 + math.exp(-x))


class SigmoidLUT(IBuilder):

    def __init__(self, depth: int, min_reduce: int):
        self.__depth = depth
        self.__min_reduce = min_reduce

    def build(self, node, graph) -> Tuple[List[Node], List[Node], Optional[str]]:
        """
        It takes the input scale and zero point, and the output scale and zero point, and creates a lookup
        table for the sigmoid function
        
        :param node: The node that we are currently processing
        :param graph: The graph that the node is in
        :return: The return value is a tuple of three elements. The first element is a list of nodes. The
        second element is an empty list. The third element is a string.
        """

        nodes, _, _ = QLinearSigmoid().build(node, graph)
        input_idx, i_scale_idx, i_zp_idx, out_scale_idx, out_zp_idx = node.input
        input_s, input_zp = nodes[0]["scale"], nodes[0]["zero_point"]
        output_s, output_zp = nodes[1]["scale"], nodes[1]["zero_point"]

        if None in [input_s, input_zp, output_s, output_zp]:
            raise ValueError(
                "SigmoidLUT requires quantized input and output, None values found"
            )

        lut = create_lut(sigmoid, input_s, input_zp, output_s, output_zp)  # type: ignore
        reduced_lut = ReducedLUT(lut, self.__depth, self.__min_reduce).serialize()
        
        return (
            [
                {
                    "LUT": {"LUT": lut, "offset": 0, "reduced_LUT": reduced_lut},
                    "scale": input_s,
                    "tensor": None,
                    "zero_point": input_zp,
                },
            ],
            [],
            "Reduced LUT for sigmoid activation function",
        )
