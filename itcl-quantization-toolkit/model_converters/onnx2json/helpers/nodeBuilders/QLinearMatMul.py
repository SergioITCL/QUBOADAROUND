from typing import List, Optional, Tuple
from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.buildBaseNode import build_quantized_node, build_base_node
from itcl_quantization.json.specification import Node


class QLinearMatMul(IBuilder):
    """ QLinearMatMul Operator serializer
    """
    def build(self, node, graph) -> Tuple[List[Node], List[Node], Optional[str]]:
        """
        It builds a quantized matrix multiplication node.
        
        :param node: The node that we are building
        :param graph: The graph that the node is in
        :return: The return value is a tuple of three elements.
        """

        (
            input_idx,
            i_scale_idx,
            i_zp_idx,
            a_idx,
            a_scale_idx,
            a_zp_idx,
            b_scale_idx,
            b_zp_idx,
        ) = node.input

        output = node.output[0]

        return (
            [
                build_quantized_node(graph, None, i_scale_idx, i_zp_idx, "uint8"),
                build_quantized_node(
                    graph, a_idx, a_scale_idx, a_zp_idx, "uint8", transpose=True
                ),
                build_quantized_node(graph, None, b_scale_idx, b_zp_idx, "uint8"),
            ],
            [],
            "Quantized Matrix Multiplication Node",
        )
