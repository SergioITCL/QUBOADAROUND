from typing import List, Optional, Tuple
from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.buildBaseNode import build_base_node, build_quantized_node
from itcl_quantization.json.specification import Node


class QLinearAdd(IBuilder):
    """QLinearAdd Operator serializer
    """
    def build(self, node, model) -> Tuple[List[Node], List[Node], Optional[str]]:
        """
        It takes the input node, and returns a list of nodes that are the inputs to the input node, a list
        of nodes that are the outputs of the input node, and a string that is the name of the input node
        
        :param node: The node that we are building
        :param model: The model object
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
                build_quantized_node(model, None, i_scale_idx, i_zp_idx, "uint8"),
                build_quantized_node(model, a_idx, a_scale_idx, a_zp_idx, "uint8"),
                build_quantized_node(model, None, b_scale_idx, b_zp_idx, "uint8"),
            ],
            [],
            "",
        )
