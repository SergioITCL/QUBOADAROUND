from typing import List, Optional, Tuple
from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.buildBaseNode import build_base_node
from itcl_quantization.json.specification import Node


class Softmax(IBuilder):
    def build(self, node, graph) -> Tuple[List[Node], List[Node], Optional[str]]:
        input_key = node.input[0]
        output_key = node.output[0]

        return (
            [build_base_node(graph, input_key)],
            [build_base_node(graph, output_key)],
            "",
        )
