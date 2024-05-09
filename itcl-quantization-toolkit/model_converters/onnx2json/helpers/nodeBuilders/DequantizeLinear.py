from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.buildBaseNode import build_quantized_node


class DequantizeLinear(IBuilder):
    """Dequantize Linear Operator serializer

    Args:
        IBuilder (_type_): _description_
    """
    def build(self, node, graph):
        input, scale, zp = [x for x in node.input]
        output_key = node.output[0]

        return  [build_quantized_node(graph, None, scale, zp, "uint8")], [], "Casts a int8 value into float"