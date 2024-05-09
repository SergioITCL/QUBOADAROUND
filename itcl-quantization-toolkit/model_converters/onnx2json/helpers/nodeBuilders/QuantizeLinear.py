from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.buildBaseNode import build_quantized_node

class QuantizeLinear(IBuilder):
    """Quantize Linear Operator serializer"""    
    def build(self, node, graph):
        input, scale, zp = [x for x in node.input]
        output_key = node.output[0]

        return  [build_quantized_node(graph, None, scale, zp, "uint8")], [], "Casts a float value into uint8"