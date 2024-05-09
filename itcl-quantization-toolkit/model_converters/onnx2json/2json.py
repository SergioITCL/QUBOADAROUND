import json
import os
from typing import List
import onnx
from helpers.graph import OnnxGraph
from helpers.nodeBuilders.buildBaseNode import layer_info_to_tensor
from itcl_quantization.json.specification import JsonNetwork, Node


def main():
    model = onnx.load("../../models/mnist/uint8/mnist_tanh_uint8.onnx")


    for model in os.listdir("../../models/mnist/uint8"):
        input = os.path.join("../../models/mnist/uint8", model)
        output = input.replace("onnx", "json")

        with open(output, "w") as f:
            model = onnx.load(input)
            json.dump(model2json(model), f)



def model2json(model) -> JsonNetwork:
    graph = OnnxGraph(model)

    return {
        "ir_version": model.ir_version,
        "engine": "ONNX",
        "graph": graph.build_graph(),
        "input": [{"zero_point": None, "scale": None, "tensor":  layer_info_to_tensor(_in)} for _in in model.graph.input],
        "output": [{"zero_point": None, "scale": None, "tensor":  layer_info_to_tensor(_out)} for _out in model.graph.input],
    }


if __name__ == "__main__":
    main()
