import numpy as np
from itcl_quantization.json.specification import Attribute, Node, Operator
from keras2json.layers._activations import get_activation_name
from keras2json.layers._ids import CONV2D
from keras2json.layers.i_layer import ILayer, build_node
from tensorflow import keras



class Conv2d(ILayer):
    @classmethod
    def from_model(cls, layer: keras.layers.Layer) -> list[Operator]:
        weights: list[np.ndarray] = layer.get_weights()

        W = weights[0]
        B = weights[1]
        strides: tuple[int, int] = tuple(layer.strides)
        padding = 0 if "valid" in str(layer.padding).lower() else 1

        nodes: list[Node] = [build_node(W, "kernel"), build_node(B, "bias")]

        attributes: dict[str, Attribute] = {
            "strides": {"dtype": "int", "value": list(strides)},
            "padding": {"dtype": "int", "value": [padding]},
        }

        op: Operator = {
            "op_type": CONV2D,
            "name": "2D Convolution Layer",
            "description": "2D Convolution Layer",
            "inputs": nodes,
            "outputs": [],
            "attributes": attributes,
        }

        fn_name = get_activation_name(layer.activation)

        activation: Operator = {
            "op_type": f"{fn_name}_F",
            "name": f"{fn_name} Float Activation",
            "description": f"{fn_name} Float Activation",
            "inputs": [],
            "outputs": [],
            "attributes": {},
        }

        return [op, activation]
