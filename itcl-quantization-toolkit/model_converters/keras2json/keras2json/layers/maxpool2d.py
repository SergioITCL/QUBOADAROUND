import numpy as np
from itcl_quantization.json.specification import Attribute, Node, Operator
from keras2json.layers._activations import get_activation_name
from keras2json.layers._ids import MAXPOOL2D
from keras2json.layers.i_layer import ILayer, build_node
from tensorflow import keras



class MaxPool2D(ILayer):
    @classmethod
    def from_model(cls, layer: keras.layers.Layer) -> list[Operator]:

        strides: tuple[int, int] = tuple(layer.strides)
        padding = 0 if "valid" in str(layer.padding).lower() else 1
        pool_size: tuple[int, int] = tuple(layer.pool_size)
        
        attributes: dict[str, Attribute] = {
            "strides": {"dtype": "int", "value": list(strides)},
            "pool_size": {"dtype": "int", "value": list(pool_size)},
            "padding": {"dtype": "int", "value": [padding]},
        }


        op: Operator = {
            "op_type": MAXPOOL2D,
            "name": "2D Max Pooling Layer",
            "description": layer.name,
            "inputs": [],
            "outputs": [],
            "attributes": attributes,
        }


        return [op]
