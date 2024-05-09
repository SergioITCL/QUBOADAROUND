from itcl_quantization.json.specification import Operator
from keras2json.layers.i_layer import ILayer
from tensorflow import keras


class Flatten(ILayer):
    @classmethod
    def from_model(cls, layer: keras.layers.Layer) -> list[Operator]:

        op: Operator = {
            "op_type": "FLATTEN",
            "name": "Flatten Layer",
            "description": "Flatten Layer",
            "inputs": [],
            "outputs": [],
            "attributes": None,
        }

        return [op]
