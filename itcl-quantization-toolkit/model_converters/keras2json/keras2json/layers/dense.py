import numpy as np
from itcl_quantization.json.specification import Node, Operator
from keras2json.layers.i_layer import ILayer
from keras2json.layers._ids import FULLY_CONN
from tensorflow import keras


class Dense(ILayer):


    def __init__(self, layer: keras.layers.Layer):
        self._activation = layer.activation
        self._layer = layer
        self._name = layer.name
        weights = self._layer.get_weights()

        self._kernel = np.array(weights[0])

        if len(weights) > 1:
            self._bias: np.ndarray = np.array(weights[1])
        else:
            self._bias = np.zeros(len(self._kernel))

    def _dense(self) -> Operator :

        kernel: Node = {
            
            "scale": None,
            "zero_point": None,
            "tensor": {
                "dtype": self._kernel.dtype.name,
                "name": f"{self._name}/Kernel",
                "shape": list(self._kernel.T.shape),
                "tensor": self._kernel.T.tolist()
            },
            'LUT': None, 
        }

        bias: Node = {
            "scale": None,
            "zero_point": None,
            "tensor": {
                "dtype": self._bias.dtype.name,
                "name": f"{self._name}/Bias",
                "shape": list(self._bias.shape),
                "tensor": self._bias.tolist()
            },
            "LUT": None
        }

        return {
            "op_type": FULLY_CONN,
            'name': self._layer.name,
            'description': "Dense Layer",
            "inputs": [
                kernel,
                bias,
            ],
            "outputs": [],
            "attributes": None
        }

    def _activation_fn(self) -> Operator:
        
        name = str(self._activation.__name__)
        
        return {
            "op_type": name,
            "name": f"{name} activation function",
            "description": f"Dense Layer Activation Function {name}",
            "inputs": [],
            "outputs": [],
            "attributes": None
        }

    @classmethod
    def from_model(cls, layer: keras.layers.Layer) -> list[Operator]:
        instance = cls(layer)
        instance._activation_fn()
        instance._dense()
        return [instance._dense(), instance._activation_fn()]
