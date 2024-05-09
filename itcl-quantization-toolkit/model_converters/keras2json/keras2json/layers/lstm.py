import numpy as np
from keras2json.layers.i_layer import ILayer
from tensorflow import keras
from itcl_quantization.json.specification import Operator, Node, Attribute
from keras2json.layers._ids import LSTM as LSTM_ID

_LAYER_TYPES = ["input", "forget_gate", "cell", "output"]


class LSTM(ILayer):
    @staticmethod
    def _weight(w: np.ndarray, name: str) -> Node:
        return {
            "scale": None,
            "zero_point": None,
            "tensor": {
                "dtype": w.dtype.name,
                "name": name,
                "shape": list(w.shape),
                "tensor": w.tolist(),
            },
            "LUT": None,
        }

    @classmethod
    def from_model(cls, layer: keras.layers.Layer) -> list[Operator]:
        weights: list[np.ndarray] = layer.get_weights()

        n_units: int = layer.get_config()["units"]

        W_full = weights[0].T
        U_full = weights[1].T
        B_full = weights[2]

        # print(U_full.shape)
        # print(B_full.reshape())
        #
        # W = np.split(W_full,4)
        # U = np.split(U_full,4)
        # B = np.split(B_full,4)

        W = np.array([[0]] * 4)
        U = np.array([[0]] * 4)
        B = np.array([[0]] * 4)
        nodes: list[Node] = []

        for i, w in enumerate(W):
            nodes.append(LSTM._weight(w, f"kernel/{_LAYER_TYPES[i]}"))

        for i, w in enumerate(U):
            nodes.append(LSTM._weight(w, f"recurrent_kernel/{_LAYER_TYPES[i]}"))

        for i, w in enumerate(B):
            nodes.append(LSTM._weight(w, f"bias/{_LAYER_TYPES[i]}"))

        for w, name in zip(
            [W_full, U_full, B_full], ["kernel", "recurrent_kernel", "bias"]
        ):
            nodes.append(LSTM._weight(w, f"{name}/complete"))

        name = layer.get_config()["name"]

        if len(layer._batch_input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, timesteps, input_dim), got {layer._batch_input_shape} in layer {name}"
            )

        window_size = layer._batch_input_shape[-2]

        return_seq = layer.get_config()["return_sequences"]

        attributes: dict[str, Attribute] = {
            "units": {"dtype": "int", "value": [n_units]},
            "return_sequences": {"dtype": "bool", "value": [bool(return_seq)]},
            "steps": {"dtype": "int", "value": [window_size]},
        }

        op: Operator = {
            "op_type": LSTM_ID,
            "name": "LSTM Layer",
            "description": "LSTM Layer",
            "inputs": nodes,
            "outputs": [],
            "attributes": attributes,
        }

        return [op]
