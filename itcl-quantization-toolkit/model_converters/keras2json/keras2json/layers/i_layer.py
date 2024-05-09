from tensorflow import keras
from itcl_quantization.json.specification import Operator, Node
import numpy as np

class ILayer():
    
    @classmethod
    def from_model(cls, layer: keras.layers.Layer) -> list[Operator]:
        raise NotImplementedError


def build_node(w: np.ndarray, name: str) -> Node:
    return {
        "scale": None,
        "zero_point": None,
        "tensor": {
            "dtype": w.dtype.name,
            "name": name,
            "shape": list(w.shape),
            "tensor": w.tolist()
        },
        "LUT": None
    }

    