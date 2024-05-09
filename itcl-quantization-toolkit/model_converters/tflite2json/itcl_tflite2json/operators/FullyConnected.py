from typing import Dict
import numpy as np
from tflite import Model, Operator
from itcl_tflite2json.operators.BaseOperator import BaseOperator
from itcl_quantization.json.specification import Operator as OperatorJson
import tensorflow.lite as tfl


class FullyConnectedOperator(BaseOperator):
    def __init__(
        self, interpreter: tfl.Interpreter, layer_name2idx: Dict[str, int], model: Model
    ):
        super().__init__(interpreter, layer_name2idx, model)

    def build(self, operator: Operator) -> OperatorJson:
        """Override method that transposes the weights matrix"""
        op = super().build(operator)
        tensor = op["inputs"][1]["tensor"]
        if tensor is None:
            raise ValueError("Weight Tensor not defined")

        weights = tensor["tensor"] or np.array([])
        
        weights = np.array(weights)
        tensor["shape"] = list(weights.shape)
        tensor["tensor"] = weights.tolist()

        return op
