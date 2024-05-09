import numpy as np
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.layer import ILayer


class SoftMax(ILayer):
    """
    SOFTMAX
    Input 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
    Output 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
        restriction: (scale, zero_point) = (1.0 / 256.0, -128)
    """

    def __init__(self, input_scale, input_zerop) -> None:
        super().__init__()

    @classmethod
    def from_model(cls, operator: Operator):
        return cls(operator["inputs"][0]["scale"], operator["inputs"][0]["zero_point"])

    def infer(self, input_: np.ndarray):
        return input_
