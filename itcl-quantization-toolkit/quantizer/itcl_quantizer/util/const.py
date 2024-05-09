
from dataclasses import dataclass


MODULE = "itcl_quantizer"


@dataclass
class ACTIVATION_NAMES:
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    NONE = "none"