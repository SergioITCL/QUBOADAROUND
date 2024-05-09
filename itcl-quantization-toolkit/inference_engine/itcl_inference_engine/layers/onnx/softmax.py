import numpy as np

from itcl_inference_engine.layers.common.operator import IOperator


class Softmax(IOperator):
    """The operator computes normalized input as a softmax normalization"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_model(cls, operator):
        """Buils the operator from a given node

        Args:
            operator (Operator): Json Operator Dict

        Returns:
            Softmax Instance
        """
        return cls()

    def infer(self, input_: np.ndarray) -> np.ndarray:
        """Normalize the input

        Softmax(input) = Exp(input) / ReduceSum(Exp(input))
        Args:
            input (np.ndarray): Input Tensor as float32 (Dequantized)

        Returns:
            np.ndarray: Float32 output tensor (Each value should be between 0 and 1)
        """

        return np.exp(input_) / np.sum(np.exp(input_))
