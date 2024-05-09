import numpy as np

from itcl_inference_engine.layers.common.operator import IOperator


class Tanh(IOperator):
    """Tanh Operator
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_model(cls, operator):
        return super().from_model(operator)

    def infer(self, input_: np.ndarray) -> np.ndarray:
        """Calculates the tanh for each element of the input array.

        Args:
            input (np.ndarray): an np.float32 tensor (Dequantized)

        Returns:
            np.ndarray fp32: Dequantized tensor Ouput
        """
        return np.array(np.tanh(input_))
