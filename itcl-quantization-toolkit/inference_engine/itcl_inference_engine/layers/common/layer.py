"""Layer Interface that wraps all of the neural network layer implementations under one interface type."""
from typing import Protocol

import numpy as np
from itcl_quantization.json.specification import Operator


class ILayer(Protocol):
    """Layer Interface that wraps all of the TFLITE layer implementations under one interface type."""

    @classmethod
    def from_model(cls, operator: Operator):
        """Alternative Constructor of the layer.

        Builds the layer given a JSON Operator

        Args:
            operator (Operator): Json Operator with all the input and output tensors.

        """
        raise NotImplementedError

    def infer(self, input_: np.ndarray) -> np.ndarray:
        """Infer Method
        This method infers the input of the layer with the corresponding operation.
        The shape of the np.ndarray must match the input shape of the layer.

        Args:
            input (np.ndarray): Input tensor to be inferred
        Returns:
            np.ndarray: The inferred output tensor
        """
        raise NotImplementedError
