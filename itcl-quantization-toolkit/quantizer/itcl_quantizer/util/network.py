from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from itcl_inference_engine.network.sequential import SequentialNetwork
from itcl_quantization.json.specification import JsonNetwork

from itcl_quantizer.interfaces.serializable import ISerializable

if TYPE_CHECKING:
    from itcl_quantizer.tensor_extractor.abstract_layer import QuantizationResult
    from itcl_quantizer.tensor_extractor.operator import Operator
    from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase


class Network(ISerializable):
    """Class that stores a quantized network

    All the tensors and operators are references of the QuantizationResult constructor list

    Implements:
        ISerializable
    """

    def __init__(self, quant_results: list[QuantizationResult]):
        """
        Args:
            quant_results (list[QuantizationResult])
        """
        self._quant_res = quant_results
        self._layers: list[Operator[NodeTensorBase, NodeTensorBase]] = []

        for res in quant_results:
            for op in res.operators or []:
                if op is not None:
                    self._layers.append(op)

    def as_quant_results(self) -> list[QuantizationResult]:
        """
            Returns the network as quantization results.
            Returns the same list used to build the class.

        Returns:
            list[QuantizationResult]:
        """
        return self._quant_res

    def as_operators(self) -> list[Operator[NodeTensorBase, NodeTensorBase]]:
        """Returns the network as a sequential list of Operators

        Returns:
            list[Operator[NodeTensorBase, NodeTensorBase]]: _description_
        """
        return self._layers

    def as_json(self) -> JsonNetwork:
        """Returns the list as a serializable dictionary.

        Returns:
            JsonNetwork
        """

        return {  # type: ignore
            "engine": "tflite",
            "generated_at": str(datetime.now()),
            "ir_version": "0.1",
            "operators": len(self._layers),
            "graph": [l.as_json() for l in self._layers],
            "input": [
                {
                    "scale": None,
                    "zero_point": None,
                    "tensor": {
                        "shape": [1, 784],
                        "dtype": "FLOAT32",
                        "name": "INPUT",
                        "tensor": None,
                    },
                }
            ],
            "output": [
                {
                    "scale": None,
                    "zero_point": None,
                    "tensor": {
                        "shape": [1, 10],
                        "dtype": "FLOAT32",
                        "name": "OUTPUT",
                        "tensor": None,
                    },
                }
            ],
        }

    def as_sequential_network(self) -> SequentialNetwork:
        """Return the network as a sequential network that can be infered.

        Returns:
            SequentialNetwork: Instance of SequentialNetwork.
        """
        net = self.as_json()
        return SequentialNetwork(net)
