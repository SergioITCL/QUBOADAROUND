import json
import time
from typing import List, Protocol

import numpy as np
from itcl_quantization.json.specification import JsonNetwork, Operator

from itcl_inference_engine import logger
from itcl_inference_engine.layers.common.layer import ILayer
from itcl_inference_engine.layers.onnx.build_operator import build_operator
from itcl_inference_engine.layers.tflite.build_layer import build_layer
from itcl_inference_engine.util.quantization import to_int8


class Network(Protocol):
    def infer(
        self,
        input_: np.ndarray,
        isInputQuant: bool = False,
        verbose: bool = False,
        isBatch: bool = False,
    ) -> np.ndarray:
        ...

    schema: JsonNetwork


class SequentialNetwork(Network):
    """Network Class"""

    def __init__(
        self,
        net: JsonNetwork,
    ):
        """Sequential Network Constructor that builds a network from a json file.

        Args:
            model_path (str): Path to .json file

        Raises:
            ValueError: If the input and output tensors are not in the file.
        """
        self.schema = net
        self.__engine = self.schema["engine"]
        self._intermediate_results: list[np.ndarray] = []

        logger.info(f"Generated At: {self.schema.get('generated_at')}")

        # Get the first input and output tensors
        input_node, output_node = self.schema["input"][0], self.schema["output"][0]

        if input_node is None or output_node is None:
            raise ValueError(
                "Input and output tensors are required, please rebuild the .json model file."
            )

        # Get the input and output zero point.
        self.input_scale, self.input_zero_point = (
            input_node["scale"],
            input_node["zero_point"],
        )

        self.output_scale, self.output_zero_point = (
            output_node["scale"],
            output_node["zero_point"],
        )
        self.has_quant_layers = "QUANT" in self.schema["graph"][0]["name"].upper()
        if self.has_quant_layers:
            logger.info("Quantized model detected with quantization layers")

        self.layers: List[ILayer] = self.__build_network()

    @classmethod
    def from_json_file(cls, model_path: str):

        with open(model_path, encoding="utf-8") as f:
            net: JsonNetwork = json.load(f)
            return cls(net)

    def infer(
        self, input_: np.ndarray, isInputQuant=False, verbose=False, isBatch=False
    ) -> np.ndarray:
        """Inference Method
        Infers an input tensor with the neural network.

        Args:
            input (np.ndarray): Input Tensor (1d, 2d, etc)
            isInputQuant (bool, optional): Boolean that specifies if the current input is quantized in INT8. Defaults to False.
            verbose (bool, optional): If True, the network will logger.info each layer output tensor. Defaults to False.

        Returns:
            np.ndarray: Output Layer result. If isInputQuant is false, the result will be dequantized
        """
        self._intermediate_results = []
        input_ = np.array(input_)
        if isBatch:
            logger.warning("\nSUPPORT FOR `isBatch: bool` PARAMETER IS DEPRECATED!\n")
        if not self.has_quant_layers:
            logger.warning(
                "\n\n\nSUPPORT FOR MODELS WITHOUT QUANTIZATION & DEQUANTIZATION LAYERS IS DEPRECATED\n\n\n"
            )
            input_ = to_int8(input_, self.input_zero_point or 0, self.input_scale or 1)

        for i, layer in enumerate(self.layers):

            input_ = res = layer.infer(input_)
            self._intermediate_results.append(input_)
            if verbose:
                logger.info(f"Layer {i}: {res}")

        # if not self.has_quant_layers:
        #    return from_int8(input, self.output_zero_point or 0, self.output_scale or 1)

        return input_

    def get_engine(self):
        engine_name = self.__engine.upper()
        logger.info(f"Loading {engine_name} engine")
        if "TFLITE" == engine_name:
            return self.build_layer_tflite
        elif "ONNX" == engine_name:
            return self.build_layer_onnx
        else:
            raise ValueError(f"Engine {engine_name} not supported")

    def __build_network(self) -> List[ILayer]:
        """Helper Method that build the sequential network.

        Returns:
            List[ILayer]: List of layers
        """
        schema = self.schema
        layers: List[ILayer] = []
        build_layer = self.get_engine()

        # Get and build all the nodes
        for layer in schema["graph"]:
            logger.info(layer["name"])
            layers.append(build_layer(layer))

        return layers

    @staticmethod
    def build_layer_tflite(layer: Operator) -> ILayer:
        """Helper static method that build

        Args:
            layer (Operator): A layer from the json file

        Raises:
            ValueError: If the layer is not supported.

        Returns:
            ILayer: An instance of the layer
        """
        return build_layer(layer)

    @staticmethod
    def build_layer_onnx(layer: Operator) -> ILayer:
        """Helper static method that build

        Args:
            layer (Operator): A layer from the json file

        Raises:
            ValueError: If the layer is not supported.

        Returns:
            ILayer: An instance of the layer
        """
        return build_operator(layer)

    def __str__(self) -> str:
        return json.dumps(self.schema)
