from typing import Dict, List
import tensorflow.lite as tfl
from tflite import Operator, Model, Tensor
from itcl_quantization.json.specification import (
    Operator as OperatorJson,
    Node as NodeJson,
)
from itcl_tflite2json.util.enum2str import layers2str, dtype2str
import numpy as np


class BaseOperator:
    """Base Operator Class
    Builds an operator from a Tensor and a Model
    """

    def __init__(
        self, interpreter: tfl.Interpreter, layer_name2idx: Dict[str, int], model: Model
    ):
        """
        It creates a new class called TFLiteModel.
        
        :param interpreter: tfl.Interpreter
        :type interpreter: tfl.Interpreter
        :param layer_name2idx: A dictionary that maps the name of a layer to its index in the interpreter
        :type layer_name2idx: Dict[str, int]
        :param model: The model that you want to use to predict
        :type model: Model
        """
        self.__interpreter = interpreter
        self.__layer_name2idx = layer_name2idx
        self.__model = model

    def build(self, operator: Operator) -> OperatorJson:
        """Build Method"""
        subgraph = self.__model.Subgraphs(0)
        operator_type = self.__model.OperatorCodes(operator.OpcodeIndex())
        operator_type_name = layers2str(operator_type.BuiltinCode())
        inputs: List[Tensor] = [subgraph.Tensors(i) for i in operator.InputsAsNumpy()]  # type: ignore
        outputs: List[Tensor] = [
            subgraph.Tensors(i) for i in operator.OutputsAsNumpy()
        ]  # type: ignore

        return {
            "description": operator_type_name or "",
            "op_type": operator_type_name or "",
            "name": operator_type_name or "",
            "inputs": [
                self.build_from_node(
                    self.__interpreter, i, self.__model, self.__layer_name2idx
                )
                for i in inputs
            ],
            "outputs": [
                self.build_from_node(
                    self.__interpreter, o, self.__model, self.__layer_name2idx
                )
                for o in outputs
            ],
        }

    @staticmethod

    def build_from_node(
        interpreter: tfl.Interpreter,  # type: ignore
        node: Tensor,
        model: Model,
        layer_name2idx: Dict[str, int],
    ) -> NodeJson:
        """
        It takes a tflite model, and returns a dictionary of the model's nodes, with the node's name, shape,
        dtype, and tensor values
        
        :param interpreter: tfl.Interpreter
        :type interpreter: tfl.Interpreter
        :param node: Tensor,
        :type node: Tensor
        :param model: The model object
        :type model: Model
        :param layer_name2idx: A dictionary that maps the layer name to the index of the layer in the
        interpreter
        :type layer_name2idx: Dict[str, int]
        :return: A dictionary with the following keys:
        """
        node_name = str(node.Name().decode("utf8") or "?")

        # Get the saved tensors in file
        numpy_tensor = model.Buffers(node.Buffer()).DataAsNumpy()

        # If there are no tensors: skip
        if isinstance(numpy_tensor, int):
            numpy_tensor = None
        else:
            # Get the tensor data from the tflite interpreter
            numpy_tensor = interpreter.get_tensor(layer_name2idx.get(node_name, ""))

        q = node.Quantization()
        # If there ar no quantization params: skip
        if q is not None:

            s, zp = q.ScaleAsNumpy() or 0, q.ZeroPointAsNumpy() or 0

            if isinstance(s, np.ndarray) or isinstance(zp, np.ndarray):
                scale, zero_point = float(s), int(zp)
            else:
                scale, zero_point = None, None
        else:
            scale = None
            zero_point = None
        return {
            "scale": scale,
            "zero_point": zero_point,
            "tensor": {
                "shape": list(numpy_tensor.shape if numpy_tensor is not None else list(node.ShapeAsNumpy().tolist())),  # type: ignore
                "dtype": dtype2str(node.Type()),
                "name": node_name,
                "tensor": numpy_tensor.tolist() if numpy_tensor is not None else None,
            },
        }
