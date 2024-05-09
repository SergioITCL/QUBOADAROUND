from typing import Callable
import numpy as np
from itcl_quantizer.equalizers.adaround.iround_optimizer import IRoundOptimizer
from itcl_quantizer.tensor_extractor.abstract_layer import (
    AbstractLayer,
    QuantizationResult,
)

from itcl_quantizer.tensor_extractor.operator import Operator
from itcl_quantization import LayerIds

from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase


class Dequantize(AbstractLayer):
    """Dequantization Layer
    This layer creates a DEQUANTIZE Layer that dequantized the output from INT8 back to FP32

    Args:
        AbstractLayer (_type_): _description_
    """

    def quantize(self, input_result: QuantizationResult) -> QuantizationResult:

        if not input_result.operators:
            raise ValueError("Dequantize: No operators found")

        op_out = input_result.operators[-1].outputs[-1]

        if op_out is None:
            raise ValueError("Dequantize: No output node found")

        op = Operator[NodeTensorBase, NodeTensorBase](
            LayerIds.dequantize, LayerIds.dequantize, [op_out], [], None
        ).set_description("Dequantization Layer")

        return QuantizationResult(
            input_data=input_result.input_data,
            out_node=None,
            operators=[op],
            layer=self,
        )
