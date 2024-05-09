from itcl_quantization import Quantization
import numpy as np

from itcl_quantizer.quantizer.distributions.distribution import Distribution
from itcl_quantizer.tensor_extractor.abstract_layer import (
    AbstractLayer,
    QuantizationResult,
)
from itcl_quantizer.tensor_extractor.operator import Operator
from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase, NodeTensorTensor
from itcl_quantization import LayerIds


class KerasInput(AbstractLayer):
    def __init__(self, quantization: Quantization):
        self.__Q = quantization

    def quantize(self, q_result: QuantizationResult) -> QuantizationResult:
        float_input = q_result.input_data
        input_dist = Distribution(float_input)

        scale, zp = input_dist.quantize(self.__Q)

        node = (
            NodeTensorBase(scale, zp, "Input Quant", self.__Q)
            .with_tensor(float_input)
            .exclude_batch_dimension()
            .exclude_tensor()
        )

        operator = Operator[NodeTensorBase, NodeTensorBase](
            LayerIds.quantize, LayerIds.quantize, [], [node], None
        )

        return QuantizationResult(
            input_data=float_input, out_node=node, operators=[operator], layer=self
        )
