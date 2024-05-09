from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.operator import IOperator
from itcl_inference_engine.layers.common.sigmoid import SigmoidLUT
from itcl_inference_engine.layers.common.tanh import TanhLUT
from itcl_inference_engine.layers.onnx.dequantize_linear import \
    DequantizeLinear
from itcl_inference_engine.layers.onnx.q_linear_add import QLinearAdd
from itcl_inference_engine.layers.onnx.q_linear_mmul import QLinearMatMul
from itcl_inference_engine.layers.onnx.q_linear_sigmoid import QLinearSigmoid
from itcl_inference_engine.layers.onnx.quantize_linear import QuantizeLinear
from itcl_inference_engine.layers.onnx.softmax import Softmax
from itcl_inference_engine.layers.onnx.tanh import Tanh


def build_operator(operator: Operator) -> IOperator:
    """Given an operator and a model with the the operators and initializers (Tensors)
    Builds the python version of the operator

    Args:
        operator (_type_): _description_
        model (_type_): _description_

    Raises:
        ValueError: If the operator is not compatible with the Inference Engine

    Returns:
        IOperator: An operator with all its quantization parameters and tensors loaded.
    """

    opt = operator["op_type"]

    if opt == "QLinearAdd":
        return QLinearAdd.from_model(
            operator,
        )

    elif opt == "QLinearMatMul":
        return QLinearMatMul.from_model(operator)
    elif opt == "QuantizeLinear":
        return QuantizeLinear.from_model(operator)
    elif opt == "QLinearSigmoid":
        return QLinearSigmoid.from_model(operator)
    elif opt == "SigmoidLUT":
        return SigmoidLUT.from_model(operator)
    elif opt == "DequantizeLinear":
        return DequantizeLinear.from_model(operator)
    elif opt == "Softmax":
        return Softmax()
    elif opt == "Tanh":
        return Tanh()
    elif opt == "TanhLUT":
        return TanhLUT.from_model(operator)
    else:
        raise ValueError(f"{opt} is not a valid operator")
