from dataclasses import dataclass
from typing import Any, List, Optional, Union

from helpers.nodeBuilders.DequantizeLinear import DequantizeLinear
from helpers.nodeBuilders.IBuilder import IBuilder
from helpers.nodeBuilders.QLinearAdd import QLinearAdd
from helpers.nodeBuilders.QLinearMatMul import QLinearMatMul
from helpers.nodeBuilders.QLinearSigmoid import QLinearSigmoid, SigmoidLUT
from helpers.nodeBuilders.QuantizeLinear import QuantizeLinear
from helpers.nodeBuilders.Softmax import Softmax
from helpers.nodeBuilders.Tanh import Tanh, TanhLUT
from itcl_quantization.json.specification import Operator
from util.settings import settings

@dataclass
class BuildOperatorCandidate:
    """Dataclass that stores a candidate for building an operator.


    """

    op_type: str # Operator Type, can be an advanced operator or a simple operator


    operators: Union[List[Any], Any] # A list of operators. Simple operators are stored as the only operator in the list. 
    

def buildOperator(node: BuildOperatorCandidate, model) -> Operator:
    """Builds an operator."""
    op_type = node.op_type
    operators = node.operators

    builder: Optional[IBuilder] = None
    lut_settings = settings["settings"]["lut"]
    if op_type == "Tanh":
        builder = Tanh()
    elif op_type == "TanhLUT":
        builder = TanhLUT()
    elif op_type == "Softmax":
        builder = Softmax()
    elif op_type == "QuantizeLinear":
        builder = QuantizeLinear()
    elif op_type == "DequantizeLinear":
        builder = DequantizeLinear()
    elif op_type == "QLinearAdd":
        builder = QLinearAdd()
    elif op_type == "QLinearMatMul":
        builder = QLinearMatMul()
    elif op_type == "QLinearSigmoid":
        op_type = "SigmoidLUT"
        builder = SigmoidLUT(lut_settings["lut_depth"], lut_settings["min_removal"])

    if builder is None:
        raise ValueError(f"{op_type} is not compatible")

    inputs, outputs, description = builder.build(operators, model)

    return {
        "op_type": op_type,
        "name": op_type,
        "inputs": inputs,
        "outputs": outputs,
        "description": description,
    }
