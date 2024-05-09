import json
import os
import tensorflow.lite as tfl
from typing import Dict
from itcl_tflite2json.operators.BaseOperator import BaseOperator
from itcl_tflite2json.operators.FullyConnected import FullyConnectedOperator
from itcl_quantization.json.specification import (
    JsonNetwork,
    Operator as OperatorJson,
)
import tflite
from tflite import Operator, Model
from itcl_tflite2json.operators.SigmoidLUT import SigmoidLUT
from itcl_tflite2json.operators.TanhLUT import TanhLUT
from itcl_tflite2json.util.enum2str import layers2str
from pathlib import Path

def build_from_operator(
    interpreter: tfl.Interpreter,
    operator: Operator,
    layer_name2idx: Dict[str, int],
    model: Model,
) -> OperatorJson:
    """
    Builds a single operator from the given operator and model.
    """
    operator_type = model.OperatorCodes(operator.OpcodeIndex())
    operator_type_name = layers2str(operator_type.BuiltinCode())
    builder = BaseOperator(interpreter, layer_name2idx, model)

    if False and "FULLY_CONNECTED" == operator_type_name:
        # No extra builder necessary (at the moment)
        builder = FullyConnectedOperator(interpreter, layer_name2idx, model)
    elif "TANH" == operator_type_name:
        builder = TanhLUT(interpreter, layer_name2idx, model)
    elif "LOGISTIC" == operator_type_name:
        builder = SigmoidLUT(interpreter, layer_name2idx, model)
    return builder.build(operator)


def main():
    for file in Path("../models").glob("*.tflite"):
        
        #file = "../models/mnist/tflite/mnist_high_complexity_float32_tanh_quant_deq_in_out_layers.tflite"
        out = str(file).replace("tflite", "json")

        print("Loading model from:", file)
        interpreter = tfl.Interpreter(model_path=str(file))
        interpreter.allocate_tensors()
        layer_name2idx = {
            l["name"]: l["index"] for l in interpreter.get_tensor_details()
        }
        with open(file, "rb") as f:
            model_data = bytearray(f.read())
            model = tflite.Model.GetRootAsModel(model_data, 0)

            subgraph = model.Subgraphs(0)

            inputs_ids = [subgraph.Inputs(i) for i in range(subgraph.InputsLength())]
            outputs_ids = [subgraph.Outputs(i) for i in range(subgraph.OutputsLength())]

            inputs_operators = [
                BaseOperator.build_from_node(
                    interpreter, subgraph.Tensors(i), model, layer_name2idx
                )
                for i in inputs_ids
            ]
            outputs_operators = [
                BaseOperator.build_from_node(
                    interpreter, subgraph.Tensors(i), model, layer_name2idx
                )
                for i in outputs_ids
            ]

            network: JsonNetwork = {
                "ir_version": model.Version(),
                "engine": "TFLITE",
                "graph": [
                    build_from_operator(
                        interpreter, subgraph.Operators(i), layer_name2idx, model
                    )
                    for i in range(subgraph.OperatorsLength())
                ],
                "operators": subgraph.OperatorsLength(),
                "input": inputs_operators,
                "output": outputs_operators,
            }

            print("Writing to:", out)
            with open(out, "w") as f:
                json.dump(network, f)


if __name__ == "__main__":
    main()
