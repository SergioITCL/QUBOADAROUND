from copy import deepcopy
from typing import Callable, Dict, Tuple, Any

import numpy as np
from itcl_quantization import Quantization
from itcl_quantization.json.specification import LUT
from itcl_quantization.quantization.lut import ReducedLUT
from numpy.typing import NDArray

from itcl_quantizer.quantizer.distributions.distribution import Distribution
from itcl_quantizer.tensor_extractor.operator import Operator
from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase as Node
from itcl_quantizer.util.typing import NPGeneric

# A list of function names that can be saved as Look Up Tables
LUT_FUNCTIONS = ["TANH", "SIGMOID"]

# Functions that shouldn't be included in the final model as they are not required.
SKIP_FUNCTIONS = ["LINEAR", "SOFTMAX", "RELU", "RELU6"]

# A dict with overriden functions
OVERRIDE_FUNCTIONS: Dict[str, Callable[[Any], Any]] = {
    "SOFTMAX": lambda x: x,
}

ACTIVATION_PARAMS = {
    #    "SIGMOID": (1/256, -128),
    #    "TANH": (1/128, 0),
}


def quantize_activation_node(
    fn: Callable[[NDArray[NPGeneric]], NPGeneric],
    activation_name: str,
    input_data: NDArray[NPGeneric],
    quantization: Quantization,
    input_node: Node,
    trim_dist: Callable[[Distribution], Distribution] = lambda x: x,
    out_params: Tuple[float, int] | None = None,
    skip_lut: bool = False,
) -> Tuple[Operator | None, NDArray[NPGeneric]]:
    """
    This function takes an activation function, an input data, and a quantization object, and returns
    an operator and the activated output data

    Args:
      fn (Callable[[np.ndarray], np.ndarray]): Callable[[np.ndarray], np.ndarray]
      activation_name (str): str,
      input_data (np.ndarray): The input data to the activation function, fp32
      quantization (Quantization): Quantization object to quantize the distribution
      input_node (Node): The input node to the activation function.
      trim_dist (Callable[[Distribution], Distribution]): Callable[[Distribution], Distribution], distribution trimmer
        out_params (Tuple[float, int] | None): Tuple[float, int] | None, output activation params
    """
    activation_name = activation_name.upper()

    if activation_name in SKIP_FUNCTIONS:
        return None, input_data

    # Override the function if possible
    fn = OVERRIDE_FUNCTIONS.get(activation_name, fn)

    # infer the float input_data
    activated_data = fn(input_data)

    # Create and Trim the distribution
    activation_dist = Distribution(activated_data)

    activation_dist = trim_dist(activation_dist)

    # if out_params were specified, override them
    if out_params is not None:
        out_s, out_zp = out_params
    else:  # Calculate the out params on quantization time
        out_s, out_zp = ACTIVATION_PARAMS.get(
            activation_name, activation_dist.quantize(quantization)
        )

    # Create and save the LUT (if possible)
    if activation_name in LUT_FUNCTIONS and not skip_lut:
        activation_name += "LUT"
        input_node.with_lut(fn, out_s, out_zp)

    # Create the output node
    output_node = (
        Node(out_s, out_zp, f"{activation_name}/OUT", quantization)
        .with_tensor(activated_data)
        .exclude_batch_dimension()
        .exclude_tensor()
    )

    operator = Operator(
        activation_name, activation_name, [input_node], [output_node], fn
    )
    return operator, np.array(activated_data)
