from typing import Optional, Union
from itcl_quantization.json.specification import Node, Tensor, TensorW
import numpy as np


def build_quantized_node(
    model,
    tensor_id: Union[str, None],
    scale_id: str,
    zerop_id: str,
    dtype: str,
    transpose: bool = False,
) -> Node:
    """
    It takes a tensor ID, a scale ID, a zero point ID, and a data type, and returns a dictionary
    containing the scale, zero point, and tensor

    :param model: The model object
    :param tensor_id: The name of the tensor to be quantized
    :type tensor_id: Union[str, None]
    :param scale_id: The name of the scale parameter
    :type scale_id: str
    :param zerop_id: The name of the zero point tensor
    :type zerop_id: str
    :param dtype: The data type of the tensor
    :type dtype: str
    :param transpose: Whether to transpose the tensor, defaults to False
    :type transpose: bool (optional)
    :return: A dictionary with the keys: scale, zero_point, and tensor.
    """
    initializers = model.graph.initializer
    id2idx = {init.name: i for i, init in enumerate(initializers)}

    scale_init = initializers[id2idx[scale_id]]
    zerop_init = initializers[id2idx[zerop_id]]
    
    # If there is a tensor, fill out the tensor details
    if tensor_id is not None:
        input_init = initializers[id2idx[tensor_id]]
        input_arr = np.frombuffer(input_init.raw_data, dtype=dtype)
        input_arr = np.reshape(input_arr, input_init.dims)

        if transpose:
            input_arr = np.transpose(input_arr)

        tensor: Optional[Tensor] = {"shape": list(input_arr.shape), "dtype": dtype, "tensor": list(input_arr.tolist())}  # type: ignore
    else:
        tensor = None

    return {
        "scale": float(scale_init.float_data[0]),
        "zero_point": int(zerop_init.int32_data[0]),
        "tensor": tensor,
    }


def layer_info_to_tensor(layer_info) -> Tensor:
        
    return {
        "name": layer_info.name,
        "dtype": "?",
        "shape": layer_info.type.tensor_type.shape.dim[1].dim_value,
        "tensor": None,
    }


def build_base_node(model, tensor_id) -> Node:
    """
    Builds a node from a tensor ID.

    Args:
        model (_type_): _description_
        tensor_id (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        Node: _description_
    """
    id2idx = {init.name: i for i, init in enumerate(model.graph.value_info)}
    idx = id2idx.get(tensor_id)
    
    if idx: # The tensor id is expected to be in the model
        layer_info = model.graph.value_info[idx]

    # Some tensors, like the input and output tensors, are not in the model body
    elif tensor_id == model.graph.input[0].name:
        # Search in the input tensors
        layer_info = model.graph.input[0]
    elif tensor_id == model.graph.output[0].name:
        # Search in the output tesnsors
        layer_info = model.graph.output[0]
    else:
        raise ValueError(f"No layer named {tensor_id}")

    return {
        "tensor": layer_info_to_tensor(layer_info),
        "scale": None,
        "zero_point": None,
    }
