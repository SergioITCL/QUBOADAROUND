import json
import logging
from datetime import datetime

from itcl_quantization.json.specification import JsonNetwork, Operator
from tensorflow import keras

from keras2json import layers


def _from_layer(keras_layer) -> list[Operator]:
    name = str(keras_layer.__class__.__name__).upper()

    if "DENSE" in name:
        builder = layers.Dense
    elif "LSTM" in name:
        builder = layers.LSTM
    elif "CONV2D" in name:
        builder = layers.Conv2d
    elif "FLATTEN" in name:
        builder = layers.Flatten
    elif "MAXPOOLING2D" in name:
        builder = layers.MaxPool2D
    elif name in ["INPUT", "OUTPUT", "INPUTLAYER", "DROPOUT"]:
        return []
    else:
        raise ValueError(f"Layer {name} cannot be converted")

    return builder.from_model(keras_layer)


def keras2json(model: keras.Model) -> JsonNetwork:
    """Converts a keras model into a json network

    Args:
        model (keras.Model): Keras Model

    Returns:
        JsonNetwork: A dict representing the network
    """

    operators: list[Operator] = []
    for l in model.layers:
        operators.extend(_from_layer(l))

    network: JsonNetwork = {
        "ir_version": "v1",
        "engine": "keras",
        "generated_at": datetime.now().isoformat(),
        "graph": operators,
        "operators": len(operators),
        "input": [],
        "output": [],
    }

    return network
