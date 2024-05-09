import json
import logging
from typing import Callable, cast

import numpy as np
from itcl_inference_engine.network.sequential import Network as NetworkIE
from itcl_quantization import Quantization
from tensorflow import keras

from itcl_quantizer.config.models.keras import KerasDenseCfg, QuantizerCfg
from itcl_quantizer.equalizers.adaround import AdaroundNet
from itcl_quantizer.equalizers.param_equalizer.bfs_equalizer import BFSEqualizer
from itcl_quantizer.equalizers.param_equalizer.param_equalizer import (
    ParamEqualizerFullNet,
    ParamEqualizerNet,
)
from itcl_quantizer.tensor_extractor.abstract_layer import QuantizationResult
from itcl_quantizer.tensor_extractor.keras.layers import (
    Dequantize,
    KerasInput,
    KerasLSTM,
    KerasDense,
)
from itcl_quantizer.tensor_extractor.keras.utils import CheckType
from itcl_quantizer.util.network import Network


def _quantize(
    layers, representative_input: np.ndarray, cfg: QuantizerCfg
) -> list[QuantizationResult]:
    """Quantize all the keras layers

    Args:
        layers (list): A list of Keras Layers
        representative_input (np.ndarray): Representative Input Dataset

    Returns:
        list[QuantizationResult]: _description_
    """
    keras_input = KerasInput(Quantization(cfg.quantize.dtype))

    sequential_quantized = [
        keras_input.quantize(
            QuantizationResult(representative_input, None, None, keras_input)
        )
    ]

    for i, layer in enumerate(layers):
        previous = sequential_quantized[-1]
        layer_q = get_layer_quantizer(layer, cfg)

        if layer_q is None:
            continue

        quantized_op = layer_q.quantize(previous)
        sequential_quantized.append(quantized_op)

    # add the Dequantize Layer:
    sequential_quantized.append(Dequantize().quantize(sequential_quantized[-1]))

    return sequential_quantized


def build(
    model_path: str,
    output_path: str,
    representative_input: np.ndarray,
    loss_fn: Callable[[NetworkIE], float] | None = None,
    cfg: QuantizerCfg = QuantizerCfg(),
) -> Network:
    """
    It takes a Keras model and a representative input, and returns a JSON file that can be used to
    generate a TFLite model

    Args:
      model_path (str): The path to the model you want to convert.
      representative_input (np.ndarray): This is a sample input that the model will see. It's used to
        calculate the quantization parameters.
      loss_fn(function): The loss function calculates the loss of the network. It will be used with the network_equalizers
      to fine tune the network.
    """
    model = keras.models.load_model(model_path)
    print(f"Loaded model {model_path} with {len(model.layers)} layers")
    # get layers:
    layers = model.layers

    quantized = _quantize(layers, representative_input, cfg)
    network = Network(quantized)

    if loss_fn is not None:
        # ParamEqualizerNet(network, loss_fn, lambda: ParamEqAnnealer(0.1, 0.1, steps=2000)).equalize()

        if param_cfg := cfg.param_equalizer:
            # TODO: add a way to choose the equalizer (sequential or full net)
            ParamEqualizerFullNet(
                network, loss_fn, lambda: BFSEqualizer(param_cfg.max_retries)
            ).equalize()

        if adanet_cfg := cfg.ada_round_net:
            AdaroundNet(network, loss_fn, adanet_cfg.build).round()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(network.as_json(), f)

    return network


def get_layer_quantizer(layer, cfg: QuantizerCfg):
    name: str = layer.name

    if CheckType.is_dense(layer):
        dense = cast(KerasDenseCfg, cfg.specific_layers.get(name, cfg.dense))
        return KerasDense(
            layer,
            activation_dtype=dense.activation_dtype,
            bias_add_dtype=dense.bias_add_dtype,
            bias_dtype=dense.bias_dtype,
            kernel_dtype=dense.kernel_dtype,
            adaround_optimizer=dense.adaround_optimizer.build()
            if dense.adaround_optimizer
            else None,
            kernel_symmetric=dense.kernel_symmetric,
        )

    if CheckType.is_LSTM(layer):
        return KerasLSTM(layer)

    if CheckType.is_skippable(layer):
        return None

    if CheckType.is_input(layer):
        return None

    raise ValueError(f"Unknown layer type: {layer}")
