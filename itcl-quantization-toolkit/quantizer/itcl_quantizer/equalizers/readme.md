# Equalizers

This module contains weight and quantization params equalizers that reduce
the overall noise of a quantized network.

All the changes that makes an equalizer should update the final network nodes and operators by reference. 

## Quantization Parameter Equalizer (param_equalizer)

This module contains the an equalizer that optimizes the network overall cost by tweaking the scale and zero point of multiple tensors.

## AdaRound Equalizer (adaround)

This module contains an adaptive rounding optimizer that individually selects if a tensor should be floored or ceiled. 

Tis module takes into account the layer dequantization noise (The difference between a float inference and a dequantized tensor output) or the network cost.


