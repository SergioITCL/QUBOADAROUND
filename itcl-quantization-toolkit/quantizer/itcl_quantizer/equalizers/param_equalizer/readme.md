# Quantization Parameter Equalizer

This is a layer equalizer that tweaks the quantization parameters (scale and zero point) to minimize the cost function


## How can a layer node be equalized?

Equalizable layers should override the ```param_equalizer()``` method, this method updates the quantization parameters of specific nodes. 

## Details

Each Node can be equalized by the ParameterEqualizer and the AbstractParamOptimizer. This equalization state updates the nodes scale and zero point. 

The Cost Function expects the parameters to be updated by reference. 
