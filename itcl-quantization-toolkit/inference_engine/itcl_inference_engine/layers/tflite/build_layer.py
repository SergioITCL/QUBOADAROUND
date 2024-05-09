from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.layer import ILayer
from itcl_inference_engine.layers.common.relu import RELU
from itcl_inference_engine.layers.common.sigmoid import SigmoidLUT
from itcl_inference_engine.layers.common.tanh import TanhLUT
from itcl_inference_engine.layers.itclq.lstm import LSTM
from itcl_inference_engine.layers.tflite.dequantize import Dequantize
from itcl_inference_engine.layers.tflite.fullyconnected import FullyConnected
from itcl_inference_engine.layers.tflite.logistic import Sigmoid
from itcl_inference_engine.layers.tflite.quantize import Quantize
from itcl_inference_engine.layers.tflite.softmax import SoftMax
from itcl_inference_engine.layers.tflite.tanh import Tanh
from itcl_inference_engine.util.check_layer_type import LayerType


def build_layer(layer: Operator) -> ILayer:
    layer_type = layer["op_type"]

    layer_builder = None
    if LayerType.is_tanh(layer_type):
        layer_builder = Tanh
    elif LayerType.is_tanhLUT(layer_type):

        layer_builder = TanhLUT
    elif LayerType.is_fully_connected(layer_type):
        layer_builder = FullyConnected
    elif LayerType.is_logistic(layer_type):
        layer_builder = Sigmoid
    elif LayerType.is_sigmoidLUT(layer_type):
        layer_builder = SigmoidLUT
    elif LayerType.is_softmax(layer_type):
        layer_builder = SoftMax
    elif LayerType.is_quantize(layer_type):
        layer_builder = Quantize
    elif LayerType.is_dequantize(layer_type):
        layer_builder = Dequantize
    elif LayerType.is_relu(layer_type):
        layer_builder = RELU
    elif LayerType.is_LSTM(layer_type):
        layer_builder = LSTM
    if layer_builder is None:
        raise ValueError("Invalid Layer " + layer_type)

    return layer_builder.from_model(layer)
