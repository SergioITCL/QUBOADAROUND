import tensorflow as tf
import numpy as np
from itcl_quantizer.tensor_extractor.abstract_layer import QuantizationResult
from itcl_quantizer.tensor_extractor.keras.layers import KerasLSTM, KerasInput
from itcl_quantization import Quantization


np.random.seed(0) 

class TestLSTM:

    def get_input_data(self, layer):
        arr = np.random.rand(10, 10, 2) # 10 samples of 10 time steps and 2 features per time step

        return QuantizationResult(arr, operators=[], out_node=None, layer=layer)


    def test_lstm_main(self):
        input_layer = tf.keras.layers.Input(shape=(10, 2))
        lstm_layer = tf.keras.layers.LSTM(2, )
        net = tf.keras.Sequential([

            input_layer,
            lstm_layer
        ])
        net.compile(loss='mse', optimizer='adam')

        data = self.get_input_data(input_layer)
        print(data)

        q = Quantization("int8")

        quant_input = KerasInput(q).quantize(data)
        
        KerasLSTM(lstm_layer).quantize(quant_input)



