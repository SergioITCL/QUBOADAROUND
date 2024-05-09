import numpy as np
from tensorflow import keras
from itcl_quantizer.tensor_extractor.abstract_layer import QuantizationResult
from itcl_quantizer.tensor_extractor.keras.layers.keras_dense import KerasDense
from itcl_quantizer.tensor_extractor.keras.layers.keras_input import KerasInput
from itcl_quantization import Quantization


class TestDense:
    def get_input_data(self, layer):
        float_data = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5])
        float_data = np.expand_dims(float_data, axis=1)
        return QuantizationResult(float_data, operators=[], out_node=None, layer=layer)

    def test_dense_main(self):
        input_layer = keras.layers.Input(shape=(1,))
        dense_layer = keras.layers.Dense(
            10, activation="tanh", kernel_initializer="zeros"
        )
        keras.Sequential([input_layer, dense_layer])  # Initialize the weights

        data = self.get_input_data(input_layer)

        q = Quantization("int8")
        quant_input = KerasInput(q).quantize(data)
        data = q.quantize(data.input_data, 6, 0.002)
        res = KerasDense(dense_layer).quantize(quant_input)
        op = res.operators or []
        q_dense = op[0]
        q_activation = op[1]

        assert q_activation.inputs[0].LUT is not None
        assert len(q_activation["inputs"][0]["LUT"]["LUT"]) == 256
        assert q_activation["inputs"][0]["LUT"]["offset"] == 128

        # Input
        np.testing.assert_almost_equal(
            q_dense["inputs"][0]["scale"], 0.00980392156862745
        )
        assert q_dense["inputs"][0]["zero_point"] == -26

        # Kernel
        kernel = q_dense["inputs"][1]
        np.testing.assert_almost_equal(kernel["scale"], 0.003937007874015748)
        assert kernel["zero_point"] == 0
        assert (kernel["tensor"]["dtype"]).upper() == "INT8"

        # Bias
        bias = q_dense["inputs"][2]
        assert len(bias["tensor"]["tensor"]) == 10
        np.testing.assert_almost_equal(bias["scale"], 3.85981164119191e-05)
        assert bias["zero_point"] == 0
        assert (bias["tensor"]["dtype"]).upper() == "INT32"

        # Bias Add
        bias_add = q_dense["outputs"][0]
        np.testing.assert_almost_equal(bias_add["scale"], 0.00392156862745098)
        assert bias_add["zero_point"] == 0

    def test_relu_is_skipped(self):

        input_layer = keras.layers.Input(shape=(1,))
        dense_layer = keras.layers.Dense(
            10, activation="relu", kernel_initializer="ones"
        )
        keras.Sequential([input_layer, dense_layer])  # Initialize the weights

        data = self.get_input_data(input_layer)
        q = Quantization("int8")
        quant_input = KerasInput(q).quantize(data)
        res = KerasDense(dense_layer).quantize(quant_input)

        assert len(res.operators or []) == 1
