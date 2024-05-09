import numpy as np

from itcl_inference_engine.layers.onnx.quantize_linear import QuantizeLinear


class TestQuantizeLinear:
    def test_op(self):
        x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = np.uint8(128)
        y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)

        res = QuantizeLinear(y_scale, y_zero_point).infer(x)
        np.testing.assert_array_almost_equal(res, y)
