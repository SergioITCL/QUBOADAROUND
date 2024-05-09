import numpy as np

from itcl_inference_engine.layers.onnx.dequantize_linear import \
    DequantizeLinear


class TestDequantizeLinear:
    def test_op(self):
        x = np.array([0, 3, 128, 255]).astype(np.uint8)
        x_scale = np.array([2], dtype=np.float32)
        x_zero_point = np.array([128], dtype=np.float32)

        y = np.array([-256, -250, 0, 254], dtype=np.float32)

        res = DequantizeLinear(x_scale, x_zero_point).infer(x)
        np.testing.assert_array_almost_equal(res, y)
