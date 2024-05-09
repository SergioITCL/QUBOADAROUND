import numpy as np

from itcl_inference_engine.layers.tflite.quantize import Quantize


class TestQuantize:
    def test_op(self):
        x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
        y_scale = 0.8
        y_zero_point = 64
        y = np.array([64, 67, 68, 127, -128, -128]).astype(np.int8)

        res = Quantize(y_scale, y_zero_point, "int8").infer(x)
        np.testing.assert_array_almost_equal(res, y)
