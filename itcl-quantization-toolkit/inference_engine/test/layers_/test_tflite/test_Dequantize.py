import numpy as np

from itcl_inference_engine.layers.tflite.dequantize import Dequantize


class TestDequantize:
    def test_op(self):
        x = np.array([0, 3, 128, -127]).astype(np.int8)
        x_scale = float(2)
        x_zero_point = 128

        y = np.array([-256, -250, -512, -510], dtype=np.float32)

        res = Dequantize(x_scale, x_zero_point, "int8").infer(x)
        np.testing.assert_array_almost_equal(res, y)
