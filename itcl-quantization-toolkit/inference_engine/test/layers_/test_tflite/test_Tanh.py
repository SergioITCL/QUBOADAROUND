import numpy as np

from itcl_inference_engine.layers.tflite.tanh import Tanh


class TestTanh:
    def test_op(self):

        x = np.array([42, 49, 28, 42, -2])
        y = np.array([103, 117, 37, 103, -115])

        tanh = Tanh(0.0588817298412323, 23)

        np.testing.assert_almost_equal(y, tanh.infer(x))

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = (np.tanh(x) * 128).astype(np.int8)

        tanh = Tanh(1, 0)

        np.testing.assert_almost_equal(y, tanh.infer(x))
