import numpy as np

from itcl_inference_engine.layers.onnx.tanh import Tanh


class TestTanh:
    def test_op(self):
        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.tanh(x)  # expected output [-0.76159418, 0., 0.76159418]

        tanh = Tanh()

        np.testing.assert_almost_equal(y, tanh.infer(x))

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.tanh(x)

        np.testing.assert_almost_equal(y, tanh.infer(x))
