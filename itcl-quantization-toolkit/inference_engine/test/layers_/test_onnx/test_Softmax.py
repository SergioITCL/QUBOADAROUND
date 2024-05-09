import numpy as np

from itcl_inference_engine.layers.onnx.softmax import Softmax


class TestSoftmax:
    def test_op(self):

        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output [[0.09003058, 0.24472848, 0.66524094]]
        y = Softmax().infer(x)

        res = Softmax().infer(x)
        np.testing.assert_array_almost_equal(res, y)
