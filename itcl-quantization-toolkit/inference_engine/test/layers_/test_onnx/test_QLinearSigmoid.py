import numpy as np

from itcl_inference_engine.layers.onnx.q_linear_sigmoid import QLinearSigmoid


class TestSigmoid:
    def test_op(self):

        a = np.array([[133, 141, 126, 12, 231]], dtype=np.uint8)

        a_scale = np.array([0.05834893882274628], dtype=np.float32)
        a_zero_point = np.array([150], dtype=np.uint8)

        b_scale = np.array([0.0015328848967328668], dtype=np.float32)
        b_zero_point = np.array([83], dtype=np.uint8)

        output = np.array([[255, 255, 212, 83, 255]], dtype=np.uint8)

        add_op = QLinearSigmoid(a_scale, a_zero_point, b_scale, b_zero_point)

        np.testing.assert_almost_equal(output, add_op.infer(a))
