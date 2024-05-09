import numpy as np

from itcl_inference_engine.layers.onnx.q_linear_mmul import QLinearMatMul


class TestQLinearMatMul:
    def test_op(self):
        a = np.array(
            [
                [208, 236, 0, 238],
                [3, 214, 255, 29],
            ],
            dtype=np.uint8,
        )

        a_scale = np.array([0.0066], dtype=np.float32)
        a_zero_point = np.array([113], dtype=np.uint8)

        b = np.transpose(
            np.array(
                [[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]],
                dtype=np.uint8,
            )
        )

        b_scale = np.array([0.00705], dtype=np.float32)
        b_zero_point = np.array([114], dtype=np.uint8)

        y_scale = np.array([0.0107], dtype=np.float32)
        y_zero_point = np.array([118], dtype=np.uint8)

        output = np.array(
            [
                [168, 115, 255],
                [1, 66, 151],
            ],
            dtype=np.uint8,
        )

        matmulop = QLinearMatMul(
            a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
        )

        np.testing.assert_almost_equal(output, matmulop.infer(a))

    def test_overflow(self):
        a = np.array(
            [[255, 255, 255], [255, 255, 255], [255, 255, 255]], dtype=np.uint8
        )

        a_scale = np.array([0.0066], dtype=np.float32)
        a_zero_point = np.array([113], dtype=np.uint8)

        b = np.transpose(
            np.array([[255, 255, 255], [255, 255, 255], [0, 255, 255]], dtype=np.uint8)
        )

        b_scale = np.array([0.00705], dtype=np.float32)
        b_zero_point = np.array([114], dtype=np.uint8)

        y_scale = np.array([0.0107], dtype=np.float32)
        y_zero_point = np.array([118], dtype=np.uint8)

        output = np.array(
            [[222, 255, 255], [222, 255, 255], [222, 255, 255]], dtype=np.uint8
        )

        matmulop = QLinearMatMul(
            a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
        )
        print(matmulop.infer(a))
        np.testing.assert_almost_equal(output, matmulop.infer(a))
