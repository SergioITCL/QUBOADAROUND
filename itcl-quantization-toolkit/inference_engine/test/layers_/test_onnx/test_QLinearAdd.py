import numpy as np

from itcl_inference_engine.layers.onnx.q_linear_add import QLinearAdd


class TestQLinearAdd:
    def test_op(self):

        a = np.array([[150, 150, 150, 150, 150]], dtype=np.uint8)

        a_scale = np.array([0.05834893882274628], dtype=np.float32)
        a_zero_point = np.array([150], dtype=np.uint8)

        b = np.array([89, 255, 0, 21, 187], dtype=np.uint8)

        b_scale = np.array([0.0015328848967328668], dtype=np.float32)
        b_zero_point = np.array([83], dtype=np.uint8)

        y_scale = np.array([0.058881718665361404], dtype=np.float32)
        y_zero_point = np.array([151], dtype=np.uint8)

        output = np.array([[151, 155, 149, 149, 154]], dtype=np.uint8)

        add_op = QLinearAdd(
            a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
        )

        np.testing.assert_almost_equal(output, add_op.infer(a))

    def test_op_2(self):

        a = np.array(
            [
                [
                    119,
                    130,
                    131,
                    132,
                    137,
                    139,
                    129,
                    116,
                    131,
                    125,
                    133,
                    139,
                    117,
                    138,
                    134,
                    137,
                    130,
                    133,
                    118,
                    120,
                    125,
                    131,
                    127,
                    133,
                    142,
                    127,
                    120,
                    131,
                    138,
                    127,
                    119,
                    126,
                ]
            ]
        )

        a_scale = np.array([0.01712208241224289], dtype=np.float32)
        a_zero_point = np.array([129], dtype=np.uint8)

        b = np.array(
            [
                65,
                153,
                68,
                187,
                95,
                119,
                0,
                58,
                115,
                255,
                68,
                70,
                129,
                189,
                108,
                62,
                147,
                139,
                59,
                75,
                63,
                82,
                90,
                67,
                75,
                63,
                91,
                172,
                31,
                150,
                144,
                155,
            ]
        )

        b_scale = np.array([0.0017296670703217387], dtype=np.float32)
        b_zero_point = np.array([99], dtype=np.uint8)

        y_scale = np.array([0.017320986837148666], dtype=np.float32)
        y_zero_point = np.array([127], dtype=np.uint8)

        output = np.array(
            [
                [
                    114,
                    133,
                    126,
                    139,
                    135,
                    139,
                    117,
                    110,
                    131,
                    139,
                    128,
                    134,
                    118,
                    145,
                    133,
                    131,
                    133,
                    135,
                    112,
                    116,
                    119,
                    127,
                    124,
                    128,
                    137,
                    121,
                    117,
                    136,
                    129,
                    130,
                    122,
                    130,
                ]
            ],
            dtype=np.uint8,
        )

        add_op = QLinearAdd(
            a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
        )

        np.testing.assert_almost_equal(output, add_op.infer(a))
