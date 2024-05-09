import numpy as np

from itcl_inference_engine.layers.tflite.fullyconnected import FullyConnected


class TestFullyConnected:
    def test_simple(self):
        input = np.array([103, 117, 37, 103, -115])

        input_scale = 0.0078125
        input_zerop = 0

        # random array of 5 x 5
        weights = np.array(
            [
                [100, 32, -12, 16, 4],
                [23, 12, 67, -24, -120],
                [1, -2, -3, -4, -5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
            ]
        )

        weights_scale = 0.005604765843600035
        weights_zerop = 0

        bias = [100, -100, 20, 40, -80]
        bias_scale = 0.0078125
        bias_zerop = 0

        output_scale = 0.017320988699793816
        output_zerop = -1

        fully_connected = FullyConnected.build(
            input_scale,
            input_zerop,
            weights_scale,
            weights_zerop,
            weights,
            bias_scale,
            bias_zerop,
            bias,
            output_scale,
            output_zerop,
        )

        np.testing.assert_almost_equal([37, 43, -1, 3, 6], fully_connected.infer(input))

    def test_batch(self):
        input = np.array([[103, 117, 37, 103, -115], [21, 36, 112, -84, -102]])
        input_scale = 0.0078125
        input_zerop = 0

        # random array of 5 x 5
        weights = np.array(
            [
                [100, 32, -12, 16, 4],
                [23, 12, 67, -24, -120],
                [1, -2, -3, -4, -5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
            ]
        )

        weights_scale = 0.005604765843600035
        weights_zerop = 0

        bias = [100, -100, 20, 40, -80]
        bias_scale = 0.0078125
        bias_zerop = 0

        output_scale = 0.017320988699793816
        output_zerop = -1

        fully_connected = FullyConnected.build(
            input_scale,
            input_zerop,
            weights_scale,
            weights_zerop,
            weights,
            bias_scale,
            bias_zerop,
            bias,
            output_scale,
            output_zerop,
        )

        np.testing.assert_almost_equal(
            np.array([[37, 43, -1, 3, 6], [0, 56, 0, -2, -3]]),
            fully_connected.infer(input),
        )

    def test_linear(self):

        input = np.array(
            [103, 117, 37, 103, -115],
        )
        input_scale = 0.0078125
        input_zerop = 0
        # load
        weights = np.load("test/test_data/layers/FullyConnected/weights.npy")

        weights_scale = 0.005604765843600035
        weights_zerop = 0

        bias = np.array(
            [
                -1339,
                2136,
                -1206,
                3468,
                -166,
                776,
                -3900,
                -1618,
                651,
                6173,
                -1230,
                -1160,
                1197,
                3551,
                360,
                -1451,
                1897,
                1595,
                -1574,
                -961,
                -1423,
                -652,
                -369,
                -1254,
                -939,
                -1414,
                -317,
                2894,
                -2679,
                2019,
                1780,
                2206,
            ]
        )
        bias_scale = 0.00004378723315312527
        bias_zerop = 0

        out = np.array(
            [
                37,
                52,
                5,
                56,
                -35,
                -5,
                48,
                -13,
                -24,
                -14,
                5,
                -28,
                -53,
                -11,
                19,
                -62,
                -36,
                -48,
                10,
                12,
                -2,
                26,
                -47,
                34,
                -23,
                41,
                4,
                -7,
                -1,
                26,
                67,
                91,
            ]
        )
        output_scale = 0.017320988699793816
        output_zerop = -1

        fully_connected = FullyConnected.build(
            input_scale,
            input_zerop,
            weights_scale,
            weights_zerop,
            weights,
            bias_scale,
            bias_zerop,
            bias,
            output_scale,
            output_zerop,
        )

        np.testing.assert_almost_equal(out, fully_connected.infer(input))
