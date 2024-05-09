from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase
from itcl_quantization import Quantization
import numpy as np


class TestNodeTensor:
    Q = Quantization("int8")

    def test_correct_rounding(self):
        SHAPE = (4, 4)

        for initial, expected in [(3.2, 6), (1.8, 4), (-0.2, 0), (-0.4, -1)]:
            t = np.full(SHAPE, initial)

            node = NodeTensorBase(0.5, 0, "", self.Q).with_tensor(t)
            np.testing.assert_almost_equal(node.quantized, np.full((SHAPE), expected))

    def test_initial_rounding_policy(self):
        SHAPE = (4, 4)

        for initial, expected in [(1.2, 0), (1.4, 1), (-0.1, 1), (-0.4, 0)]:
            print(initial)

            t = np.full(SHAPE, initial)
            expected_rounding_policy = np.full(SHAPE, expected)
            node = NodeTensorBase(0.5, 0, "", self.Q).with_tensor(t)
            np.testing.assert_almost_equal(
                node.rounding_policy, expected_rounding_policy
            )

    def test_wrong_update_rounding_policy(self):
        SHAPE = (4, 4)
        try:
            t = np.full(SHAPE, 2.2)
            new_rp = np.full((16,), 0)

            node = NodeTensorBase(0.5, 0, "", self.Q).with_tensor(t)
            node.rounding_policy = new_rp
            raise ValueError("Test Failed, invalid rounding policy update")
        except ValueError:
            pass

    def test_update_rounding_policy(self):
        SHAPE = (4, 4)

        for original, policy, expected in [
            (1.2, 1, 3),
            (1.2, 0, 2),
            (1.4, 1, 3),
            (1.4, 0, 2),
            (-0.2, 0, -1),
            (-0.6, 0, -2),
        ]:

            t = np.full(SHAPE, original)
            new_policy = np.full(SHAPE, policy)
            node = NodeTensorBase(0.5, 0, "", self.Q).with_tensor(t)

            node.rounding_policy = new_policy
            expected = np.full(SHAPE, expected)
            np.testing.assert_almost_equal(node.quantized, expected)

    def test_dequantization(self):
        SHAPE = (4, 4)

        for (original, dequantized) in [(0.4, 0.5), (0.8, 1), (-1.2, -1)]:
            t = np.full(SHAPE, original)
            node = NodeTensorBase(0.5, 0, "", self.Q).with_tensor(t)
            dq = node.dequantized
            expected_dq = np.full(SHAPE, dequantized)
            np.testing.assert_almost_equal(dq, expected_dq)

    def test_LUT(self):
        node = NodeTensorBase(1, 0, "", self.Q)

        lut = node.with_lut(lambda x: np.abs(x), out_scale=1, out_zp=0).LUT

        if not lut:
            raise ValueError("LUT is NONE")
        print(lut.as_json())
        assert lut.offset == 128
        assert len(lut) == 256
        assert lut[0] == 127
        assert lut[0 + lut.offset] == 0
        assert lut[255] == 127
