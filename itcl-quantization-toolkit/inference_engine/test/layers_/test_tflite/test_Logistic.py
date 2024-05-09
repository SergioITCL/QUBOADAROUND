import unittest

import numpy as np

from itcl_inference_engine.layers.tflite.logistic import Sigmoid


class TestSigmoid(unittest.TestCase):
    def test_op(self):

        x = np.array([42, 49, 28, 42, -2])
        y = np.array([65, 82, 19, 65, -80])

        sigmoid = Sigmoid(0.0588817298412323, 23)

        np.testing.assert_almost_equal(y, sigmoid.infer(x))
