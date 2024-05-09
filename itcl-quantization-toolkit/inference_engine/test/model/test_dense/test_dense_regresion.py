import os
import pathlib

import numpy as np

from itcl_inference_engine.network.sequential import SequentialNetwork


class TestFullWindModel:

    wd = pathlib.Path(__file__).parent.absolute()  # working dir

    def load_expected(self, path: pathlib.Path):
        inter_tensor_loadz = np.load(path)
        # Keys are a string, they net to be converted into an int to be sorted correctly
        keys = [int(k) for k in inter_tensor_loadz.keys()]
        return [inter_tensor_loadz[str(key)] for key in sorted(keys)]

    def get_net(self):
        return SequentialNetwork.from_json_file(str(self.wd / "wind_speed_int8.json"))

    def test_model(self):

        input_data = np.load(self.wd / "test_data_input.npy")

        net = self.get_net()
        net.infer(input_data)
        expected_intermediate_tensors = self.load_expected(
            self.wd / "intermediate_tensors.npy"
        )
        pred_intermediate_tensors = net._intermediate_results

        print("Expected:")
        print(expected_intermediate_tensors)
        print("=========")
        print("Predicted")
        print(pred_intermediate_tensors)

        for expected, pred in zip(
            expected_intermediate_tensors[:-1], pred_intermediate_tensors[:-1]
        ):
            np.testing.assert_equal(pred, expected)
        np.testing.assert_almost_equal(
            expected_intermediate_tensors[-1], pred_intermediate_tensors[-1]
        )
        print(expected_intermediate_tensors)
        print("=========")
        print(pred_intermediate_tensors)

    def test_model_500(self):
        input_data = np.load(self.wd / "test_data_input_500.npy")
        net = self.get_net()
        net.infer(input_data)

        expected_intermediate_tensors = self.load_expected(
            self.wd / "intermediate_tensors_500.npy"
        )
        pred_intermediate_tensors = net._intermediate_results

        for expected, pred in zip(
            expected_intermediate_tensors[:-1], pred_intermediate_tensors[:-1]
        ):
            np.testing.assert_equal(pred, expected)
        np.testing.assert_almost_equal(
            expected_intermediate_tensors[-1], pred_intermediate_tensors[-1]
        )

        print(
            "mse",
            np.square(
                expected_intermediate_tensors[-1] - pred_intermediate_tensors[-1]
            ).mean(),
        )
