from time import time
import numpy as np
from itcl_inference_engine.network.sequential import Network, SequentialNetwork
from typing import TypedDict

from itcl_quantizer.tensor_extractor.keras.keras_builder import build


class DataType(TypedDict):
    X_train: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray


data: DataType = np.load("data.npy")

if __name__ == "__main__":

    def loss(net: Network) -> float:
        res = net.infer(data["X_test"].astype(np.float32)).squeeze()
        expected = data["Y_test"]
        return np.square(expected - res).mean()

    t = time()

    model = "wind.json"
    model = "./wind_speed_int8.json"

    # network = SequentialNetwork.from_json_file(model)
    # print(f"loss of {model}: {loss(network)}")

    network = build("./model.h5", "wind.json", data["X_test"], loss)

    print(loss(network.as_sequential_network()))
    print(f"Model Loss: {loss(network.as_sequential_network())}")

    print(f"Model quantized in {time() - t} seconds")
