from pathlib import Path

import numpy as np
from itcl_inference_engine.network.sequential import SequentialNetwork

BASE = str(Path(__file__).parent)
DATA = f"{BASE}/data_1_step"
MODEL = f"{DATA}/model.json"


SLICE = None


def main():

    x_test, y_test = (
        np.load(f"{DATA}/x_test.npy")[:SLICE],
        np.load(f"{DATA}/pred_y_test.npy")[:SLICE],
    )

    net = SequentialNetwork.from_json_file(MODEL)

    y_pred = net.infer(x_test)

    assert y_pred.shape == y_test.shape, f"{y_pred.shape} != {y_test.shape}"
    print("====RES====")
    print("PRED:")
    print(y_pred)
    print("REAL:")
    print(y_test)
    # mse between y_pred and y
    print(np.mean((y_pred - y_test) ** 2))


if __name__ == "__main__":
    main()
