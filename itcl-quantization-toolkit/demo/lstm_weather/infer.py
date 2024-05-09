from pathlib import Path

import numpy as np
from itcl_inference_engine.network.sequential import SequentialNetwork
import matplotlib.pyplot as plt
BASE = str(Path(__file__).parent)
DATA = f"{BASE}/data_2steps"
MODEL = f"{BASE}/qmodels/2steps.json"
def main():

    x_test, y_test = (
        np.load(f"{DATA}/x_test.npy"),
        np.load(f"{DATA}/y_test.npy"),
    )
    y_test_pred = np.load(f"{DATA}/pred_y_test.npy")
    
    print(x_test.shape, y_test.shape)

    net = SequentialNetwork.from_json_file(MODEL)

    y_pred = net.infer(x_test)

    assert y_pred.shape == y_test_pred.shape, f"{y_pred.shape} != {y_test.shape}"
    # mse between y_pred and y
    print(np.mean((y_pred - y_test_pred ) ** 2))

    plt.plot(y_pred, label="y_pred")
    plt.plot(y_test_pred, label="y_test_pred")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
