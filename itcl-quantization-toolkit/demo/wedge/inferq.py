import json
from pathlib import Path

import numpy as np
from itcl_inference_engine.network.sequential import SequentialNetwork

PARENT = str(Path(__file__).parent)


def main():

    f = open(f"{PARENT}/tmp.json", encoding="utf-8")
    json_net = json.load(f)
    f.close()
    net = SequentialNetwork(json_net)

    X, y = np.load(f"{PARENT}/data/x_test.npy"), np.load(f"{PARENT}/data/y_test.npy")
    y_true = y

    y_pred = net.infer(X).squeeze()

    print(y_pred.shape, y_true.shape)
    # mse between y_pred and y
    print(np.mean((y_pred - y_true) ** 2))

    # plot y_pred and y in the same plot to compare
    import matplotlib.pyplot as plt

    plt.plot(y_pred, label="y_pred")
    plt.plot(y_true, label="y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
