import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from itcl_quantizer import keras_build

BASE_DIR = str(Path(__file__).resolve().parent)
DATA = f"{BASE_DIR}/data_1_step"


print("pwd", BASE_DIR)
x_train, y_train = np.load(f"{DATA}/x_train.npy"), np.load(f"{DATA}/y_train.npy")


rep_dataset_idx = int(len(x_train) * 0.5)
rep_dataset_idx = 2
print(x_train.shape)
print(x_train[:rep_dataset_idx].shape)
quant_model = keras_build(
    f"{DATA}/model.h5", f"{DATA}/model.json", x_train[:rep_dataset_idx]
)

net = quant_model.as_sequential_network()

pred = net.infer(x_train[:rep_dataset_idx])

real = y_train[:rep_dataset_idx]

assert pred.shape == real.shape

mse = np.mean((pred - real) ** 2)
print("MSE", mse)

with open(f"{DATA}/model.json", "w") as f:
    f.write(json.dumps(quant_model.as_json()))
