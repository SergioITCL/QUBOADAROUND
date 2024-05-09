import os
import random

import numpy as np
from network.Network import SequentialNetwork

onnx_models = [
    f"models/onnx/{model}"
    for model in filter(lambda x: ".json" in x, list(os.listdir("models/onnx")))
]
tflite_models = [f"models/tflite/{model}" for model in os.listdir("models/tflite")]

# choose two random models from the array
onnx_test_models = random.sample(onnx_models, 2)
tflite_test_models = random.sample(tflite_models, 2)

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test_dense = (
    x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2])).astype(
        "float32"
    )
    / 255
)

for model in onnx_test_models + tflite_test_models:
    print(model)
    net = SequentialNetwork.from_json_file(model)

    hits = 0
    for x, y in zip(x_test_dense, y_test):
        res = net.infer(x)

        lbl = np.argmax(res)
        if y == lbl:
            hits += 1

    print(f"ACC: {hits / len(x_test_dense)}")
    print(f"Hits: {hits}")
    print("\n\n")
    assert hits / len(x_test_dense) > 0.8
