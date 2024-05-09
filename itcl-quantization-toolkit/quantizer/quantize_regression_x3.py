from time import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itcl_quantizer.tensor_extractor.keras.keras_builder import build
from itcl_inference_engine.network.sequential import Network


fn = lambda x: x**3
if __name__ == "__main__":
    train_x = linespace = np.expand_dims(np.arange(-10, 10, 0.005), axis=1)

    scaler = MinMaxScaler()
    scaler.fit(train_x)

    train_y = fn(train_x)
    train_x = scaler.transform(train_x)
    train_y = scaler.transform(train_y)

    def loss(net: Network) -> float:

        res = net.infer(train_x)

        return np.square(train_y - res).mean()

    t = time()

    net = build(
        "./models/keras/regression/x3/model.h5", "x3_symmetric.json", train_x, loss
    )

    print(f"Model Loss: {loss(net.as_sequential_network())}")

    print(f"Model quantized in {time() - t} seconds")
