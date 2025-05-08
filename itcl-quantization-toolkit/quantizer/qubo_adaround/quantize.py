
from functools import partial
from pathlib import Path
from time import time
import numpy as np
import tensorflow as tf
from itcl_inference_engine.network.sequential import Network
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import load_model


from itcl_quantizer.config.models.keras import QuantizerCfg, RoundingQUBOCfg
from itcl_quantizer.config.models.keras import RoundingAnnealerCfg
from itcl_quantizer.tensor_extractor.keras.keras_builder import build

_PARENT = Path(__file__).parent
def main():
    # load mnist dataset:
    
    _, (x_test, y_test) = keras.datasets.mnist.load_data()

    # scale mnist
    x_test = x_test.astype("float32") / 255
    x_test = np.array([tf.image.resize(np.expand_dims(image, axis=-1), (7, 7)).numpy() for image in x_test]) # type: ignore
    # flatten mnist
    x_test = x_test.reshape(x_test.shape[0], -1)
    print(x_test.shape)
    data = x_test[:1000]  # first 1000 samples
    datax = x_test[:1000]  # first 1000 samples
    datay = y_test[:1000]

    def loss(net: Network) -> float:
        res = net.infer(x_test)
        res = res.T.argmax(axis=0)
        #print(res.shape)
        hits = 0
        for pred, exp in zip(res, y_test):
            # print(pred, exp)
            if pred == exp:
                hits += 1
        return -hits

    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = "int4"
    cfg.ada_round_net = RoundingQUBOCfg()  # set the rounding optimizer
    
    t = time()
    net = build(
        model_path=f"{_PARENT}/models/mnist.h5",
        output_path=f"{_PARENT}/models/quantized.json",
        representative_input=data,  # loss
        loss_fn=loss,
        cfg=cfg,
    )
    print(f"Quantization completed in {time() - t}s")

    seq_net = net.as_sequential_network()
    print("Network Loss:", loss(seq_net))

if __name__ == "__main__":
    main()
