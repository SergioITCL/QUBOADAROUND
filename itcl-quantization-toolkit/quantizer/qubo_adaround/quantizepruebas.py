from functools import partial
from pathlib import Path
from time import time
import numpy as np
import tensorflow as tf
from itcl_inference_engine.network.sequential import Network
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt



from itcl_quantizer.config.models.keras import QuantizerCfg, RoundingQUBOCfg
from itcl_quantizer.config.models.keras import RoundingAnnealerCfg
from itcl_quantizer.tensor_extractor.keras.keras_builder import build

_PARENT = Path(__file__).parent
def Uber():
    # load mnist dataset:
    x_test=np.load('C:/Users/sergio.muniz/Desktop/QUBO/itcl-quantization-toolkit/quantizer/qubo_adaround/X_testUb.npy') 
    y_test=np.load('C:/Users/sergio.muniz/Desktop/QUBO/itcl-quantization-toolkit/quantizer/qubo_adaround/y_testUb.npy')

    # scale mnist
    #x_test = x_test.astype("float32") / 255
    print(x_test.shape)
    #x_test = x_test.reshape(x_test.shape[0], -1)

    data = x_test[:30000]  # first 1000 samples

    def loss(net: Network) -> float:
        predictions = net.infer(x_test)
        mse = np.mean((predictions - y_test) ** 2)
        return mse
    
    model = load_model(f"{_PARENT}/models/Uber.h5")
    if isinstance(model,keras.Model):
        predictions = model.predict(x_test)
    mse = np.mean((predictions - y_test) ** 2)
    losse = model.evaluate(x_test, y_test, verbose=2)
    print('mse',mse)
    print('loss',losse)

    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = "int16"
    cfg.ada_round_net = RoundingQUBOCfg()  # set the rounding optimizer
    
    t = time()
    net = build(
        model_path=f"{_PARENT}/models/Uber.h5",
        output_path=f"{_PARENT}/models/quantized.json",
        representative_input=data,  # loss
        loss_fn=loss,
        cfg=cfg,
    )
    print(f"Quantization completed in {time() - t}s")
    t=time() - t
    seq_net = net.as_sequential_network()
    print("Network Loss:", loss(seq_net)/30000)
    loss =loss(seq_net)/30000
    return loss, t


def Aba():
    # load mnist dataset:
    x_test=np.load('C:/Users/sergio.muniz/Desktop/QUBO/itcl-quantization-toolkit/quantizer/qubo_adaround/X_testaba.npy') 
    y_test=np.load('C:/Users/sergio.muniz/Desktop/QUBO/itcl-quantization-toolkit/quantizer/qubo_adaround/y_testaba.npy')
    data = x_test[:627]  # first 1000 samples
    model = load_model(f"{_PARENT}/models/TIC.h5")
    result=model.evaluate(x_test, y_test, verbose=2)
    predictions = model.predict(x_test)
    # Convertir las predicciones a etiquetas de clases (si es necesario)
    predicted_labels = np.argmax(predictions, axis=1)
    # Calcular la precisión comparando las etiquetas predichas con las verdaderas
    accuracy = np.mean(predicted_labels == y_test)
    print("Accuracy:", accuracy)
    def accuracy(net: Network) -> float:
        res = net.infer(x_test)
        res = res.T.argmax(axis=0)

        hits = 0
        total = len(y_test)
        for pred, exp in zip(res, y_test):
            if pred == exp:
                hits += 1
        return hits / total

    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = "int16"
    cfg.ada_round_net = RoundingQUBOCfg()  # set the rounding optimizer
    
    t = time()
    net = build(
        model_path=f"{_PARENT}/models/TIC.h5",
        output_path=f"{_PARENT}/models/quantized.json",
        representative_input=data,  # loss
        loss_fn=accuracy,
        cfg=cfg,
    )
    print(f"Quantization completed in {time() - t}s")
    t=time() - t
    seq_net = net.as_sequential_network()
    print("Network Loss:", accuracy(seq_net))
    loss =accuracy(seq_net)
    return loss, t
if __name__ == "__main__":
    t1, N1= Uber()
    with open('resultado.txt', 'w') as f:
        print('Uber Adaround', file=f)
        print('Time', t1,'s', file=f)
        print('Network Loss', N1, file=f)
    print('2')
    t2, N2=Aba()
    with open('resultado.txt', 'a') as f:
        print('Aba Adaround', file=f)
        print('Time', t2,'s', file=f)
        print('Network Loss', N2, file=f)
    print('3')
