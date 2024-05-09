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

Numeros=np.zeros((100))
Numeros2=np.zeros((100))
for i in range(0,100):

    _PARENT = Path(__file__).parent
    def main():
        # load mnist dataset:
        x_test=np.load('C:/Users/sergio.muniz/Desktop/QUBO/itcl-quantization-toolkit/quantizer/qubo_adaround/X_test.npy') 
        y_test=np.load('C:/Users/sergio.muniz/Desktop/QUBO/itcl-quantization-toolkit/quantizer/qubo_adaround/y_test.npy')
        # scale mnist
        #x_test = x_test.astype("float32") / 255
        print(x_test.shape)
        #x_test = x_test.reshape(x_test.shape[0], -1)
        data = x_test[:30000]
        #datax = x_test[:30000]  # first 1000 samples
        #datay = y_test[:30000]
        #print('data',data.shape)
        #print('ytest')
        #print(y_test)
        def loss(net: Network) -> float:
            inicio = time()
            #print('ytest')
            #print(y_test)
            predictions = net.infer(x_test)

            np.savetxt("output.csv", predictions, delimiter=",")
            #print('predic',predictions.shape)
            #print(predictions)
            #print('y',y_test.shape)
            mse = np.mean((predictions - y_test) ** 2)
            fin = time()
            print("Tiempo transcurrido:", "{:.10f}".format(fin-inicio))
            return mse
        
        model = load_model(f"{_PARENT}/models/model.h5")
        if isinstance(model,keras.Model):
            predictions = model.predict(x_test)
        mse = np.mean((predictions - y_test) ** 2)
        losse = model.evaluate(x_test, y_test, verbose=2)
        print('mse',mse)
        print('loss',losse)

        cfg = QuantizerCfg()
        cfg.dense.kernel_dtype = "int8"
        cfg.ada_round_net = RoundingQUBOCfg()  # set the rounding optimizer
        
        t = time()
        net = build(
            model_path=f"{_PARENT}/models/model.h5",
            output_path=f"{_PARENT}/models/quantized.json",
            representative_input=data,  # loss
            loss_fn=loss,
            cfg=cfg,
        )
        print(f"Quantization completed in {time() - t}s")

        seq_net = net.as_sequential_network()
        print("Network Loss:", loss(seq_net))
        inicio = time()
        loss(seq_net)
        fin = time()
        print('tiempo de inferencia cuantizado',fin-inicio)
        print("Network Loss:", loss(seq_net))
        Numeros[i]=loss(seq_net)
        Numeros2[i]=i
    if __name__ == "__main__":
        main()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(Numeros2, Numeros)
ax.scatter([101], [0.301613])
ax.scatter([102], [0.30212934])
#ax.scatter([103], [0.07270723])
ax.set_xlabel('Número de la prueba')
ax.set_ylabel('MSE')
ax.set_title('Boston Int-8')
plt.show()
