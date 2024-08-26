from functools import partial
from pathlib import Path
from time import time
import numpy as np
import tensorflow as tf
from itcl_inference_engine.network.sequential import Network
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from time import time
#import time

from itcl_quantizer.config.models.keras import QuantizerCfg, RoundingQUBOCfg
from itcl_quantizer.config.models.keras import RoundingAnnealerCfg
from itcl_quantizer.tensor_extractor.keras.keras_builder import build

Numeros=np.zeros((100))
Numeros2=np.zeros((100))
for i in range(0,1):

    _PARENT = Path(__file__).parent
    def main():
        # load mnist dataset:
        x_test=np.load('X_test.npy') 
        y_test=np.load('y_test.npy')    


        data = x_test[:1000]  # first 1000 samples
        datax = x_test[:1000]  # first 1000 samples
        datay = y_test[:1000]  # first 1000 samples
        
        
        model = load_model(f"{_PARENT}/models/model.h5")
        inicio = time()
        los,accuracy = model.evaluate(x_test, y_test, verbose=2) # type: ignore
        fin = time()
        print('tiempo de inferencia sin cuantizar',fin-inicio)
        print('accuracy',accuracy)
        

        def accuracy(net: Network) -> float:
            inicio = time()
            res = net.infer(x_test)
            res = res.T.argmax(axis=0)
            hits = 0
            total = len(y_test)
            for pred, exp in zip(res, y_test):
                if pred == exp:
                    hits += 1
            fin = time()
            print("Tiempo transcurrido:", "{:.10f}".format(fin-inicio))
            return hits / total

        cfg = QuantizerCfg()
        cfg.dense.kernel_dtype = "int32"

        cfg.ada_round_net = RoundingQUBOCfg()  # set the rounding optimizer
        
        t = time()
        net = build(
            model_path=f"{_PARENT}/models/model.h5",
            output_path=f"{_PARENT}/models/quantized.json",
            representative_input=data,  # loss
            loss_fn=accuracy,
            cfg=cfg,
        )
        print(f"Quantization completed in {time() - t}s")

        seq_net = net.as_sequential_network()
        inicio = time()
        accuracy(seq_net)
        fin = time()
        print('tiempo de inferencia cuantizado',fin-inicio)
        print("Network Loss:", accuracy(seq_net))
        Numeros[i]=accuracy(seq_net)
        Numeros2[i]=i
    if __name__ == "__main__":
        main()
#print(Numeros)

import matplotlib.pyplot as plt


# Datos de ejemplo

# Crear el gráfico
fig, ax = plt.subplots()

# Puntos azules para Numeros y Numeros2
scatter1 = ax.scatter(Numeros2, Numeros, color='blue', label='Coeficientes aleatorios')

# Otros puntos con colores diferentes
scatter2 = ax.scatter([101], [0.1525], color='orange', label='Round-to-nearest')
scatter3 = ax.scatter([102], [0.1864], color='green', label='ADAROUND - Neal')




# Configuración del gráfico
ax.set_xlabel('Número de la prueba', fontsize=16)
ax.set_ylabel('Precisión',fontsize=16)
ax.set_title('Modelo MNIST 10x10 cuantizado a Int-4',fontsize=16)

# Añadir la leyenda
ax.legend(loc='lower left')

plt.show()