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


from itcl_quantizer.config.models.keras import QuantizerCfg, RoundingQUBOCfg
from itcl_quantizer.config.models.keras import RoundingAnnealerCfg
from itcl_quantizer.tensor_extractor.keras.keras_builder import build

_PARENT = Path(__file__).parent
def main():
    # load mnist dataset:
    x_test=np.load('/home/sergio/ADAROUND/QUBOADAROUND/itcl-quantization-toolkit/quantizer/qubo_adaround/X_test.npy') 
    y_test=np.load('/home/sergio/ADAROUND/QUBOADAROUND/itcl-quantization-toolkit/quantizer/qubo_adaround/y_test.npy')    


    data = x_test[:1000]  # first 1000 samples
  
    model = load_model(f"{_PARENT}/models/model.h5")
    inicio = time()
    los,accuracy = model.evaluate(x_test[1000:], y_test[1000:], verbose=2) # type: ignore
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
        return hits/total

    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = "int2"

    cfg.ada_round_net = RoundingQUBOCfg(qubo_sampler="neal", random_adaround_coefficients=False)  # set the rounding optimizer
    
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
if __name__ == "__main__":
    main()
