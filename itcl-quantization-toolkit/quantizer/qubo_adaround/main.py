from functools import partial
from pathlib import Path
from time import time
import numpy as np
import tensorflow as tf
from itcl_inference_engine.network.sequential import Network
import matplotlib.pyplot as plt
from keras.models import load_model
from time import time


from itcl_quantizer.config.models.keras import QuantizerCfg, RoundingQUBOCfg
from itcl_quantizer.config.models.keras import RoundingAnnealerCfg
from itcl_quantizer.tensor_extractor.keras.keras_builder import build

_PARENT = Path(__file__).parent
def main():

    # parameters of the quantization
    int_to_quantize = "int2"
    n_data_for_quantization = 10000

    # parameters for solving the associated qubo problems
    random_adaround_coefficients = False
    quantization_to_round_nearest  = False
    random_qubo_num_variables = False
    qubo_sampler  ="neal"

    qaoa_num_reps = 1
    dwave_num_reads  = 30
    dwave_annealing_time  = 20
    dwave_chain_strength  = 0

    dictionary_subspace = False
    
    # load mnist dataset and its evaluation:
    x_test=np.load('models_and_data/X_test.npy') 
    y_test=np.load('models_and_data/y_test.npy')    
    data = x_test[:n_data_for_quantization]  # first 1000 samples
    print(x_test.shape)

    model = load_model('models_and_data/model.h5')
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
        return hits/total


    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = int_to_quantize
    cfg.ada_round_net = RoundingQUBOCfg(qubo_sampler = qubo_sampler, random_adaround_coefficients = random_adaround_coefficients, cuantization_to_round_nearest = quantization_to_round_nearest, qaoa_num_reps = qaoa_num_reps)  # set the rounding optimizer
    
    t = time()
    net = build(
        model_path="models_and_data/model.h5",
        output_path="models_and_data/quantized.json",
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
