from typing import TYPE_CHECKING, List, Tuple
from itcl_quantization.quantization.operators import Quantization
from itcl_quantizer.equalizers.adaround.Calculationsint import ADAInt
from itcl_quantizer.quantizer.distributions.distribution import Distribution
import tensorflow as tf
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import dimod
from typing import Union
import greedy
import time
from dwave.system import DWaveSampler
from dwave.system import EmbeddingComposite
from itcl_quantizer.equalizers.adaround.iround_optimizer import IRoundOptimizer
from itcl_quantizer.tensor_extractor.abstract_layer import (
    AbstractLayer,
    QuantizationResult,
)
from itcl_quantizer.tensor_extractor.keras.utils import CheckType
from itcl_quantizer.equalizers.adaround.qubocalculation import *
from typing import cast
from itcl_quantizer.tensor_extractor.tensor import NodeTensorTensor
if TYPE_CHECKING:
    from itcl_quantizer.tensor_extractor.keras.layers.keras_dense import KerasDense

class QUBOAnnealer(IRoundOptimizer):
    """Rounding Annealer"""

    _layer: AbstractLayer
    _round_policy: List[np.ndarray]
    _results: QuantizationResult
    _input_data: np.ndarray

    def __init__(
        self,
        random_adaround_coefficients : bool = False, 
        qubo_sampler : str ="neal",
        qaoa_num_reps: int = 1,
        dwave_num_reads : int = 30,
        dwave_annealing_time : int = 20,
        dwave_chain_strength : int = 0,
        dictionary_subspace: bool = False,
        random_qubo_num_variables : Union[bool, int] = False


    ) -> None:
        super().__init__()
        self.random_adaround_coefficients = random_adaround_coefficients 
        self.qubo_sampler = qubo_sampler
        self.qaoa_num_reps = qaoa_num_reps
        self.dwave_num_reads = dwave_num_reads
        self.dwave_annealing_time = dwave_annealing_time
        self.dwave_chain_strenght = dwave_chain_strength 
        self.dictionary_subspace = dictionary_subspace
        self.random_qubo_num_variables = random_qubo_num_variables

    def set_input_data(self, data: np.ndarray) -> IRoundOptimizer:
        self._input_data = data
        return self

    def set_quant_results(self, results: QuantizationResult) -> IRoundOptimizer:
        self._results = results
        return super().set_quant_results(results)

    def set_initial_neigh(self, neigh: List[np.ndarray]) -> "IRoundOptimizer":
        self._round_policy = neigh
        return self

    def set_layer(self, layer: AbstractLayer) -> IRoundOptimizer:
        self._layer = layer
        return super().set_layer(layer)

    def _qubo_dense(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        """QUBO for dense layers

        Args:
            tensors (List[np.ndarray]): The rounding policy

        Returns:
            float: The cost of the rounding policy
        """

        if self.random_qubo_num_variables == False:
            dense_layer: KerasDense = self._layer  # type: ignore
            kernel, bias = dense_layer.get_weights()
            input_f = self._input_data
            input_f = np.array(input_f)
            output_f = self._results.input_data
            output_f = np.array(output_f)
            # output_f = TODO
                    
            assert self._results.operators

            q_dense_op = self._results.operators[0]
        
            q_kernel = q_dense_op.inputs[1]
            kernel_s, kernel_zp = q_kernel.scale, q_kernel.zero_point
            q_bias = q_dense_op.inputs[2]
            bias_s, bias_zp = q_bias.scale, q_bias.zero_point
            q_input = q_dense_op.inputs[0]
            input_s, input_zp = q_input.scale, q_input.zero_point
        
        else:
            kernel,bias,input_f,output_f, kernel_s,input_s,bias_s= random_qubo_problem(self.random_qubo_num_variables)


        InputDimension=input_f[0].shape[0]
        OutputDimension=bias.shape[0]
        Numero_datasets=input_f[:,0].shape[0]
        Bterm1=np.zeros((Numero_datasets,InputDimension))
        Bterm2=np.zeros((Numero_datasets,InputDimension*InputDimension))
        Bterm3=np.zeros((Numero_datasets,1))
        Bterm4=np.zeros((Numero_datasets,InputDimension))
        Redondeo_Kernel=np.zeros((OutputDimension,InputDimension))
        Redondeo_Bias=np.zeros((OutputDimension))
        M = np.zeros((OutputDimension*InputDimension+OutputDimension, OutputDimension*InputDimension+OutputDimension))
        DiccionarioT={}

    

        if self.random_adaround_coefficients == True:
            Redondeo_Kernel = np.random.choice([0, 1], size=(OutputDimension, InputDimension))
            Redondeo_Bias = np.random.choice([0, 1], size=(OutputDimension))     

            return [Redondeo_Kernel,Redondeo_Bias]
        
        else:
                    
            #inicio = time.time()
            Input_Round2=Input_Round_Calculation(input_f,input_s)
            dTermino2=dterm_Calculation(bias,kernel,Input_Round2,output_f,kernel_s,input_s,bias_s)
            
            for l in range(0,OutputDimension):
         
                diccionario1=Bterm1_Calculation_Subespacio(Bterm1,Input_Round2,dTermino2,l,InputDimension,Numero_datasets,kernel_s,input_s)
                #print(diccionario11)
                
                if l==0:
                    diccionario2,Bt2=Bterm2_Calculation_Subespacio(Bterm2,Input_Round2,l,InputDimension,Numero_datasets,kernel_s,input_s)
                else:
                    diccionario2=Bterm_Calculation_Vuelta(Bt2,InputDimension,l)
                
                diccionario3=Bterm3_Calculation_Subespacio(Bterm3,dTermino2,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
                #print(diccionario3)
                if l==0:
                    diccionario4,Bt4=Bterm4_Calculation_Subespacio(Bterm4,Input_Round2,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
                else:
                    diccionario4=Bterm_Calculation_Vuelta4(Bt4,InputDimension,OutputDimension,l)
                #print(diccionario4)

                diccionarioM=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,{})
                if self.dictionary_subspace == True:
                    print(diccionarioM)
    
                #print(diccionarioM)

                
                max_abs_value_key = max(diccionarioM, key=lambda k: abs(diccionarioM[k]))
                fuerza=abs(diccionarioM[max_abs_value_key])


                DiccionarioT=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,DiccionarioT)
                #Matrix_Calculation(M,diccionario1,diccionario2,diccionario3,diccionario4, OutputDimension,InputDimension)
                if self.qubo_sampler == "qaoa":
                    result=QAOA_Solution2(diccionarioM,self.qaoa_num_reps)
                    Redondeo_Kernel[l], Redondeo_Bias[l]=Tensor_Redondeo2(result,Redondeo_Kernel,Redondeo_Bias,l,InputDimension)
                else:
                    if self.dwave_chain_strenght =="max":
                        max_abs_value_key = max(diccionarioM, key=lambda k: abs(diccionarioM[k]))
                        fuerza=abs(diccionarioM[max_abs_value_key])
                        result=Quantum_annealing_simulator(diccionarioM,self.qubo_sampler,self.dwave_num_reads,self.dwave_annealing_time,fuerza)

                        Redondeo_Kernel[l], Redondeo_Bias[l]=Tensor_Redondeo(result,Redondeo_Kernel,Redondeo_Bias,l,InputDimension)
                    else:
                        result=Quantum_annealing_simulator(diccionarioM,self.qubo_sampler,self.dwave_num_reads,self.dwave_annealing_time,self.dwave_chain_strenght)
                        Redondeo_Kernel[l], Redondeo_Bias[l]=Tensor_Redondeo(result,Redondeo_Kernel,Redondeo_Bias,l,InputDimension)

            tensors=[Redondeo_Kernel,Redondeo_Bias]
            
            
            return tensors

    def optimize(self) -> Tuple[List[np.ndarray], float]:
        """Optimization Method

        Returns:
            Tuple[List[np.ndarray], float]: Returns the optimized rounding policy and
             the final loss/cost
        """
        print("QUBOAnnealer.optimize()")
        print(
            "cfg access",
        )

        from itcl_quantizer.tensor_extractor.keras.layers.keras_dense import KerasDense

        if isinstance(self._layer, KerasDense):
            return self._qubo_dense(self._round_policy), 0.0

        return [], 0.0