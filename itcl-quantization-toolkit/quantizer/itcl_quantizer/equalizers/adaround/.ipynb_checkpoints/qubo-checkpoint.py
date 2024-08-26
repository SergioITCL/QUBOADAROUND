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
    ) -> None:
        super().__init__()

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
  
        dense_layer: KerasDense = self._layer  # type: ignore
        kernel, bias = dense_layer.get_weights()
        input_f = self._input_data
        input_f = np.array(input_f)
        output_f = self._results.input_data
        output_f = np.array(output_f)
        # output_f = TODO
         
        kernel_roundp = tensors[0]  # todo: update
        bias_roundp = tensors[1]  # todo: update
        
        assert self._results.operators

        q_dense_op = self._results.operators[0]
    
        q_kernel = q_dense_op.inputs[1]
        kernel_s, kernel_zp = q_kernel.scale, q_kernel.zero_point
        q_bias = q_dense_op.inputs[2]
        bias_s, bias_zp = q_bias.scale, q_bias.zero_point
        q_input = q_dense_op.inputs[0]
        input_s, input_zp = q_input.scale, q_input.zero_point

        '''
        zeros = np.zeros((64,13))
        print(kernel)
        print('kernelzp',kernel_zp)
        print(kernel.shape)
        for i in range(0,64):
            for j in range(0,13):
                if kernel[i][j]/kernel_s-np.round(kernel[i][j]/kernel_s)<0:
                    zeros[i][j]=1
        print(zeros)
        '''

        '''
        print(input_f)
        if 'dense' in q_kernel.name:
            print(input_f.shape)
            np.savetxt("input.csv", input_f, delimiter=",")
        '''
        '''
    
        umbral = 0.1

        # Identificar las filas que cumplen con la condición del umbral
        filas_a_mantener = np.all(input_f <= umbral, axis=1)

        # Seleccionar las filas que deseas mantener en la matriz original
        input_f = input_f[filas_a_mantener]
        '''
   
    
        '''
        n=49
        kernel = np.random.normal( 0,1, size=(10,n))
        bias = np.random.normal( -1,1, size=(10,))
        #input_f = np.random.exponential(scale=10.0, size=(1000, n))
        input_f = np.random.exponential(1,size=(1000, n))


        #input_f = np.random.normal(-1, 1, size=(1000,n))
        #print(input_f)
        #input_f = np.zeros((1000, n))
        #input_f[:, 2] = -1 * np.random.rand(1000)
        #input_f[:, 1] = -1 * np.random.rand(1000)

        #print(input_f)
        output_f = np.zeros((input_f.shape[0], bias.shape[0]))
        for i in range(0,input_f[:,0].shape[0]):
            output_f[i]=kernel.dot(input_f[i])+bias
        
        
        
        q2_kernel = Quantization('int8')
        kernel_dist = Distribution(kernel)

        kernel_s, w_zpk = kernel_dist.quantize(q2_kernel,
             force_zp=0, symmetric=False
        )
        q2_input = Quantization('int8')
        input_dist= Distribution(input_f)
        input_s, w_zpi = input_dist.quantize(q2_input, symmetric=False
        )
        bias_s=input_s*kernel_s
        '''

        '''
        DiccionarioT={}
        AdaInt=ADAInt(input_f,kernel,bias,output_f,input_s,kernel_s,bias_s)
        Redondeo_Kernel=np.zeros((AdaInt.output_dimension(),AdaInt.input_dimension()))
        Redondeo_Bias=np.zeros((AdaInt.output_dimension()))
        M = np.zeros((AdaInt.output_dimension()*AdaInt.input_dimension()+AdaInt.output_dimension(), AdaInt.output_dimension()*AdaInt.input_dimension()+AdaInt.output_dimension()))
        for l in range(0,AdaInt.output_dimension()):
            print('l',l)
            Dicionario1={}
            Dicionario2={}
            diccionario2={}
            Dicionario3={}
            Dicionario4={}
            diccionario4={}
            Dicionario={}
            diccionario1=AdaInt.bterm1_calculation_subespacio(Dicionario1,l)
            if l==0:           
            
                diccionario2, Bt2=AdaInt.bterm2_calculation_subespacio(Dicionario2,l)
            else:
                diccionario2=AdaInt.bterm2_calculation_repeticion(Bt2,diccionario2,l)
            diccionario3=AdaInt.bterm3_calculation_subespacio(Dicionario3,l)
            if l==0:
                diccionario4,bt4=AdaInt.bterm4_calculation_subespacio(Dicionario4,l)
            else:
                diccionario4=AdaInt.bterm4_calculation_repeticion(diccionario4,bt4,l)
            
            diccionario_unido=AdaInt.unir_diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,Dicionario)
            #AdaInt.matrix_calculation(M,diccionario1,diccionario2,diccionario3,diccionario4)
            result=AdaInt.quantum_annealing(diccionario_unido)
            print(result)
            #result2=AdaInt.qaoa_solution(diccionario_unido)
            Redondeo_Kernel[l], Redondeo_Bias[l]=AdaInt.tensor_redondeo(result,Redondeo_Kernel,Redondeo_Bias,l)
        tensors=[Redondeo_Kernel,Redondeo_Bias]
        #print(tensors)
        #np.savetxt("input_ele1.csv", input_f, delimiter=",")

        '''
      

        xdim=input_f[0].shape
        InputDimension=xdim[0]
        ydim=bias.shape
        OutputDimension=ydim[0]
        Numero_datasets=input_f[:,0].shape[0]
        Bterm1=np.zeros((Numero_datasets,InputDimension))
        Bterm2=np.zeros((Numero_datasets,InputDimension*InputDimension))
        Bterm3=np.zeros((Numero_datasets,1))
        Bterm4=np.zeros((Numero_datasets,InputDimension))
        Redondeo_Kernel=np.zeros((OutputDimension,InputDimension))
        Redondeo_Bias=np.zeros((OutputDimension))
        Redondeo_Kernel2=np.zeros((OutputDimension,InputDimension))
        Redondeo_Bias2=np.zeros((OutputDimension))
        M = np.zeros((OutputDimension*InputDimension+OutputDimension, OutputDimension*InputDimension+OutputDimension))
        DiccionarioT={}
        
        #inicio = time.time()
        Input_Round2=Input_Round_Calculation(input_f,input_s)
        dTermino2=dterm_Calculation(bias,kernel,Input_Round2,output_f,kernel_s,input_s,bias_s)
        
        for l in range(0,OutputDimension):
            print('l',l)
            Dicionario1={}
            Dicionario2={}
            diccionario2={}
            Dicionario3={}
            Dicionario4={}
            diccionario4={}
            Dicionario={}
            diccionario1=Bterm1_Calculation_Subespacio(Dicionario1,Bterm1,Input_Round2,dTermino2,l,InputDimension,Numero_datasets,kernel_s,input_s)
            #print(diccionario11)
            if l==0:
                diccionario2,Bt2=Bterm2_Calculation_Subespacio(Dicionario2,Bterm2,Input_Round2,l,InputDimension,Numero_datasets,kernel_s,input_s)
            else:
                diccionario2=Bterm_Calculation_Vuelta(Bt2,diccionario2,InputDimension,l)
            diccionario3=Bterm3_Calculation_Subespacio(Dicionario3,Bterm3,dTermino2,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
            #print(diccionario3)
            if l==0:
                diccionario4,Bt4=Bterm4_Calculation_Subespacio(Dicionario4,Bterm4,Input_Round2,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
            else:
                diccionario4=Bterm_Calculation_Vuelta4(Dicionario4,Bt4,InputDimension,OutputDimension,l)
            #print(diccionario4)
            diccionarioM=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,Dicionario)
            
            max_abs_value_key = max(diccionarioM, key=lambda k: abs(diccionarioM[k]))
            fuerza=abs(diccionarioM[max_abs_value_key])


            DiccionarioT=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,DiccionarioT)
            #Matrix_Calculation(M,diccionario1,diccionario2,diccionario3,diccionario4, OutputDimension,InputDimension)
            result=Quantum_annealing_simulator(diccionarioM,fuerza)
            #print('result1',result)
            result2=QAOA_Solution2(diccionarioM)
            Redondeo_Kernel[l], Redondeo_Bias[l]=Tensor_Redondeo2(result2,Redondeo_Kernel,Redondeo_Bias,l,InputDimension)
        #print('diccionario1',DiccionarioT)  
        #print(DiccionarioT)
        
        #Redondeo_Kernel = np.random.choice([0, 1], size=(OutputDimension, InputDimension))
        #Redondeo_Bias = np.random.choice([0, 1], size=(OutputDimension))
        # Convertir la matriz en un vector
        
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