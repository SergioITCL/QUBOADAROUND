import time
from itcl_quantization.quantization.operators import Quantization
from itcl_quantizer.quantizer.distributions.distribution import Distribution
import numpy as np
import math
import dimod
import random
import greedy
import neal
import pytest

# from dwave.system import DWaveSampler, EmbeddingComposite
# from dwave.system import LeapHybridSampler
# from dwave.inspector import show

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import *
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms import QAOA
from qiskit_aer import Aer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import BackendSampler
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram


def Input_Round_Calculation(Input,qinput_scale):
    '''Rounds to the nearest the dataset

    Args:
        Input, the input dataset in float
        qinput_scale, round-to-nearest quantization scale of the input
    Returns:
        Input_Round (np.array), Input matrix round-to-nearest quantization
   '''
    Input_Round=Input.copy()
    Input_Round=np.round(Input_Round/qinput_scale)
    return Input_Round

def dterm_Calculation(Bias,Kernel,InputRound,Output,qkernel_scale,qinput_scale,qbias_scale):
    ''' Calculates d_term from the ADAROUND problem
    Args:
    Bias (numpy, float), vias vector from the layer
    Kernel (numpy,float), weight matrix from the layer 
    InputRound(numpy,int), Input matrix round to nearest quantization
    Output(numpy,float), Output matrix from the layer
    qkernel_scale (float), round-to-nearest quantization scale kernel
    qinput_scale(float), round-to-nearest quantization scale of the input

    return:
    d term 
    '''
    dc=Output/(qkernel_scale*qinput_scale)-(qbias_scale/(qkernel_scale*qinput_scale))*np.floor(Bias/(qkernel_scale*qinput_scale))  
    ds=np.dot(np.floor(Kernel/qkernel_scale),InputRound.T)   
    d=dc-ds.T
    return d

def Bterm1_Calculation_Subespacio(Bterm1,InputRound,dterm,Indice_del_subespacio,Dimension_Input,Numero_de_datasets,qkernel_scale,qinput_scale):
    '''
    Calculates the coefficients of the first term of Bterm and stores them in a dictionary.

    Args:
        Diccionario1 (dictionary): A dictionary where the coefficients are stored.
        Bterm1 (numpy, float): An auxiliary variable to store the coefficients temporarily.
        InputRound (numpy, int): Input matrix rounded to the nearest quantization.
        dterm (numpy, float): Dterm matrix of the dterm coefficients.
        Indice_del_subespacio (int): Index of the subspace.
        Dimension_Input (int): Dimension of the input.
        Numero_de_datasets (int): Number of elements in the dataset.
        qkernel_scale (float): Round-to-nearest quantization scale of the kernel.
        qinput_scale (float): Round-to-nearest quantization scale of the input.

    Returns:
        A dictionary with the coefficients of the first term of Bterm.
    '''
    Diccionario1={}
    for s in range(0,Numero_de_datasets):  
        Bterm1[s]=(qkernel_scale*qinput_scale)**2*(1/Numero_de_datasets)*InputRound[s]*(InputRound[s]-2*dterm[s][Indice_del_subespacio])
    Suma = np.sum(Bterm1, axis=0)
    for i in range(0,Dimension_Input):
        Diccionario1[(Indice_del_subespacio*Dimension_Input+i+1,Indice_del_subespacio*Dimension_Input+i+1)]=Suma[i]
    return Diccionario1

def Bterm2_Calculation_Subespacio(Bterm2,InputRound,Indice_del_subespacio,Dimension_Input,Numero_de_datasets,qkernel_scale,qinput_scale):
    '''
    Calculates the coefficients of the second term of Bterm and stores them in a dictionary.

    Args:
        Diccionario2 (dictionary): A dictionary where the coefficients are stored.
        Bterm2 (numpy, float): An auxiliary variable to store the coefficients temporarily.
        InputRound (numpy, int): Input matrix rounded to the nearest quantization.
        Indice_del_subespacio (int): Index of the subspace.
        Dimension_Input (int): Dimension of the input.
        Numero_de_datasets (int): Number of elements in the dataset.
        qkernel_scale (float): Round-to-nearest quantization scale of the kernel.
        qinput_scale (float): Round-to-nearest quantization scale of the input.

    Returns:
        Diccionario2, A dictionary with the coefficients of the second term of Bterm.
        Bt2, An auxiliar numpy array used in the following calculations
    '''  
    Diccionario2={}
    for s in range(0, Numero_de_datasets):
        M=np.outer(InputRound[s].T,InputRound[s])
        
        np.fill_diagonal(M, 0)
        Bterm2[s]=np.ravel(M)
        Bt2 = (qkernel_scale*qinput_scale)**2*(1/Numero_de_datasets)*np.sum(Bterm2, axis=0)
        #print(Bt2)
        #Bt2 = Suma.reshape(Dimension_Input, Dimension_Input)
    for i in range(0,Dimension_Input):
        for j in range(0,Dimension_Input):
            if Bt2[j+i*Dimension_Input]==0:
                continue
            else:
                Diccionario2[(Dimension_Input*Indice_del_subespacio +i+1,Dimension_Input*Indice_del_subespacio +j+1)]=Bt2[j+i*Dimension_Input]
    return Diccionario2,Bt2

def Bterm_Calculation_Vuelta(Bt2,Dimension_Input,Indice_del_subespacio):
    Diccionario2={}
    for i in range(0,Dimension_Input):
        for j in range(0,Dimension_Input):
            if Bt2[j+i*Dimension_Input]==0:
                continue
            else:
                Diccionario2[(Dimension_Input*Indice_del_subespacio +i+1,Dimension_Input*Indice_del_subespacio +j+1)]=Bt2[j+i*Dimension_Input]
    
    return Diccionario2

def Bterm3_Calculation_Subespacio(Bterm3,dterm,Indice_del_subespacio,Dimension_Input,Dimension_Output,SB,SX,SW,Numero_de_datasets):
    '''
    Calculates the coefficients of the third term of Bterm and stores them in a dictionary.

    Args:
        Diccionario3 (dictionary): A dictionary where the coefficients are stored.
        Bterm3 (numpy, float): An auxiliary variable to store the coefficients temporarily.
        dterm (numpy, float): Dterm matrix of the dterm coefficients.
        Indice_del_subespacio (int): Index of the subspace.
        Dimension_Input (int): Dimension of the input.
        Dimension_Output (int): Dimension of the Output.
        Numero_de_datasets (int): Number of elements in the dataset.
        qkernel_scale (float): Round-to-nearest quantization scale of the kernel.
        qinput_scale (float): Round-to-nearest quantization scale of the input.

    Returns:
        A dictionary with the coefficients of the first term of Bterm.
    '''
    Diccionario3={}
    Bterm3 = (SX*SW)**2*(1/Numero_de_datasets)*(SB/(SX*SW))*((SB/(SX*SW))-2*dterm[:,Indice_del_subespacio])
    Bt3 = np.sum(Bterm3, axis=0)
    Diccionario3[(Dimension_Input*Dimension_Output+1+Indice_del_subespacio,Dimension_Input*Dimension_Output+1+Indice_del_subespacio)] = Bt3
    return Diccionario3

def Bterm4_Calculation_Subespacio(Bterm4,InputRound,Indice_del_subespacio,Dimension_Input,Dimension_Output,SB,SX,SW,Numero_de_datasets):
    Diccionario4={}
    for s in range(0,Numero_de_datasets):
        Bterm4[s]=(SW*SX)**2*(1/Numero_de_datasets)*(SB/(SX*SW))*InputRound[s]
    Bt4 = np.sum(Bterm4, axis=0)
    #print(Bt4)
    for i in range(0,Dimension_Input):
        if Bt4[i]==0:
            continue
        else:
            Diccionario4[(Indice_del_subespacio*Dimension_Input+i+1,Dimension_Input*Dimension_Output+1+Indice_del_subespacio)]=Bt4[i]
            Diccionario4[(Dimension_Input*Dimension_Output+1+Indice_del_subespacio,Indice_del_subespacio*Dimension_Input+i+1)]=Bt4[i]
    return Diccionario4,Bt4

def Bterm_Calculation_Vuelta4(Bt4,Dimension_Input,Dimension_Output,Indice_del_subespacio):
    Diccionario4={}
    for i in range(0,Dimension_Input):
        if Bt4[i]==0: 
            continue
        else:
            Diccionario4[(Indice_del_subespacio*Dimension_Input+i+1,Dimension_Input*Dimension_Output+1+Indice_del_subespacio)]=Bt4[i]
            Diccionario4[(Dimension_Input*Dimension_Output+1+Indice_del_subespacio,Indice_del_subespacio*Dimension_Input+i+1)]=Bt4[i]
    return Diccionario4




def Quantum_annealing_simulator(Diccionario,qubo_sampler,dwave_num_reads,dwave_annealing_time,fuerza):
    J=Diccionario

    h={}
    problem = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.BINARY)
    api_key = 'DEV-caa8e3d0f6dbfb0175cf148a15109b11979d3329'
    if qubo_sampler == "dwave":
        
        
        sampler = EmbeddingComposite(DWaveSampler(token=api_key))
        if fuerza == 0:
            result = sampler.sample(problem, num_reads=dwave_num_reads,annealing_time=dwave_annealing_time)
        else:
            result = sampler.sample(problem, num_reads=dwave_num_reads,annealing_time=dwave_annealing_time,chain_strength=fuerza)


    elif qubo_sampler == "neal":

        solver = neal.SimulatedAnnealingSampler()
        result = solver.sample(problem)

  

    elif qubo_sampler =="hybrid":
        solver = LeapHybridSampler(token=api_key)
        result=solver.sample(problem)


    elif qubo_sampler == "brute_force":
        a=1
        #solver2 = dimod.ExactSolver()
        #response = solver2.sample(problem)
        #min_energy_sample = next(response.samples())
        #min_energy = next(response.data()).energy
        #print(min_energy_sample,min_energy)
    else:
        raise ValueError(f"Sampler '{qubo_sampler}' no reconocido.")

    return result


def Unir_Diccionarios(Diccionario1,Diccionario2,Diccionario3,Diccionario4,Diccionario):
    '''
    Merge all the dictionaries containing all the B-term coefficients.
    Args:
        Dictionary1 (Dictionary): Contains the coefficients of B-term 1.
        Dictionary2 (Dictionary): Contains the coefficients of B-term 2.
        Dictionary3 (Dictionary): Contains the coefficients of B-term 3.
        Dictionary4 (Dictionary): Contains the coefficients of B-term 4.
        Dictionary (Dictionary): An auxiliary dictionary to store all the B-term coefficients.
    Return:
        A dictionary storing all the coefficients of the B-term calculation.
    '''

    Diccionario.update(Diccionario1)
    Diccionario.update(Diccionario2)
    Diccionario.update(Diccionario3)
    Diccionario.update(Diccionario4)
    return Diccionario
    
def Tensor_Redondeo (Resultado_Annealing,Resultado_Annealing_pesos,Resultado_Annealing_bias,Indice_del_subespacio,Dimension_Input):
    Dmin=np.array(Resultado_Annealing.record.sample[0])
    Resultado_Annealing_pesos[Indice_del_subespacio]=Dmin[0:Dimension_Input]
    Resultado_Annealing_bias[Indice_del_subespacio]=Dmin[Dimension_Input]
    return Resultado_Annealing_pesos[Indice_del_subespacio],Resultado_Annealing_bias[Indice_del_subespacio]

def Matrix_Calculation(M,Diccionario1,Diccionario2,Diccionario3,Diccionario4, Dimension_Output, Dimension_Input):
    for i in range(0,Dimension_Output*Dimension_Input+Dimension_Output):
        for j in range(0,Dimension_Output*Dimension_Input+Dimension_Output):
            if (i==j) and ((i+1,j+1) in Diccionario1):
                M[i][j]=Diccionario1[(i+1,j+1)]
            if (i+1,j+1) in Diccionario2:
                M[i][j]=Diccionario2[(i+1,j+1)]
            if (i==j) and ((i+1,j+1) in Diccionario3):
                M[i][j]=Diccionario3[(i+1,j+1)]
            if (i+1,j+1) in Diccionario4:
                M[i][j]=Diccionario4[(i+1,j+1)]
    return np.savetxt("Matrix.txt", M, fmt='%.8f',delimiter=",")


def Tensor_Redondeo2 (Resultado_QAOA,Resultado_Annealing_pesos,Resultado_Annealing_bias,Indice_del_subespacio,Dimension_Input):
    Dmin=Resultado_QAOA
    Resultado_Annealing_pesos[Indice_del_subespacio]=Dmin[0:Dimension_Input]
    Resultado_Annealing_bias[Indice_del_subespacio]=Dmin[Dimension_Input]
    return Resultado_Annealing_pesos[Indice_del_subespacio],Resultado_Annealing_bias[Indice_del_subespacio]

def QAOA_Solution2(Diccionario,num_reps):
    Indice_Maximo = max(max(key) for key in Diccionario)
    Diccionario_Primado={}
    for i in range(0,Indice_Maximo+1):
        for j in range(0,Indice_Maximo+1):
            if (i,j) in Diccionario:
                Diccionario_Primado[(f'{i}',f'{j}')]=Diccionario[(i,j)]
 
    
    qp = QuadraticProgram()
    for i in range(1,Indice_Maximo):
        if (i,i) in Diccionario:
            qp.binary_var(f'{i}')
    qp.binary_var(f'{Indice_Maximo}')
    qp.minimize(quadratic = Diccionario_Primado)



    inicio = time.time()
    sim = Aer.get_backend('aer_simulator_statevector_gpu')
    sampler = BackendSampler(sim)
    spsa = SPSA(maxiter=250)
    qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=num_reps)

    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    result2 = qaoa_optimizer.solve(qp)
    res2 = np.array(result2.x)
    print('QAOA')
    print(result2)
    fin = time.time()
    print('tiempo gpu',fin-inicio)

    return res2


def random_qubo_problem(number_variables):
    np.random.seed(42)
    kernel = np.random.normal( 0,1, size=(10,number_variables))
    bias = np.random.normal( -1,1, size=(10,))
    input_f = np.random.exponential(1,size=(1000, number_variables))

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

    return kernel,bias,input_f,output_f, kernel_s,input_s,bias_s
