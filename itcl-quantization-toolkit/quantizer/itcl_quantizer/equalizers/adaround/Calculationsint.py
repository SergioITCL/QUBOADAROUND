import time
import numpy as np
import math
import dimod
import random
import greedy
import neal
import pytest
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler

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

class ADAInt:

    def __init__(self, input,kernel,bias,output,input_scale,kernel_scale,bias_scale):
            self.input = input
            self.kernel = kernel
            self.bias = bias
            self.output = output
            self.input_scale = input_scale
            self.kernel_scale = kernel_scale
            self.bias_scale = bias_scale

    def input_dimension(self):
        xdim=self.input[0].shape
        return xdim[0]
    
    def output_dimension(self):
        ydim=self.bias.shape
        return ydim[0]
    
    def numero_datasets(self):
         return self.input[:,0].shape[0]

    def input_round_calculation(self):
        '''Rounds to the nearest the dataset

        Args:
        Input, the input dataset in float
        qinput_scale, round-to-nearest quantization scale of the input
        Returns:
        Input_Round (np.array), Input matrix round-to-nearest quantization
        '''
        Input_Round=self.input.copy()
        Input_Round=np.round(Input_Round/self.input_scale)
        return Input_Round
    def dterm_calculation(self):
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
        dc=self.output/(self.kernel_scale*self.input_scale)-(self.bias_scale/(self.kernel_scale*self.input_scale))*np.floor(self.bias/(self.kernel_scale*self.input_scale))  
        ds=np.dot(np.floor(self.kernel/self.kernel_scale),self.input_round_calculation().T)   
        d=dc-ds.T
        return d
    def bterm1_calculation_subespacio(self,diccionario1,indice_del_subespacio):
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
        bterm1=np.zeros((self.numero_datasets(),self.input_dimension()))
        for s in range(0,self.numero_datasets()):  
            bterm1[s]=(self.kernel_scale*self.input_scale)**2*(1/self.numero_datasets())*self.input_round_calculation()[s]*(self.input_round_calculation()[s]-2*self.dterm_calculation()[s][indice_del_subespacio])
        Suma = np.sum(bterm1, axis=0)
        for i in range(0,self.input_dimension()):
            diccionario1[(indice_del_subespacio*self.input_dimension()+i+1,indice_del_subespacio*self.input_dimension()+i+1)]=Suma[i]
        return diccionario1     
      
    def bterm2_calculation_subespacio(self, diccionario2,indice_del_subespacio):
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
        bterm2=np.zeros((self.numero_datasets(),self.input_dimension()*self.input_dimension()))
        for s in range(0, self.numero_datasets()):
            M=np.outer(self.input_round_calculation()[s].T,self.input_round_calculation()[s])
            
            np.fill_diagonal(M, 0)
            bterm2[s]=np.ravel(M)
            Bt2 = (self.kernel_scale*self.input_scale)**2*(1/self.numero_datasets())*np.sum(bterm2, axis=0)
            #print(Bt2)
            #Bt2 = Suma.reshape(Dimension_Input, Dimension_Input)
        for i in range(0,self.input_dimension()):
            for j in range(0,self.input_dimension()):
                if Bt2[j+i*self.input_dimension()]==0:
                    continue
                else:
                    diccionario2[(self.input_dimension()*indice_del_subespacio +i+1,self.input_dimension()*indice_del_subespacio +j+1)]=Bt2[j+i*self.input_dimension()]
        return diccionario2,Bt2 

    def bterm2_calculation_repeticion(self,Bt2, diccionario2,indice_del_subespacio):
        for i in range(0,self.input_dimension()):
            for j in range(0,self.input_dimension()):
                if Bt2[j+i*self.input_dimension()]==0:
                    continue
                else:
                    diccionario2[(self.input_dimension()*indice_del_subespacio +i+1,self.input_dimension()*indice_del_subespacio +j+1)]=Bt2[j+i*self.input_dimension()]
        return diccionario2           
    
    def bterm3_calculation_subespacio(self,diccionario3,indice_del_subespacio):
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
        
        bterm3 = (self.input_scale*self.kernel_scale)**2*(1/self.numero_datasets())*(self.bias_scale/(self.input_scale*self.kernel_scale))*((self.bias_scale/(self.input_scale*self.kernel_scale))-2*self.dterm_calculation()[:,indice_del_subespacio])
        bt3 = np.sum(bterm3, axis=0)
        diccionario3[(self.input_dimension()*self.output_dimension()+1+indice_del_subespacio,self.input_dimension()*self.output_dimension()+1+indice_del_subespacio)] = bt3
        return diccionario3   
    
    def bterm4_calculation_subespacio(self,diccionario4,indice_del_subespacio):
        bterm4=np.zeros((self.numero_datasets(),self.input_dimension()))
        for s in range(0,self.numero_datasets()):
            bterm4[s]=(self.kernel_scale*self.input_scale)**2*(1/self.numero_datasets())*(self.bias_scale/(self.kernel_scale*self.input_scale))*self.input_round_calculation()[s]
        Bt4 = np.sum(bterm4, axis=0)
        #print(Bt4)
        for i in range(0,self.input_dimension()):
            if Bt4[i]==0:
                continue
            else:
                diccionario4[(indice_del_subespacio*self.input_dimension()+i+1,self.input_dimension()*self.output_dimension()+1+indice_del_subespacio)]=Bt4[i]
                diccionario4[(self.input_dimension()*self.output_dimension()+1+indice_del_subespacio,indice_del_subespacio*self.input_dimension()+i+1)]=Bt4[i]
        return diccionario4,Bt4
    
    def bterm4_calculation_repeticion(self,diccionario4,bt4,indice_del_subespacio):
        for i in range(0,self.input_dimension()):
            if bt4[i]==0:
                continue
            else:
                diccionario4[(indice_del_subespacio*self.input_dimension()+i+1,self.input_dimension()*self.output_dimension()+1+indice_del_subespacio)]=bt4[i]
                diccionario4[(self.input_dimension()*self.output_dimension()+1+indice_del_subespacio,indice_del_subespacio*self.input_dimension()+i+1)]=bt4[i]
        return diccionario4
    
    def quantum_annealing(self,diccionario):
        J=diccionario
        h={}
        problem=dimod.BinaryQuadraticModel(h,J,0.0,dimod.BINARY)
        api_key = 'DEV-af84b145c783a09fc23798ac31bee43d343a6787'
        '''
        sampler2 = EmbeddingComposite(DWaveSampler(token=api_key))
        problem = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.BINARY)
        #print(problem)
        print('cuantico')
        
        result2 = sampler2.sample(problem, num_reads=30,annealing_time=20)
        print(result2)
        '''
        #solver = greedy.SteepestDescentSolver()
        #result = solver.sample(problem, num_reads = 50)
        #print('dwave')
        #print('SteepestDEscentSolver')
        #print(result)
        
        
        solver = neal.SimulatedAnnealingSampler()
        result3 = solver.sample(problem)
        #print('Neal')
        #print(result3)
        '''
        solver = LeapHybridSampler(token=api_key)
        result4=solver.sample(problem)
        print('hybrid')
        print(result4)
        '''
        '''
        solver2 = dimod.ExactSolver()
        response = solver2.sample(problem)
        min_energy_sample = next(response.samples())
        min_energy = next(response.data()).energy
        print(min_energy_sample,min_energy)
        '''
        return result3

    def unir_diccionarios(self,diccionario1,diccionario2,diccionario3,diccionario4,diccionario):
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
        diccionario.update(diccionario1)
        diccionario.update(diccionario2)
        diccionario.update(diccionario3)
        diccionario.update(diccionario4)
        return diccionario      
    
    def tensor_redondeo(self,resultado_annealing,resultado_annealing_pesos,resultado_annealing_bias,indice_del_subespacio):
        Dmin=np.array(resultado_annealing.record.sample[0])
        resultado_annealing_pesos[indice_del_subespacio]=Dmin[0:self.input_dimension()]
        resultado_annealing_bias[indice_del_subespacio]=Dmin[self.input_dimension()]
        return resultado_annealing_pesos[indice_del_subespacio],resultado_annealing_bias[indice_del_subespacio]
    
    def matrix_calculation(self,M,diccionario1,diccionario2,diccionario3,diccionario4):
        for i in range(0,self.output_dimension()*self.input_dimension()+self.output_dimension()):
            for j in range(0,self.output_dimension()*self.input_dimension()+self.output_dimension()):
                if (i==j) and ((i+1,j+1) in diccionario1):
                    M[i][j]=diccionario1[(i+1,j+1)]
                if (i+1,j+1) in diccionario2:
                    M[i][j]=diccionario2[(i+1,j+1)]
                if (i==j) and ((i+1,j+1) in diccionario3):
                    M[i][j]=diccionario3[(i+1,j+1)]
                if (i+1,j+1) in diccionario4:
                    M[i][j]=diccionario4[(i+1,j+1)]
        return np.savetxt("Matrix2.txt", M, fmt='%.8f',delimiter=",")
    
    def qaoa_solution(self,diccionario):
        Indice_Maximo = max(max(key) for key in diccionario)
        Diccionario_Primado={}
        for i in range(0,Indice_Maximo+1):
            for j in range(0,Indice_Maximo+1):
                if (i,j) in diccionario:
                    Diccionario_Primado[(f'{i}',f'{j}')]=diccionario[(i,j)]
        #print(Diccionario_Primado)
        
        qp = QuadraticProgram()
        for i in range(1,Indice_Maximo):
            if (i,i) in diccionario:
                qp.binary_var(f'{i}')
        qp.binary_var(f'{Indice_Maximo}')
        qp.minimize(quadratic = Diccionario_Primado)
        #print(qp.export_as_lp_string())

        '''
        np_solver = NumPyMinimumEigensolver()
        np_optimizer = MinimumEigenOptimizer(np_solver)
        result = np_optimizer.solve(qp)
        res1=np.array(result.x)
        print('precise')
        print(result)
        '''
        
        inicio = time.time()
        sim = Aer.get_backend('aer_simulator_statevector')
        #sampler = QuantumInstance(backend=sim, shots=200)
        sampler = BackendSampler(sim)
        #sampler=Sampler()
        #sim = AerSimulator(method='statevector’, device='GPU')
        spsa = SPSA(maxiter=250)
        qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=3)
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        result2 = qaoa_optimizer.solve(qp)
        res2 = np.array(result2.x)
        print('QAOA')
        print(result2)
        fin = time.time()
        print('tiempo gpu',fin-inicio)
        '''
        inicio = time.time()
        sim = Aer.get_backend('aer_simulator_statevector')
        #sampler = QuantumInstance(backend=sim, shots=200)
        sampler = BackendSampler(sim)
        #sampler=Sampler()
        #sim = AerSimulator(method='statevector’, device='GPU')
        spsa = SPSA(maxiter=250)
        qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=1)
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        result2 = qaoa_optimizer.solve(qp)
        res2 = np.array(result2.x)
        print('QAOA')
        print(result2)
        fin = time.time()
        
        print('tiempo cpu',fin-inicio)
        '''
        return res2    
        
    def tensor_redondeo2(self,resultado_qaoa,resultado_annealing_pesos,resultado_annealing_bias,indice_del_subespacio):
        Dmin=resultado_qaoa
        resultado_annealing_pesos[indice_del_subespacio]=Dmin[0:self.input_dimension()]
        resultado_annealing_bias[indice_del_subespacio]=Dmin[self.input_dimension()]
        return resultado_annealing_pesos[indice_del_subespacio],resultado_annealing_bias[indice_del_subespacio]
