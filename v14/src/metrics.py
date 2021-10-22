import numpy as np

import qiskit.quantum_info as qi

class metrics:

	'''
	Hamming distance between two strings
	https://en.wikipedia.org/wiki/Hamming_distance
	'''
	def DeltaH(self, config_i, config_j):
		dist = 0
		for (i,j) in zip(config_i,config_j):
			if i!=j:
				dist += 1
		dist = len(config_i) - dist
		return dist
		
	'''
	Trace distance between two density matrices
	https://en.wikipedia.org/wiki/Trace_distance
	'''
	def DeltaT(self, dm_i, dm_j):
		diff = dm_i - dm_j
		diff_dag = diff.conjugate().transpose()
		dist = np.real(0.5* np.trace(np.sqrt(np.matmul(diff_dag,diff))))
		return dist

	'''
	Bures distance between two density matrices
	https://en.wikipedia.org/wiki/Bures_metric
	Bures metric or Helstrom metric defines an infinitesimal distance between density matrix operators defining quantum states. 
	It is a quantum generalization of the Fisher information metric, and is identical to the Fubini–Study metric when restricted to the pure states alone.
	'''
	def DeltaB(self, dm_i, dm_j):
		fid = np.trace(np.sqrt( np.matmul(np.sqrt(dm_i), np.matmul(dm_j,np.sqrt(dm_i))) ))**2
		dist = np.sqrt(2*(1-np.sqrt(fid))) 
		return dist

	'''
	Hilbert-Schmidt distance between two density matrices
	https://arxiv.org/abs/1911.00277
	'''
	def DeltaHS(self, dm_i, dm_j):
		dist = np.real(np.trace((dm_i - dm_j)**2))
		return dist

	'''
	Diamond distance between two density matrices
	https://en.wikipedia.org/wiki/Diamond_norm
	'''
	def DeltaD(self, dm_i, dm_j):
		dist = qi.diamond_norm(dm_i-dm_j)
		return dist


'''
Test properties of Distance Measures
E.g. distance between the complete mixed state and a pure state should not be zero
'''

from qiskit import QuantumCircuit

# Completely Mixed State
A = np.array([[0.25,0,0,0], 
     [0,0.25,0,0], 
     [0,0,0.25,0], 
     [0,0,0,0.25]])

# Hadamard Operator (Plus State)
B = np.array([[0.25,0.25,0.25,-0.25], 
     [0.25,0.25,0.25,-0.25], 
     [0.25,0.25,0.25,-0.25], 
     [-0.25,-0.25,-0.25,0.25]])

# Hadamard Operator (Plus State)
D = np.array([[1,0,0,0], 
     [0,0,0,0], 
     [0,0,0,0], 
     [0,0,0,0]])

qcirc = QuantumCircuit(2, 2)
qcirc.h(1)
qcirc.cx(1,0)
qcirc.h(0)
C = qi.DensityMatrix.from_instruction(qcirc).data

qcirc = QuantumCircuit(2, 2)
qcirc.h(1)
qcirc.cx(1,0)
E = qi.DensityMatrix.from_instruction(qcirc).data
print(C) 

M = metrics()
# print("T: ",M.DeltaT(A,B))
print("T: ",M.DeltaT(A,C))
print("B: ",M.DeltaB(A,C))
print("HS: ",M.DeltaHS(A,C))
# print("H: ",M.DeltaH(A,B))


print("T: ",M.DeltaT(A,E))
print("B: ",M.DeltaB(A,E))
print("HS: ",M.DeltaHS(A,E))


		