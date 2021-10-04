import numpy as np

import qiskit.quantum_info as qi

class metrics:

	'''
	Hamming distance between two strings
	https://en.wikipedia.org/wiki/Hamming_distance
	'''
	def DeltaHD(self, config_i, config_j):
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
	def DeltaTD(self, dm_i, dm_j):
		dist = np.real(np.trace(np.sqrt(np.matmul((dm_i - dm_j).conjugate().transpose(),dm_i - dm_j))) / 2)
		return dist

	'''
	Bures distance between two density matrices
	https://en.wikipedia.org/wiki/Bures_metric
	Bures metric or Helstrom metric defines an infinitesimal distance between density matrix operators defining quantum states. 
	It is a quantum generalization of the Fisher information metric, and is identical to the Fubini–Study metric when restricted to the pure states alone.
	'''
	def DeltaBD(self, dm_i, dm_j):
		fid = np.trace(np.sqrt( np.matmul(np.sqrt(dm_i), np.matmul(dm_j,np.sqrt(dm_i))) ))**2
		dist = np.sqrt(2*(1-np.sqrt(fid))) 
		return dist

	'''
	Diamond distance between two density matrices
	https://en.wikipedia.org/wiki/Diamond_norm
	'''
	def DeltaDD(self, dm_i, dm_j):
		dist = qi.diamond_norm(dm_i-dm_j)
		return dist