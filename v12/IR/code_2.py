'''
 Code for QPT
'''

import numpy as np
from numpy.lib.shape_base import kron
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, qasm, Aer, execute

def vec(rho):
    return np.reshape(rho,(len(rho)**2,1),order='F')
    
def unvec(S_rho):
    return np.reshape(S_rho,(int(len(S_rho)**0.5),int(len(S_rho)**0.5)),order='F')
    
test_rho = np.matrix([[1, 2], [3, 4]])
S_rho = vec(test_rho)
rho = unvec(S_rho)
# print(test_rho,'\n',S_rho,'\n',rho)

def QProcess():
    ensemble = 2
    qproc = []
    for i in range(0,ensemble):
        qproc.append([QuantumCircuit(2, 1),0])
    qproc[0][1] = 0.5
    qproc[0][0].h(0)
    qproc[0][0].h(1)
    qproc[1][1] = 0.5
    qproc[1][0].h(0)
    qproc[1][0].cx(0,1)
    return qproc

# E = QProcess()
# for i in E:
#     print("Probability: ",i[1])
#     print(i[0])
#     print(qi.Choi(i[0]))
#     print(qi.Kraus(i[0]))

from copy import deepcopy

def QStatePrep(qcirc):
    qcirc.h(0)
    qcirc.barrier()
    return

qcirc = QuantumCircuit(2, 2)
QStatePrep(qcirc)
rho = qi.DensityMatrix.from_instruction(qcirc)
print(qcirc)
print(rho.data)

from math import pi

def PostRot(ps,qcirc):
    P0 = [[1,0],[0,1]]
    P1 = [[0,1],[1,0]]
    P2 = [[0,complex(0,-1)],[complex(0,1),0]]
    P3 = [[1,0],[0,-1]]
    B = 1
    q = 0
    for i in range(len(ps)-1,-1,-1):
        if ps[i] == '0':        # I
            B = kron(P0,B)
        elif ps[i] == '1':      # X
            B = kron(P1,B)
            qcirc.ry(-pi/2,q)
        elif ps[i] == '2':      # Y
            B = kron(P2,B)
            qcirc.rx(pi/2,q)
        elif ps[i] == '3':      # Z
            B = kron(P3,B)
        q += 1
    qcirc.barrier()
    qcirc.measure([0,1], [0,1])
    return(B,qcirc)

def toStr(n,base):
   convertString = "0123456789ABCDEF"
   if n < base:
      return convertString[n]
   else:
      return toStr(n//base,base) + convertString[n%base]


import qiskit

for i in range(0,16):
    ps = toStr(i,4).zfill(2)
    B,qcircB = PostRot(ps,deepcopy(qcirc))
    job = qiskit.execute(qcircB, Aer.get_backend('qasm_simulator'), shots=100)
    result = job.result()
    counts = result.get_counts(qcircB)
    print(ps,"---> ")
    print(np.trace(B.conj().T * rho.data))
    print(counts)
    
# make a 1-qubit quantum state
# make a density matrix of the quantum state
# measure the qubit in I,X,Y,Z basis for 100 trails in each
# 


"""
Uses the math described here (http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf)
to construct the density matrix from the tomographic histogram
"""
TOMOGRAPHY_GATES = OrderedDict([('i','Identity'),
								('x','Pauli-X'),
								('y','Pauli-Y'),
								('z','Pauli-Z')])

sg0 = [[1, 0], [0, 1]] # Identity
sg1 = [[0, 1], [1, 0]] # Pauli-X
sg2 = [[0,-1j],[1j,0]] # Pauli-Y
sg3 = [[1, 0], [0,-1]] # Pauli-Z
sigmas = {'i':sg0, 'x':sg1, 'y':sg2, 'z':sg3}

eig0 = [1,  1] # Eigenvalues of sg0
eig1 = [1, -1] # Eigenvalues of sg1, sg2, sg3
eigens = {'i':eig0, 'x':eig1, 'y':eig1, 'z':eig1}

def generate_density_matrix(hist):
    dm = np.zeros((2**NUM_QUBIT, 2**NUM_QUBIT)) * 0j
    idx = 0
    for bases in product(TOMOGRAPHY_GATES.keys(), repeat=NUM_QUBIT):
        ppm = [1]
        eigs = [1]
        for b in bases:
            ppm = np.kron(sigmas[b], ppm)
            eigs = np.kron(eigens[b], eigs)

        Si = sum(np.multiply(eigs, hist[idx])) # Multiply each sign to its respective probability and sum
        dm += Si*ppm
        idx += 1

    dm /= (2**NUM_QUBIT)
    return dm
