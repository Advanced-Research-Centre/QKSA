"""
Ancilla Assisted Quantum Process Tomography on n-qubits
Construct Choi density matrix of process from tomographic trials
"""

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer, execute

import numpy as np
from math import pi
from numpy.lib.shape_base import kron
from copy import deepcopy

process_qubits = 1
qubits = 2*process_qubits   # for AAQPT
hsdim = 2**qubits           # Hilbert space dimension
trials = 1024               # Number of tomographic trials for each observable
aprx = 2                    # Places of decimal to round the reconstructed density matrix


'''
The "black-box" quantum process is defined here
'''
def QProcess(qcirc, qreg):
    qcirc.barrier()
    qcirc.h(qreg[0])
    qcirc.barrier()
    return

'''
Make entangled state
'''
def EntangleAncilla(qcirc):
    qcirc.barrier()
    qcirc.h(1)
    qcirc.cx(1,0)
    qcirc.barrier()
    return

'''
Add post-rotation gates and eigenvalue sign for Quantum State Tomography
'''
def PostRot(ps,qcirc):
    P0 = [[1,0],[0,1]]                          # I
    P1 = [[0,1],[1,0]]                          # Pauli-X
    P2 = [[0,complex(0,-1)],[complex(0,1),0]]   # Pauli-Y
    P3 = [[1,0],[0,-1]]                         # Pauli-Z
    E0 = [1, 1]                                 # Eigenvalues of P0
    E1 = [1, -1]                                # Eigenvalues of P1
    E2 = [1, -1]                                # Eigenvalues of P2
    E3 = [1, -1]                                # Eigenvalues of P3
    B = 1
    E = 1
    q = 0
    for i in range(len(ps)-1,-1,-1):
        if ps[i] == '0':        # I
            B = kron(P0,B)
            E = kron(E0,E)
        elif ps[i] == '1':      # X
            B = kron(P1,B)
            E = kron(E1,E)
            qcirc.ry(-pi/2,q)
        elif ps[i] == '2':      # Y
            B = kron(P2,B)
            E = kron(E2,E)
            qcirc.rx(pi/2,q)
        elif ps[i] == '3':      # Z
            B = kron(P3,B)
            E = kron(E3,E)
        q += 1
    qcirc.barrier()
    qcirc.measure([0,1], [0,1])                 # TBD: make this scalable
    return(B,E,qcirc)

'''
Convert a decimal number to base-n
'''
def toStr(n,base):
   convertString = "0123456789ABCDEF"
   if n < base:
      return convertString[n]
   else:
      return toStr(n//base,base) + convertString[n%base]

def dict2hist(counts):
    phist = np.zeros(hsdim)
    for i in range(0,hsdim):
        if counts.get(str(bin(i)[2:]).zfill(qubits)) != None:
            phist[i] = counts.get(str(bin(i)[2:]).zfill(qubits)) / trials
    return phist

'''
Quantum State Tomography
'''
def QST():
    # fname = open("AAQPT_full.txt", "w")
    dm = np.zeros((hsdim, hsdim)) * 0j
    for i in range(0,4**qubits):
        ps = toStr(i,4).zfill(qubits)
        B,E,qcircB = PostRot(ps,deepcopy(qcirc))
        job = execute(qcircB, Aer.get_backend('qasm_simulator'), shots=trials, memory=True)
        result = job.result()
        # fname.write(str(result.get_memory())+"\n")
        Si = sum(np.multiply(E, dict2hist(result.get_counts(qcircB))))
        # print(np.trace(B.conj().T * rho.data))
        dm += Si * B
    est_rho = dm/hsdim
    # fname.close()
    return est_rho

def partialTrace1(dm):
    dm1 = np.zeros((2,2)) * 0j
    for i in range(0,2):
        for j in range(0,2):
            dm1[i][j] = dm[i][j]+dm[i+2][j+2]
    return dm1

qcirc = QuantumCircuit(qubits, qubits)
EntangleAncilla(qcirc)
QProcess(qcirc, [0])
print(qcirc)

rho = qi.DensityMatrix.from_instruction(qcirc)                                                                                                  # TEST QST
print("\nActual Density Matrix")                                                                                                                # TEST QST
print(rho.data)                                                                                                                                 # TEST QST

print("\nEstimated Density Matrix")
rho_choi = np.round(QST(),aprx)
print(rho_choi)

psi_inp = QuantumCircuit(process_qubits, process_qubits)
psi_inp.i(0)
rho_inp = qi.DensityMatrix.from_instruction(psi_inp).data
print("\nInput Density Matrix")
print(rho_inp)

# Use the quantum process to evolve the density matrix
QProcess(psi_inp, [0])                                                                                                                          # TEST QPT
rho_out = qi.DensityMatrix.from_instruction(psi_inp).data                                                                                       # TEST QPT 
print("\nActual Output Density Matrix")                                                                                                         # TEST QPT 
print(rho_out)                                                                                                                                  # TEST QPT

rho_out_choi = 2**process_qubits * partialTrace1( np.matmul( np.kron(np.transpose(rho_inp), np.eye(2**process_qubits) ), rho.data ))            # TEST QPT      
print("\nOutput Density Matrix using Actual Choi Matrix")                                                                                       # TEST QPT
print(rho_out_choi)                                                                                                                             # TEST QPT      

# Use the estimated Choi matrix to predict the output density matrix 
rho_out_choi_est = 2**process_qubits * partialTrace1( np.matmul( np.kron(np.transpose(rho_inp), np.eye(2**process_qubits) ), rho_choi ))
print("\nOutput Density Matrix using Estimated Choi Matrix")
print(rho_out_choi_est)

'''

(qeait) D:\GoogleDrive\RESEARCH\A1 - Programs\QKSA\v12>python qpt.py

      ░      ┌───┐ ░  ░ ┌───┐ ░
q_0: ─░──────┤ X ├─░──░─┤ H ├─░─
      ░ ┌───┐└─┬─┘ ░  ░ └───┘ ░
q_1: ─░─┤ H ├──■───░──░───────░─
      ░ └───┘      ░  ░       ░
c: 2/═══════════════════════════


Actual Density Matrix
[[ 0.25+0.j  0.25+0.j  0.25+0.j -0.25+0.j]
 [ 0.25+0.j  0.25+0.j  0.25+0.j -0.25+0.j]
 [ 0.25+0.j  0.25+0.j  0.25+0.j -0.25+0.j]
 [-0.25+0.j -0.25+0.j -0.25+0.j  0.25+0.j]]

Estimated Density Matrix
[[ 0.25+0.j    0.25-0.01j  0.25+0.01j -0.25+0.02j]
 [ 0.25+0.01j  0.24+0.j    0.25-0.j   -0.25+0.01j]
 [ 0.25-0.01j  0.25+0.j    0.26+0.j   -0.25+0.01j]
 [-0.25-0.02j -0.25-0.01j -0.25-0.01j  0.24+0.j  ]]

Input Density Matrix
[[1.+0.j 0.+0.j]
 [0.+0.j 0.+0.j]]

Actual Output Density Matrix
[[0.5+0.j 0.5+0.j]
 [0.5+0.j 0.5+0.j]]

Output Density Matrix using Actual Choi Matrix
[[0.5+0.j 0.5+0.j]
 [0.5+0.j 0.5+0.j]]

Output Density Matrix using Estimated Choi Matrix
[[0.5 +0.j   0.5 -0.02j]
 [0.5 +0.02j 0.48+0.j  ]]

 '''