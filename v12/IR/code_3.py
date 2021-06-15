"""
Quantum State Tomography on n-qubits
Construct density matrix from tomographic trials
https://github.com/prince-ph0en1x/QuEst/tree/neil
Reference: http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf
"""

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer, execute

import numpy as np
from math import pi
from numpy.lib.shape_base import kron
from copy import deepcopy

qubits = 2
hsdim = 2**qubits
trials = 1000
aprx = 1            # Places of decimal to round the reconstructed density matrix

'''
The "black-box" quantum process is defined here
'''
def QStatePrep(qcirc):
    qcirc.barrier()
    qcirc.h(0)
    qcirc.barrier()
    return

qcirc = QuantumCircuit(qubits, qubits)
QStatePrep(qcirc)
print(qcirc)

rho = qi.DensityMatrix.from_instruction(qcirc)
print("\nActual Density Matrix")
print(rho.data)

def PostRot(ps,qcirc):
    P0 = [[1,0],[0,1]]
    P1 = [[0,1],[1,0]]
    P2 = [[0,complex(0,-1)],[complex(0,1),0]]
    P3 = [[1,0],[0,-1]]
    E0 = [1, 1]
    E1 = [1, -1]
    E2 = [1, -1]
    E3 = [1, -1]
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
    qcirc.measure([0,1], [0,1]) # make this scalable
    return(B,E,qcirc)

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

dm = np.zeros((hsdim, hsdim)) * 0j
for i in range(0,4**qubits):
    ps = toStr(i,4).zfill(qubits)
    B,E,qcircB = PostRot(ps,deepcopy(qcirc))
    job = execute(qcircB, Aer.get_backend('qasm_simulator'), shots=trials)
    result = job.result()
    Si = sum(np.multiply(E, dict2hist(result.get_counts(qcircB))))
    # print(np.trace(B.conj().T * rho.data))
    dm += Si * B
est_rho = dm/hsdim

print("\nEstimated Density Matrix")
print(np.round(est_rho,aprx))

"""

(qeait) D:\GoogleDrive\RESEARCH\A1 - Programs\AutonomousQuantumPhysicist\v12\IR>python code_3.py
      ░ ┌───┐ ░
q_0: ─░─┤ H ├─░─
      ░ └───┘ ░
q_1: ─░───────░─
      ░       ░
c: 2/═══════════


Actual Density Matrix
[[0.5+0.j 0.5+0.j 0. +0.j 0. +0.j]
 [0.5+0.j 0.5+0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]]

Estimated Density Matrix
[[ 0.5+0.j  0.5-0.j -0. -0.j  0. +0.j]
 [ 0.5+0.j  0.5+0.j -0. -0.j  0. +0.j]
 [-0. +0.j -0. +0.j  0. +0.j  0. +0.j]
 [ 0. -0.j  0. -0.j  0. -0.j -0. +0.j]]

"""