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
Make entangled state
'''
def EntangleAncilla(qcirc):
    qcirc.barrier()
    qcirc.h(1)
    qcirc.cx(1,0)
    qcirc.barrier()
    return

'''
The "black-box" quantum process is defined here
'''
def QProcess(qcirc):
    qcirc.barrier()
    qcirc.h(0)
    qcirc.barrier()
    return

qcirc = QuantumCircuit(qubits, qubits)
EntangleAncilla(qcirc)
QProcess(qcirc)
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

fname = open("AAQPT_full.txt", "w")

dm = np.zeros((hsdim, hsdim)) * 0j
for i in range(0,4**qubits):
    ps = toStr(i,4).zfill(qubits)
    B,E,qcircB = PostRot(ps,deepcopy(qcirc))
    job = execute(qcircB, Aer.get_backend('qasm_simulator'), shots=trials, memory=True)
    result = job.result()
    fname.write(str(result.get_memory())+"\n")
    Si = sum(np.multiply(E, dict2hist(result.get_counts(qcircB))))
    # print(np.trace(B.conj().T * rho.data))
    dm += Si * B
est_rho = dm/hsdim

fname.close()

print("\nEstimated Density Matrix")
print(np.round(est_rho,aprx))

"""

(qeait) D:\GoogleDrive\RESEARCH\A1 - Programs\AutonomousQuantumPhysicist\v12>python qpt.py
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
[[ 0.26+0.j    0.25+0.02j  0.25+0.02j -0.25-0.j  ]
 [ 0.25-0.02j  0.25+0.j    0.25+0.01j -0.25+0.j  ]
 [ 0.25-0.02j  0.25-0.01j  0.24+0.j   -0.25+0.01j]
 [-0.25+0.j   -0.25-0.j   -0.25-0.01j  0.25+0.j  ]]

"""