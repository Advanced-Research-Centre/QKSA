import numpy as np
import qiskit.quantum_info as qi
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

qcirc = QuantumCircuit(2, 2)
qcirc.h(1)
qcirc.cx(1,0)
qcirc.h(0)
C = qi.DensityMatrix.from_instruction(qcirc).data
print(C) 

from src.metrics import metrics
M = metrics()
# print("T: ",M.DeltaT(A,B))
print("T: ",M.DeltaT(A,C))
print("B: ",M.DeltaB(A,C))
print("HS: ",M.DeltaHS(A,C))
# print("H: ",M.DeltaH(A,B))