'''
 Code for QPT
'''

import numpy as np
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

E = QProcess()
for i in E:
    print("Probability: ",i[1])
    print(i[0])
    print(qi.Choi(i[0]))
    print(qi.Kraus(i[0]))


