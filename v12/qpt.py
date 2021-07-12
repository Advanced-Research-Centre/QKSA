"""
Ancilla Assisted Quantum Process Tomography on n-qubits
Construct Choi density matrix of process from tomographic trials
"""

import numpy as np
from numpy.lib.function_base import append	
from numpy.lib.shape_base import kron
import random

class qpt:

    def __init__(self, num_qb):

        self.num_qb     = num_qb        # Number of qubits
        self.hsdim      = 2**num_qb

    def ae_dict(self):

        self.ae_db = {}
        t= 0
        for i in self.hist_a:
            if i[0] == "E":             # Filter only QST on maximally entangled states for AAQPT Choi matrix process tomography
                # print(i,self.hist_e[t])
                if i[1] in self.ae_db:
                    self.ae_db[i[1]].append(self.hist_e[t])
                else:
                    self.ae_db[i[1]] = [self.hist_e[t]]
            t += 1
        return

    def ev(self, ps):
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
        for i in range(len(ps)-1,-1,-1):
            if ps[i] == '0':        # I
                B = kron(P0,B)
                E = kron(E0,E)
            elif ps[i] == '1':      # X
                B = kron(P1,B)
                E = kron(E1,E)
            elif ps[i] == '2':      # Y
                B = kron(P2,B)
                E = kron(E2,E)
            elif ps[i] == '3':      # Z
                B = kron(P3,B)
                E = kron(E3,E)
        return(B,E)


    def dict2hist(self, ps_meas):
        phist = np.zeros(self.hsdim)
        trials = len(ps_meas)
        for i in ps_meas:
            phist[int(i,2)] += 1
        phist /= trials
        return phist

    '''
    Convert a decimal number to base-n
    '''
    def toStr(self,n,base):
        convertString = "0123456789ABCDEF"
        if n < base:
            return convertString[n]
        else:
            return self.toStr(n//base,base) + convertString[n%base]

    def est_choi(self, hist_a, hist_e):

        self.hist_a		= hist_a		# History of actions
        self.hist_e		= hist_e		# History of perceptions
        self.ae_dict()    
        
        est_rho = np.zeros((self.hsdim, self.hsdim)) * 0j
        for i in range(0,4**self.num_qb):
            ps = self.toStr(i,4).zfill(self.num_qb)
            if ps in self.ae_db:
                phist = self.dict2hist(self.ae_db[ps])
            else:
                # assume uniformly random measurement result (est_rho is completely mixed state)
                phist = np.ones(self.hsdim) * (1/self.hsdim)
            # print(ps,phist)
            B,E = self.ev(ps)
            Si = sum(np.multiply(E, phist))
            est_rho += Si * B
        est_rho = est_rho/self.hsdim
        return est_rho

    def policy(self):
        mb = random.randint(0, 4**self.num_qb - 1)      # Entropy driven policies
        mbps = self.toStr(mb,4).zfill(self.num_qb)
        return ["E", mbps]
        
        


# import qiskit.quantum_info as qi
# from qiskit import QuantumCircuit, Aer, execute

# from math import pi
# from copy import deepcopy

# process_qubits = 1
# qubits = 2*process_qubits   # for AAQPT
# hsdim = 2**qubits           # Hilbert space dimension
# trials = 1024               # Number of tomographic trials for each observable
# aprx = 2                    # Places of decimal to round the reconstructed density matrix


# '''
# The "black-box" quantum process is defined here
# '''
# def QProcess(qcirc, qreg):
#     qcirc.barrier()
#     qcirc.h(qreg[0])
#     qcirc.barrier()
#     return

# '''
# Make entangled state
# '''
# def EntangleAncilla(qcirc):
#     qcirc.barrier()
#     qcirc.h(1)
#     qcirc.cx(1,0)
#     qcirc.barrier()
#     return

# qcirc = QuantumCircuit(qubits, qubits)
# EntangleAncilla(qcirc)
# QProcess(qcirc, [0])
# print(qcirc)

# rho = qi.DensityMatrix.from_instruction(qcirc)                                                                                                  # TEST QST
# print("\nActual Density Matrix")                                                                                                                # TEST QST
# print(rho.data)                                                                                                                                 # TEST QST

# print("\nEstimated Density Matrix")
# rho_choi = np.round(QST(),aprx)
# print(rho_choi)

# psi_inp = QuantumCircuit(process_qubits, process_qubits)
# psi_inp.i(0)
# rho_inp = qi.DensityMatrix.from_instruction(psi_inp).data
# print("\nInput Density Matrix")
# print(rho_inp)

# # Use the quantum process to evolve the density matrix
# QProcess(psi_inp, [0])                                                                                                                          # TEST QPT
# rho_out = qi.DensityMatrix.from_instruction(psi_inp).data                                                                                       # TEST QPT 
# print("\nActual Output Density Matrix")                                                                                                         # TEST QPT 
# print(rho_out)                                                                                                                                  # TEST QPT

# rho_out_choi = 2**process_qubits * partialTrace1( np.matmul( np.kron(np.transpose(rho_inp), np.eye(2**process_qubits) ), rho.data ))            # TEST QPT      
# print("\nOutput Density Matrix using Actual Choi Matrix")                                                                                       # TEST QPT
# print(rho_out_choi)                                                                                                                             # TEST QPT      

# # Use the estimated Choi matrix to predict the output density matrix 
# rho_out_choi_est = 2**process_qubits * partialTrace1( np.matmul( np.kron(np.transpose(rho_inp), np.eye(2**process_qubits) ), rho_choi ))
# print("\nOutput Density Matrix using Estimated Choi Matrix")
# print(rho_out_choi_est)
