"""
Entanglement Assisted Quantum Process Tomography on n-qubits
Construct Choi density matrix of process from tomographic trials
"""

import numpy as np
from numpy.lib.function_base import append	
from numpy.lib.shape_base import kron
import random
from qiskit import QuantumCircuit
import time

class qpt:

    name = 'EAQPT'
    e_est = 0
    a_est = 5
    s_est = 16384
    t_est = 0
    
    def __init__(self, num_qb_qp):

        self.num_qb     = num_qb_qp*2                # Number of qubits required for process tomography
        self.hsdim      = 2**self.num_qb

    def EntangleAncilla(self, qcirc):
        qp_qb = int(self.num_qb / 2)
        qcirc.barrier()
        for i in range(0,qp_qb):
            qcirc.h(qp_qb+i)
            qcirc.cx(qp_qb+i,i)
        qcirc.barrier()
        return

    def setup(self, qpCirc):
        qptCirc = QuantumCircuit(self.num_qb, self.num_qb)
        self.EntangleAncilla(qptCirc)
        qp_gate = qpCirc.to_gate(label='QP')
        qptCirc.append(qp_gate,list(range(0,int(self.num_qb/2))))
        qptCirc.barrier()
        return qptCirc

    def ae_dict(self):

        self.ae_db = {}
        t= 0
        for i in self.hist_a:
            if i[0] == 'E':             # Filter only QST on maximally entangled states for AAQPT Choi matrix process tomography
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
        tic = time.perf_counter()
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
            B,E = self.ev(ps)
            Si = sum(np.multiply(E, phist))
            est_rho += Si * B
        est_rho = np.round(est_rho/self.hsdim,self.a_est)
        toc = time.perf_counter()
        self.t_est = toc - tic
        return est_rho