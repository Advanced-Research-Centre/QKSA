#Import library
from qiskit import*
from qiskit import Aer

import numpy as np
import matplotlib.pyplot as plt
from random import randint

# identity function
def f0(qc,ip,op):
    qc.i(op)

# copy function
def f1(qc,ip,op):
    qc.cx(ip,op)

# copy and invert function
def f2(qc,ip,op):
    qc.cx(ip,op)
    qc.x(op)

# invert function
def f3(qc,ip,op):
    qc.x(op)

def oracle_1_ip(qc):
    t = randint(0,3)
    print("Secret function:",t)
    qc.barrier()
    if (t == 0):
        f0(qc,0,1)
    elif (t == 1):
        f1(qc,0,1)
    elif (t == 2):
        f2(qc,0,1)
    else:
        f3(qc,0,1)
    qc.barrier()

q_input = QuantumRegister(1, name = "a")
q_output = QuantumRegister(1, name = "c")
classical = ClassicalRegister(1, name = "classical")
qc = QuantumCircuit(q_input, q_output, classical)
qc.h(q_input)
print(qc.draw())

oracle_1_ip(qc)
qc.measure(q_output,classical)
print(qc.draw())

backend = Aer.get_backend('qasm_simulator')
sim = execute(qc, backend)
result = sim.result()
counts = result.get_counts()

if '0' in counts.keys():
    if counts['0'] == 1024:
        print("f0")
    else:
        print("f1 or f2")
else:
    print("f3")