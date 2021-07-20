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

'''
Fn#     Name of the Boolean Function	        Boolean Function        Meaning
----------------------------------------------------------------------------------------------------------
f00     Null	                                0                       Always 0
f01     Identity	                            1                       Always 1
f02     Transfer	                            A                       Pass value of A
f03                                             B                       Pass value of B
f04     NOT	                                    !A                      Pass negated value of A
f05                                             !B                      Pass negated value of B
f06     AND	                                    A∙B                     1 only if A and B both are 1
f07     NAND	                                !(A∙B)                  0 only if A and B both are 1
f08     OR	                                    A+B                     0 only if A and B both are 0
f09     NOR	                                    !(A+B)                  1 only if A and B both are 0
f10     Implication	                            A+(!B)                  If B, then A
f11                                             (!A)+B                  If A, then B
f12     Inhibition	                            A∙(!B)                  A but not B
f13                                             !(A)∙B                  B but not A
f14     EX-OR	                                A⊕B                     or B, but not both
f15     EX-NOR	                                !(A⊕B)                  if A equals B
'''