from environment import environment
from qiskit import QuantumCircuit
from random import randint
from math import pi, floor
import numpy as np
from pybdm import BDM

def createEnvOpenQASM():

	global num_qb, neighbours
	print("\n. . . environment being setup . . .\n")
	qcirc = QuantumCircuit(num_qb,neighbours)
	stateSel = int(input("1: All-Zeros\n2: Equal Superposition\n3: Random Pauli\n4: GHZ-state\n5: W-state\n\nChoose Environment State : "))
	if stateSel == 1:
		print("All-Zero state on "+str(num_qb)+" qubits selected")
	elif stateSel == 2:
		print("Equal Superposition state on "+str(num_qb)+" qubits selected")
		for i in range(0,num_qb):
			qcirc.h(i)
	elif stateSel == 3:
		print("Random Pauli Rotation state on "+str(num_qb)+" qubits selected")
		for i in range(0,num_qb):
			axis = randint(0,2)
			if axis == 0:
				qcirc.rz(pi/2,i)
			elif axis == 1:
				qcirc.rx(pi/2,i)
			elif axis == 2:
				qcirc.ry(pi/2,i)		
	elif stateSel == 4:
		print("GHZ-state on "+str(num_qb)+" qubits selected")
		qcirc.h(0)
		for i in range(1,num_qb):
			qcirc.cx(0,i)
	elif stateSel == 5:
		print("W-state on "+str(num_qb)+" qubits selected")
		# https://quantumcomputing.stackexchange.com/questions/4350/general-construction-of-w-n-state
		print("current bug = future feature!")
	else:
		print("Invalid selection! Default All-Zero state on "+str(num_qb)+" qubits selected")
	print(qcirc.draw())
	f = open("env.qasm", "w")
	f.write(qcirc.qasm())
	f.close()
	print("\n. . . environment setup complete . . .\n")
	return "env.qasm"

def policy(step, percept):

	global neighbours, trials
	reqdTrials = pow(3,neighbours)
	basis = [0] * neighbours
	# there is always a first time
	if percept == []: 
		global history
		history = np.empty((reqdTrials,trials),dtype='object')
	# sequentially try out all Pauli basis combinations of neighbouring qubits for 10 cycles
	elif step < reqdTrials*trials:
		pb = np.base_repr(floor(step/trials), base=3, padding=neighbours)
		pb = pb[len(pb)-neighbours:]
		for i in range(0,len(pb)):
			basis[i] = int(pb[i])
	# select winning policy as the one with the lowest K-complexity of the percept history 
	else:
		minKC = 10000
		for i in range(0,reqdTrials):
			PBhist = ''.join(history[i])
			data = np.array(list(PBhist)).astype(np.int)
			aprxKC = BDM(ndim=1)
			nu_l = aprxKC.bdm(data)
			print(PBhist,nu_l)
			if nu_l < minKC:
				minKC = nu_l
				basis_best = i
		pb = np.base_repr(basis_best, base=3, padding=neighbours)
		pb = pb[len(pb)-neighbours:]
		for i in range(0,len(pb)):
			basis[i] = int(pb[i])
		print("Best basis : ",basis)
	return basis

def perceive(dynamics):

	global lifespan, neighbours, history, trials
	env = environment(dynamics)
	percept = []
	for step in range(0,lifespan):
		basis = policy(step, percept)
		env.setBasis(basis)
		percept = env.measure(neighbours)
		if step < pow(3,neighbours)*trials: 
			history[floor(step/trials)][step%trials] = percept[0]
		# print("Basis : ",basis," Percept : ",percept)
	print(history)

num_qb = int(input("Number of qubits : "))
neighbours = int(input("Number of neighbours of Agent 1 : ")) # assumed to be qubit ID 0 to (neighbours-1)
lifespan = int(input("Lifespan of Agent 1 : "))

dynamics = createEnvOpenQASM()

trials = 10
perceive(dynamics)