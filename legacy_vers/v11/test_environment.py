from environment import environment
from agent import agent

from qiskit import QuantumCircuit
from random import randint
from math import pi, floor
import numpy as np
from pybdm import BDM

def createEnvOpenQASM():

	num_qb = int(input("Number of qubits : ") or "2")
	print("\n. . . environment being setup . . .\n")
	qcirc = QuantumCircuit(num_qb, num_qb)
	stateSel = int(input("1: All-Zeros\n2: Equal Superposition\n3: Random Pauli\n4: GHZ-state\n5: W-state\n\nChoose Environment State : ") or "3")
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

def agent_evolve(genes):
	print("Mutants, assemble!")

def agent_learn(env,neighbours,lifespan):
	
	genes = [1, 2, neighbours, lifespan]
	agt = agent(genes, env)
	while True:
		[die, rpd] = agt.runStep()
		if die:
			print("Bye bye, cruel world!")
			break
		if rpd:
			# mutate genes here
			agent_evolve(genes)
			break

dynamics = createEnvOpenQASM()
env = environment(dynamics)

numNeighbours = int(input("Number of neighbours of Agent 1 : ") or "2") 
neighbours = list(range(0,numNeighbours)) # assumed to be qubit ID 0 to (neighbours-1)
lifespan = int(input("Lifespan of Agent 1 : ") or "3")

agent_learn(env, neighbours, lifespan)