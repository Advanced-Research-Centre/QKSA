from qiskit import QuantumCircuit, Aer, execute
from random import randint, uniform
from math import pi
from copy import deepcopy

class environment:

	qprocess = ""		# Filename of OpenQL compiled cQASM that determines environmental dynamics
	qpCirc = None
	num_qb = 0			# Number of qubits the environment is defined on
	allZ = True
	basis = []
	simulator = Aer.get_backend('qasm_simulator')

	def __init__(self, num_qb):
		self.num_qb = num_qb
		return

	def createEnv(self, qcirc=None):
		
		print("\n. . . environment being setup . . .\n")	
		if qcirc is None:
			qcirc = QuantumCircuit(self.num_qb)
			stateSel = int(input("1: All-Zeros\n2: Equal Superposition\n3: Random Pauli\n4: Random U\n5: Custom Test Env.\n6: GHZ-state\n7: W-state\n\nChoose Environment State : ") or "2")
			if stateSel == 1:
				print("All-Zero state on "+str(self.num_qb)+" qubits selected")
			elif stateSel == 2:
				print("Equal Superposition state on "+str(self.num_qb)+" qubits selected")
				for i in range(0,self.num_qb):
					qcirc.h(i)
			elif stateSel == 3:
				print("Random Pauli Rotation state on "+str(self.num_qb)+" qubits selected")
				for i in range(0,self.num_qb):
					axis = randint(0,2)
					if axis == 0:
						qcirc.rz(pi/2,i)
					elif axis == 1:
						qcirc.rx(pi/2,i)
					elif axis == 2:
						qcirc.ry(pi/2,i)	
			elif stateSel == 4:
				print("Random U-gate state on "+str(self.num_qb)+" qubits selected")
				for i in range(0,self.num_qb):
					a_theta = uniform(0, pi)
					a_phi = uniform(0, pi)
					a_lambda = uniform(0, pi)
					qcirc.u(a_theta, a_phi, a_lambda, i)	
			elif stateSel == 5:
				print("Custom Test environment on "+str(self.num_qb)+" qubits selected")
				# add custom code below
				qcirc.h(0)	
				qcirc.t(0)
				qcirc.h(0)	
				qcirc.t(0)
				# add custom code above
			elif stateSel == 6:
				print("GHZ-state on "+str(self.num_qb)+" qubits selected")
				qcirc.h(0)
				for i in range(1,self.num_qb):
					qcirc.cx(0,i)
			elif stateSel == 7:
				print("W-state on "+str(self.num_qb)+" qubits selected")
				# https://quantumcomputing.stackexchange.com/questions/4350/general-construction-of-w-n-state
				print("current bug = future feature!")
			else:
				print("Invalid selection! Default All-Zero state on "+str(self.num_qb)+" qubits selected")
		print(qcirc.draw())
		self.qpCirc = qcirc
		print("\n. . . environment setup complete . . .\n")
	
	def saveEnv(self):
		fname = open("env.qasm", "w")
		fname.write(self.qpCirc.qasm())
		fname.close()
		return "env.qasm"

	def action(self, a_t):
		self.basis = list(reversed(a_t[1]))
		return

	def measure(self, neighbours):
		
		circ = deepcopy(self.qpCirc)				# Post rotation and measure should not change original circuit

		if len(neighbours) != len(self.basis):
			print("Error: Not all measurement basis defined by agent. Default All-Z basis is selected.")
		else:
			for i in range(0,len(self.basis)):
				if self.basis[i] == '1':
					circ.ry(-pi/2,neighbours[i]) 	# for measuring along X
				elif self.basis[i] == '2':
					circ.rx(pi/2,neighbours[i]) 	# for measuring along Y

		circ.barrier()	
		for n in neighbours:
			circ.measure(n,n)
		circ.barrier()	
		# print(circ.draw())
		
		result = execute(circ, self.simulator, shots=1, memory=True).result()
		memory = result.get_memory()
		return memory 