from qiskit import QuantumCircuit, qasm, Aer, execute
from random import randint
from math import pi

class environment:

	qprocess = ""		# Filename of OpenQL compiled cQASM that determines environmental dynamics
	num_qb = 0			# Number of qubits the environment is defined on
	allZ = True
	basis = []
	simulator = Aer.get_backend('qasm_simulator')

	def __init__(self):

		num_qb = int(input("Number of qubits : ") or "2")
		print("\n. . . environment being setup . . .\n")
		qcirc = QuantumCircuit(num_qb, num_qb)
		stateSel = int(input("1: All-Zeros\n2: Equal Superposition\n3: Random Pauli\n4: GHZ-state\n5: W-state\n6: from OpenQASM file\n\nChoose Environment State : ") or "2")
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
		elif stateSel == 6:
			print("From OpenQASM file on "+str(num_qb)+" qubits selected")
			print("current bug = future feature!")
		else:
			print("Invalid selection! Default All-Zero state on "+str(num_qb)+" qubits selected")
		print(qcirc.draw())
		f = open("env.qasm", "w")
		f.write(qcirc.qasm())
		f.close()
		print("\n. . . environment setup complete . . .\n")
		self.qprocess = "env.qasm"
		self.num_qb = num_qb
		return

	def measure(self, neighbours, basis):
		
		circ = QuantumCircuit.from_qasm_file(self.qprocess)

		if len(neighbours) != len(basis):
			print("Error: Not all measurement basis defined by agent. Default All-Z basis is selected.")
		else:
			for i in range(0,len(basis)):
				if basis[i] == '1':
					circ.ry(-pi/2,neighbours[i]) 	# for measuring along X
				elif basis[i] == '2':
					circ.rx(pi/2,neighbours[i]) 	# for measuring along Y
			
		for n in neighbours:
			circ.measure(n,n)
		# print(circ.draw())
		
		result = execute(circ, self.simulator, shots=1, memory=True).result()
		memory = result.get_memory()
		return memory 