from qiskit import QuantumCircuit, qasm, Aer, execute
from math import pi

class environment:

	OpenQASM = ""	# Filename of OpenQL compiled cQASM that determines environmental dynamics
	allZ = True
	basis = []
	simulator = ""

	def __init__(self, dynamics):

		self.OpenQASM = dynamics
		self.simulator = Aer.get_backend('qasm_simulator')

	def setBasis(self, basis):

		self.basis = basis
		self.allZ = False

	def measure(self, neighbours):
		
		circ = QuantumCircuit.from_qasm_file(self.OpenQASM)
		if (not self.allZ):
			if len(self.basis) == len(neighbours):
				for n in neighbours:
					if self.basis[n] == 1:
						# circ.ry(pi/2,n) # for measuring along -X
						circ.ry(-pi/2,n) # for measuring along X
					elif self.basis[n] == 2:
						circ.rx(pi/2,n) # for measuring along Y	
			else:
				print("Error: Not all measurement basis defined by agent. Default All-Z basis is selected.")
		for n in neighbours:
			circ.measure(n,n)
		# print(circ.draw())
		result = execute(circ, self.simulator, shots=1, memory=True).result()
		memory = result.get_memory(circ)
		return memory 