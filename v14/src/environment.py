import random
import math
import copy
import sys

from qiskit import QuantumCircuit, Aer, execute, IBMQ
from qiskit.tools.monitor import job_monitor

class environment:

	num_qb = 0					# Number of qubits the environment is defined on
	qpCirc = None				# Qiskit Circuit that determines environmental dynamics
	backend = None
	qcsim = 'qasm_simulator'	# Default environment simulator
	basis = []
	A = []						# Action space
	E = []						# Perception space
	
	def __init__(self, num_qb):
		self.num_qb = num_qb
		return
	
	def createEnv(self, qcirc=None):
		print("\n. . . environment being setup . . .\n")	
		if qcirc is None:
			qcirc = QuantumCircuit(self.num_qb)
			stateSel = int(input("\t1: All-Zeros\n\t2: Equal Superposition\n\t3: GHZ-state\n\t4: Random Pauli\n\t5: Random U\n\t6: Custom Test Env.\n\n===> Choose environment state [Default: 2]: ") or "2")
			if stateSel == 1:
				print("All-Zero state on "+str(self.num_qb)+" qubits selected")
			elif stateSel == 2:
				print("Equal Superposition state on "+str(self.num_qb)+" qubits selected")
				for i in range(0,self.num_qb):
					qcirc.h(i)
			elif stateSel == 3:
				print("GHZ-state on "+str(self.num_qb)+" qubits selected")
				qcirc.h(0)
				for i in range(1,self.num_qb):
					qcirc.cx(0,i)
			elif stateSel == 4:
				print("Random Pauli Rotation state on "+str(self.num_qb)+" qubits selected")
				for i in range(0,self.num_qb):
					axis = random.randint(0,2)
					if axis == 0:
						qcirc.rz(math.pi/2,i)
					elif axis == 1:
						qcirc.rx(math.pi/2,i)
					elif axis == 2:
						qcirc.ry(math.pi/2,i)	
			elif stateSel == 5:
				print("Random U-gate state on "+str(self.num_qb)+" qubits selected")
				for i in range(0,self.num_qb):
					a_theta = random.uniform(0, math.pi)
					a_phi = random.uniform(0, math.pi)
					a_lambda = random.uniform(0, math.pi)
					qcirc.u(a_theta, a_phi, a_lambda, i)	
			elif stateSel == 6:
				print("Custom Test environment on "+str(self.num_qb)+" qubits selected")
				# add custom code below
				qcirc.h(0)	
				qcirc.t(0)
				qcirc.h(0)	
				qcirc.t(0)
				# add custom code above
			else:
				print("Invalid selection! Default All-Zero state on "+str(self.num_qb)+" qubits selected")
		else:
			backendSel = int(input("1: Qiskit QASM simulator\n2: IBMQ Belem 5q\n\nChoose Environment Backend [Default: 1]: ") or "1")
			if backendSel == 1:
				print("Qiskit QASM simulator backend selected")
				self.backend = Aer.get_backend(self.qcsim)
			elif backendSel == 2:
				print("IBMQ Belem 5q backend selected")
				fname = open(sys.path[0] + '\..\..\ibmq.txt')	# The IBMQ API Key should be in the text file in the same directory as QKSA. Tested on Anaconda + Windows.
				api = fname.readline()
				fname.close()
				IBMQ.enable_account(api)
				provider = IBMQ.get_provider('ibm-q')
				self.backend = provider.get_backend('ibmq_belem')
			else:
				print("Invalid selection! Default Qiskit QASM simulator selected")
		self.define_A()
		self.define_E()
		print()
		print(qcirc.draw())
		self.qpCirc = qcirc
		print("\n. . . environment setup complete . . .")

	def DecToBaseN(self, n, b, l):
		s = ''
		while n != 0:
			s = str(n%b)+s
			n = n//b
		return ('0'*(l-len(s)))+s

	def define_A(self):
		# define action space A as all 3-axis basis of self.num_qb qubits
		for i in range(3**self.num_qb):
			self.A.append(str(self.DecToBaseN(i,3,self.num_qb)))
		return

	def define_E(self):
		# Define percept space E as all binary strings of self.num_qb qubits
		for i in range(2**self.num_qb):
			self.E.append(str(self.DecToBaseN(i,2,self.num_qb)))
		return

	def saveEnv(self):
		fname = open("env.qasm", "w")
		fname.write(self.qpCirc.qasm())
		fname.close()
		return "env.qasm"

	def suspendEnv(self):
		if str(self.backend) != self.qcsim:
			IBMQ.disable_account()

	def action(self, a_t):
		self.basis = list(reversed(a_t[1]))
		return

	def perception(self, neighbours):
		
		circ = copy.deepcopy(self.qpCirc)				# Post rotation and measure should not change original circuit

		if len(neighbours) != len(self.basis):
			print("Error: Not all measurement basis defined by agent. Default All-Z basis is selected.")
		else:
			for i in range(0,len(self.basis)):
				if self.basis[i] == '1':
					circ.ry(-math.pi/2,neighbours[i]) 	# for measuring along X
				elif self.basis[i] == '2':
					circ.rx(math.pi/2,neighbours[i]) 	# for measuring along Y

		circ.barrier()	
		for n in neighbours:
			circ.measure(n,n)
		circ.barrier()	
		# print(circ.draw())
		
		job = execute(circ, self.backend, shots=1, memory=True)
		if str(self.backend) != self.qcsim:
			job_monitor(job, quiet=True)
		result = job.result()
		memory = result.get_memory()
		return memory 