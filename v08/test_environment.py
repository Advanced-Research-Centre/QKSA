from environment import environment
from qiskit import QuantumCircuit

def createEnvOpenQASM(num_qb, neighbours):
	# Bell pair environment
	qcirc = QuantumCircuit(num_qb,neighbours)
	qcirc.h(0)
	qcirc.cx(0, 1)
	f = open("env.qasm", "w")
	f.write(qcirc.qasm())
	f.close()
	return "env.qasm"

def perceive(dynamics, steps):
	env = environment(dynamics)
	for _ in range(0,steps):
		print("Perception:",env.measure([0,1]))

dynamics = createEnvOpenQASM(2,2)
perceive(dynamics, 10)