'''
Quantum Process Tomography
'''

# Reference: https://qiskit.org/documentation/tutorials/noise/8_tomography.html#1-qubit-process-tomography-example

import qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.ignis.verification.tomography import process_tomography_circuits, ProcessTomographyFitter

q = QuantumRegister(1)
circ = QuantumCircuit(q)
circ.h(q[0])

# Get the ideal unitary operator
a = qi.Choi(circ)
print(a)
print(a.is_unitary())
print(a.to_operator())
target_unitary = qi.Operator(circ)
print(target_unitary)

# Generate process tomography circuits and run on qasm simulator
# Preparation in Pauli basis (alternate SIC basis is not scalable):
#       "Z_p"   : |0⟩   : I
#       "Z_m"   : |1⟩   : X
#       "X_p"   : |+⟩   : H
#       "Y_m"   : |i⟩   : H then S         (doubt: isn't that Y_p ?)
# Measurement in:
#       X               : H
#       Y               : Sdag then H
#       Z               : I
# Total trails for n-qubits: 4^n * 3^n 
# process_tomography_circuits(circuit, measured_qubits, prepared_qubits=None, meas_labels='Pauli', meas_basis='Pauli', prep_labels='Pauli', prep_basis='Pauli')
qpt_circs = process_tomography_circuits(circ, q)
# trials = len(qpt_circs)
# for i in range(0, trials):
#     print(qpt_circs[i].draw())

job = qiskit.execute(qpt_circs, Aer.get_backend('qasm_simulator'), shots=40)

# Extract tomography data so that counts are indexed by measurement configuration
qpt_tomo = ProcessTomographyFitter(job.result(), qpt_circs)
data = qpt_tomo.data
for i in data:
    print(i, data[i])

# Tomographic reconstruction
choi_fit_lstsq = qpt_tomo.fit(method='lstsq')
print(choi_fit_lstsq.data)

# print('Average gate fidelity: F = {:.5f}'.format(qi.average_gate_fidelity(choi_fit_lstsq, target=target_unitary)))