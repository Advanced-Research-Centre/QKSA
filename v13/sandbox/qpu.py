import sys
fname = open(sys.path[0] + '\..\..\ibmq.txt')
api = fname.readline()
fname.close()

from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.tools.monitor import job_monitor

IBMQ.enable_account(api)
provider = IBMQ.get_provider('ibm-q')
# backends = provider.backends(filters = lambda x:x.configuration().n_qubits >= 2 and not x.configuration().simulator and x.status().operational==True)
belem = provider.get_backend('ibmq_belem')

qcirc = QuantumCircuit(1,1)
qcirc.h(0)
qcirc.measure(0,0)

job = execute(qcirc, backend=belem, shots=200)
job_monitor(job)
output = job.result().get_counts()
print(output)

IBMQ.disable_account()

'''
(qeait) D:\GoogleDrive\RESEARCH\A1 - Programs\QKSA\v13>python qpu.py
D:\Users\aritr\anaconda3\envs\qeait\lib\site-packages\qiskit\providers\ibmq\ibmqfactory.py:109: UserWarning: Timestamps in IBMQ backend properties, jobs, and job results are all now in local time instead of UTC.
  warnings.warn('Timestamps in IBMQ backend properties, jobs, and job results '
Job Status: job has successfully run
{'0': 98, '1': 102}
'''