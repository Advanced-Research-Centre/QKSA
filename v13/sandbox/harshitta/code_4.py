import numpy as np
from qpt import qpt
from numpy_ringbuffer import RingBuffer		# pip install numpy_ringbuffer

'''
Trace distance between two density matrices
https://en.wikipedia.org/wiki/Trace_distance
'''
def DeltaDM(dm_i, dm_j):
    # Distance function between elements in percept space
    dist_dm_ij = np.real(np.trace(np.sqrt(np.matmul((dm_i - dm_j).conjugate().transpose(),dm_i - dm_j))) / 2)
    return dist_dm_ij

'''
Convert a decimal number to base-n
'''
def toStr(n,base):
    convertString = "0123456789ABCDEF"
    if n < base:
        return convertString[n]
    else:
        return toStr(n//base,base) + convertString[n%base]

def loadHist(fname):
    global hist_a, hist_e
    fobj = open(fname, "r")
    for i in range(0,4**2):
        ps = toStr(i,4).zfill(2)
        res = fobj.readline()
        i = 2
        while (i < len(res)):
            hist_a.append(["E",ps])
            hist_e.append(res[i:i+2])
            i += 6
    fobj.close()

num_qb  = 2
t_p     = 16384
hist_a	= RingBuffer(capacity=t_p, dtype='object')	# History of actions
hist_e	= RingBuffer(capacity=t_p, dtype='object')	# History of perceptions

p_qpt = qpt (num_qb)

rho_choi_init = p_qpt.est_choi(hist_a, hist_e)                  # With no preceptions
		
loadHist("AAQPT_full.txt")
rho_choi = p_qpt.est_choi(hist_a, hist_e)                       # With all preceptions

# Harshitta, change this part
# ---------------------------
hist_a_part = hist_a
hist_e_part = hist_e
# ---------------------------

rho_choi_pred = p_qpt.est_choi(hist_a_part, hist_e_part)        # With limited preceptions

kg = DeltaDM(rho_choi_init,rho_choi)
print("Reward/Utility:",kg)
kg = DeltaDM(rho_choi_pred,rho_choi)
print("Reward/Utility:",kg)

'''
(qeait) D:\GoogleDrive\RESEARCH\A1 - Programs\QKSA\v12\sandbox\harshitta>python code_4.py
Reward/Utility: 0.8673336117409827
Reward/Utility: 0.0
'''