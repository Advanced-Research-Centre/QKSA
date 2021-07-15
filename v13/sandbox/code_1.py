'''
Comparison between raw, pyBDM, pyflate for binary Fibonacci series
'''

import numpy as np
from pybdm import BDM
import os

def createFibo(size):
    lenS = 0
    lenlim = 2**size[-1]
    n1 = 0
    n2 = 1
    S = np.array([],dtype=int)
    while (lenS < lenlim):
        b1 = bin(n1)
        b2 = bin(n2)
        n3 = n1+n2
        n1 = n2
        n2 = n3
        for i in b1[2:]:
            lenS += 1
            S = np.append(S,int(i))
            if lenS > lenlim:
                return S    

def create0101(size):
    S = np.array([0,1]*(2**(size[-1]-1)))
    return S

def estKC_oriS(size, S1, S2):
    print("\n--- Original string length and file size ---\n")
    for i in range(size[0],size[-1]+1):
        np.savez("tempfiles\S1_"+str(i), S1[0:2**i])
        sizeS1 = os.path.getsize("tempfiles\S1_"+str(i)+".npz")
        print(i,"\t", len(S1[0:2**i]),"\t\t\t",sizeS1*8)

# https://www.mdpi.com/1099-4300/20/8/605
def estKC_pyBDM(size, S1, S2):
    print("\n--- KC estimation using pyBDM ---\n")
    aprxKC = BDM(ndim=1)
    for i in range(size[0],size[-1]+1):
        print(i,"\t",aprxKC.bdm(S1[0:2**i]),"\t",aprxKC.bdm(S2[0:2**i]))

# https://en.wikipedia.org/wiki/Deflate
def estKC_pyflate(size, S1, S2):
    print("\n--- KC estimation using pyflate ---\n")
    for i in range(size[0],size[-1]+1):        
        np.savez_compressed("tempfiles\S1c_"+str(i), S1[0:2**i])
        sizeS1c = os.path.getsize("tempfiles\S1c_"+str(i)+".npz")
        np.savez_compressed("tempfiles\S2c_"+str(i), S2[0:2**i])
        sizeS2c = os.path.getsize("tempfiles\S2c_"+str(i)+".npz")
        print(i,"\t", sizeS1c*8,"\t\t\t",sizeS2c*8)
    
# (qeait) D:\GoogleDrive\RESEARCH\0 - Programs\AutonomousQuantumPhysicist\v12\IR>python w1.py

size = [4,18]
S1 = createFibo(size)
S2 = create0101(size)

estKC_oriS(size, S1, S2)

'''
   --- Original string length and file size ---

    4        16                      2624
    5        32                      3136
    6        64                      4160
    7        128                     6208
    8        256                     10304
    9        512                     18496
    10       1024                    34880
    11       2048                    67648
    12       4096                    133184
    13       8192                    264256
    14       16384                   526400
    15       32768                   1050688
    16       65536                   2099264
    17       131072                  4196416
    18       262144                  8390720
'''

estKC_pyBDM(size, S1, S2)

'''
    --- KC estimation using pyBDM ---

    4        30.420342613363147      26.99072664916141
    5        63.462173603472564      27.99072664916141
    6        163.33319418147045      29.31265474404877
    7        325.9210706482428       30.31265474404877
    8        675.649934997366        31.38304407194017
    9        1372.2912105292796      32.38304407194017
    10       2793.0853595703325      33.400117585299114
    11       5494.124201555914       34.400117585299114
    12       10837.05634907019       35.40435457818558
    13       20558.892569829062      36.40435457818558
    14       38593.535496655684      37.40541188496862
    15       65896.1362917695        38.40541188496862
    16       100139.07032123256      39.40567609063567
    17       129018.32557622659      40.40567609063567
    18       141702.91302816267      41.40574213449299
'''

estKC_pyflate(size, S1, S2)

'''
    --- KC estimation using pyflate ---

    4        1728                    1704
    5        1760                    1704
    6        1864                    1712
    7        2048                    1728
    8        2424                    1760
    9        3040                    1816
    10       4168                    1936
    11       6272                    2040
    12       10424                   2128
    13       18808                   2304
    14       35240                   2696
    15       68360                   3456
    16       134536                  4968
    17       266456                  8040
    18       530328                  14128
'''