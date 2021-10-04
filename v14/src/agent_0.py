from src.environment import environment
from src.least import least
from src.metrics import metrics
from src.qpt_1 import qpt

import random
import copy
import numpy as np
from math import pi, floor, inf
from numpy_ringbuffer import RingBuffer		# pip install numpy_ringbuffer
from datetime import datetime
from qiskit import QuantumCircuit, qasm, Aer, execute
from qiskit import *
import qiskit.quantum_info as qi

class agent:
				
	# Mutable hyper-parameters genes
	c_gene		= '' 
	wt_gene		= []
	l_max		= 0	
	e_max		= 0	
	a_max		= 0	
	s_max		= 0	
	t_max		= 0	
	m_c			= 0
	# Immutable hyper-parameters genes
	neighbours 	= None		
	t_p			= 0						
	t_f			= 0							
	gamma		= 0						
	R_R			= 0							
	R_D			= 0							
	lifespan	= 0						

	def __init__(self, name, env, genes):
				
		if (name == 'agent_0'): # Create Seed QKSA from prespecified gene
			# Mutable hyper-parameters genes
			self.c_gene		= genes[0]
			self.wt_gene	= genes[1]
			self.l_max		= genes[2]	
			self.e_max		= genes[3]	
			self.a_max		= genes[4]	
			self.s_max		= genes[5]	
			self.t_max		= genes[6]	
			self.m_c		= genes[7]
			# Immutable hyper-parameters genes
			self.neighbours = genes[8]	
			self.t_p		= genes[9]				
			self.t_f		= genes[10]					
			self.gamma		= genes[11]				
			self.R_R		= genes[12]					
			self.R_D		= genes[13]					
			self.lifespan	= genes[14]			
		
		self.name	= name
		self.env	= env
		self.genes	= [self.c_gene, self.wt_gene, self.l_max, self.e_max, self.a_max, self.s_max, self.t_max, self.m_c, self.neighbours, self.t_p, self.t_f, self.gamma, self.R_R, self.R_D, self.lifespan]

		self.t			= 0									# Age of agent in number of runStep calls
		self.R_t 		= 0									# Cumulative discounted return at time step t.
		self.hist_a		= RingBuffer(capacity=self.t_p, dtype='object')						# History of actions
		self.hist_e		= RingBuffer(capacity=self.t_p, dtype='object')						# History of perceptions
		self.hist_r		= RingBuffer(capacity=self.t_p)	
		
		self.define_A(len(self.neighbours))						# Action space [NEEDED?]
		self.define_E(len(self.neighbours))						# Percept space [NEEDED?]

		self.least		= least()
		self.metrics	= metrics()

		self.newChildName 	= ''
		self.childCtr 		= 0
		self.alive			= True

	# Utility method		
	def DecToBaseN(self, n, b, l):
		s = ''
		while n != 0:
			s = str(n%b)+s
			n = n//b
		return ('0'*(l-len(s)))+s

	# Core method
	def define_A(self, numQb):
		# define action space A as all 3-axis basis of numQb qubits
		self.A = []
		for i in range(3**numQb):
			self.A.append(str(self.DecToBaseN(i,3,numQb)))
		return

	# Core method
	def define_E(self, numQb):
		# Define percept space E as all binary strings of numQb qubits
		self.E = []
		for i in range(2**numQb):
			self.E.append(str(self.DecToBaseN(i,2,numQb)))
		return

	# Core method
	def act(self, a_t_star):
		# check if a_t_star is a member of the set A, else raise error
		self.exp_env.action(a_t_star)
		return

	# Core method
	def perceive(self):
		e_t = self.exp_env.perception(list(range(0,self.exp.num_qb)))
		# check if e_t is a member of the set E, else raise error
		return e_t[0]

	# Core method
	def c_est(self, data):
		# Cost estimator using LEAST metric and cost function 
		est_least = [least.L_est(data), least.E_est(data), least.A_est(data), least.S_est(data), least.T_est(data)]
		if (est_least[0] > self.l_max) or (est_least[1] > self.e_max) or (est_least[2] > self.a_max) or (est_least[3] > self.s_max) or (est_least[4] > self.t_max):
			return -1
		wt_least_est = [self.wt_gene[i] * est_least[i] for i in range(5)]
		stack = []
		for c in range(len(self.c_gene)-2,-1,-2):
			if self.c_gene[c] == 'V':
				stack.append(wt_least_est[int(self.c_gene[c+1])])
			else:
				o1 = stack.pop()
				o2 = stack.pop()
				if self.c_gene[c+1] == '0':
					stack.append(o1 + o2)		# F0: addition
				elif self.c_gene[c+1] == '1':
					stack.append(o1 * o2)		# F1: multiplication
		c_est = stack.pop()
		return c_est

	# Core method
	def Delta(self, e_i, e_j):
		# Distance function between elements in percept space
		return self.metrics.DeltaTD(e_i, e_j)

	# Utility method
	def partialTrace1(self, dm):
		# Partial trace of the lower significant subsystem of 2 qubits
		dm1 = np.zeros((2,2)) * 0j
		for i in range(0,2):
			for j in range(0,2):
				dm1[i][j] = dm[i][j]+dm[i+2][j+2]
		return dm1

	# Utility method
	def toStr(self,n,base):
		# Convert a decimal number to base-n
		convertString = "0123456789ABCDEF"
		if n < base:
			return convertString[n]
		else:
			return self.toStr(n//base,base) + convertString[n%base]

	# Test method
	def loadHist(self, fname):
		# USAGE: self.loadHist("data/AAQPT_full.txt")						
		# USAGE: rho_choi_full = self.exp.est_choi(self.hist_a, self.hist_e)
		fobj = open(fname, "r")
		for i in range(0,4**2):
			ps = self.toStr(i,4).zfill(2)
			res = fobj.readline()
			i = 2
			while (i < len(res)):
				self.hist_a.append(["E",ps])
				self.hist_e.append(res[i:i+2])
				i += 6
		fobj.close()

	# Test method
	def log(self, desc):
		fname = open("results/runlog_"+desc+".txt", "a")
		now = datetime.now()
		fname.write("\n"+str(now)+"\n")
		fname.write("\n"+str(desc)+"\n")
		for r in self.hist_r:
			fname.write(str(r)+"\n")
		fname.close()
		return

	# Core method
	def policy(self):
		'''
		Given the history, choose an action for the current step that would have the highest utility
		'''
		return

	# Core method
	def mutate(self):
		# mutate current agent's gene
		genes_new = self.genes
		return genes_new

	# Core method
	def constructor(self, genes):
		f = open('src/'+self.newChildName+'.py', "w")
		# add Quine code here
		f.write("s")
		f.close()
		return

	# Core method
	def halt(self):
		self.exp_env.suspendEnv()
		self.alive = False
		return

	# Core method
	def predict(self, a_t_star):
		'''
		Use hist_{a,e} to predict rho_t_star from a_t_star
			if multiplicity of a_t_star is less that trail, return random rho_t_star
			else return rho_t_star based on probability of hist_e for a_t_star
		'''
		# NEW: Given hist of a,e, a_t and e_t, predict the probability of e_t
		return



	# Core method
	def run(self):
		
		# Loop handled by Hypervisor
			
		self.t_p_max = self.t if (self.t-self.t_p) < 0 else self.t_p	 # How much historic data is available (adjusted for initial few steps) for calculating return
		
		print("run step")
		# Choose process reconstruction algorithm (qpt)
		# Run policy to use that qpt to generate best action and prediction based on estimated utility
		# Run action and get perception
		# Use that on qpt to get new model and access reward/return/utility

			

		
		self.newChildName = ''
		if (self.R_t < self.R_R):								# Reproduce
			self.newChildName = 'agent_'+int(self.childCtr)
			self.childCtr += 1
			self.constructor(self.mutate())
			
		if (self.R_t < self.R_D or self.t == self.lifespan):	# Halt agent (die)
			self.halt()