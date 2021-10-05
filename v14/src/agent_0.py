from src.environment import environment
from src.least import least
from src.metrics import metrics
from src.qpt_1 import qpt

import importlib
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

		self.t			= 0													# Age of agent in number of runStep calls
		self.R_t 		= 0													# Cumulative discounted return at time step t
		self.hist_a		= RingBuffer(capacity=self.t_p, dtype='object')		# History of actions
		self.hist_e		= RingBuffer(capacity=self.t_p, dtype='object')		# History of perceptions
		self.hist_r		= RingBuffer(capacity=self.t_p)	
		self.least		= least()
		self.metrics	= metrics()

		# Make set of QPT objects based on available techniques
		self.qptNos		= 1															# Number of QPT techniques to be considered
		self.qptPool	= []
		for i in range(0,self.qptNos):
			agtClass = getattr(importlib.import_module('src.qpt_'+str(i)), "qpt")	# Filename changes for each qpt while class name remains same
			agtObj = agtClass(self.env.num_qb)										# QPT gets instantiated here
			exp_env = environment(agtObj.num_qb)
			print("\nCreating Environment for "+self.name+" QPT algorithm "+agtObj.name)
			exp_env.createEnv(agtObj.setup(self.env.qpCirc))
			self.qptPool.append([i,agtObj,exp_env])

		self.newChildName 	= ''
		self.childCtr 		= 0
		self.alive			= True

	# Core method
	def act(self, env, a_t_star):
		# check if a_t_star is a member of the set A, else raise error
		env.action(a_t_star)
		return

	# Core method
	def perceive(self, env):
		e_t = env.perception(list(range(0,env.num_qb)))
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
	def policy(self, env):
		'''
		Given the history, choose an action for the current step that would have the highest utility
		'''
		a_t_star = ['E', random.choice(env.A)]
		return a_t_star

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
		for qpt in self.qptPool:
			print("   QPT strategy: "+qpt[1].name, qpt,'\n')
			rho_choi_curr = qpt[1].est_choi(self.hist_a, self.hist_e)
			print("Current estimated environment:\n")
			for line in rho_choi_curr:
				print ('  '.join(map(str, line)))
			print()		
		if hasattr(self, 'exp_env'):
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
		
		# print(self.A)
		# print(self.E)
		qpt_star = []

		
		for qpt in self.qptPool:					# For each process reconstruction algorithm (qpt)
			
			# print("   QPT strategy: "+qpt[1].name, qpt,'\n')

			if self.t == 0:
				rho_choi_curr = qpt[1].est_choi(self.hist_a, self.hist_e)
				print("Current estimated environment:\n")
				for line in rho_choi_curr:
					print ('  '.join(map(str, line)))
				print()			
		
			# Run policy to use that qpt to generate best action and prediction based on estimated utility
			a_t_star = self.policy(qpt[2])		# Action chosen by the agent at time step t.
			# rho_t_star = self.predict(a_t_star)
			qpt_star = qpt

		self.act(qpt_star[2], a_t_star)		# Action performed by the agent at time step t.
		e_t = self.perceive(qpt_star[2])	# Perception recorded by the agent at time step t.
		self.hist_a.append(a_t_star)		# Update action history
		self.hist_e.append(e_t)				# Update perception history
		
		# Use that on qpt to get new model and access reward/return/utility

		self.newChildName = ''
		if (self.R_t < self.R_R):								# Reproduce
			self.newChildName = 'agent_'+int(self.childCtr)
			self.childCtr += 1
			self.constructor(self.mutate())
			
		if (self.R_t < self.R_D or self.t == self.lifespan):	# Halt agent (die)
			self.halt()

		self.t += 1	