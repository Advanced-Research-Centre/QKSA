from src.environment import environment
from src.least import least
from src.metrics import metrics
from src.qpt_1 import qpt

import importlib
import random
import copy
import numpy as np
import os	
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

	LOG_TEST = []						

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
		self.qptNos		= 2															# Number of QPT techniques to be considered
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
	def c_est(self, qpt):
		# Cost estimator using LEAST metric and cost function 
		fname = 'src\\qpt_'+str(qpt[0])+'.py'
		l_est = os.path.getsize(fname)
		t_est = qpt[1].t_est
		est_least = [l_est, 0, 0, 0, t_est]
		# est_least = [least.L_est(data), least.E_est(data), least.A_est(data), least.S_est(data), least.T_est(data)]		# TBD
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
		# print(est_least,wt_least_est,c_est)
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
	def predict(self, qpt, past_a, a_k, past_e, e_pred_k):
		# Given hist of a,e, a_t and e_t, predict the probability of e_t	
		rho_choi = qpt[1].est_choi(past_a, past_e) # Current estimated environment from history of action and perception till last step (t-1)
		# Perception based on rho_choi and a_t
		pr_qcirc = QuantumCircuit(qpt[1].num_qb)
		prb = list(reversed(a_k))
		for i in range(0,len(prb)):
			if prb[i] == '1':
				pr_qcirc.ry(-pi/2,i) 	# for measuring along X
			elif prb[i] == '2':
				pr_qcirc.rx(pi/2,i) 	# for measuring along Y
		prU = qi.Operator(pr_qcirc)		# post-rotation unitary
		pr_rho_choi = np.matmul(prU.data,  np.matmul(rho_choi, prU.conjugate().transpose().data) )
		dist_pred = np.diag(np.real(pr_rho_choi))
		lambda_e = dist_pred[int(e_pred_k,2)]
		return lambda_e

	# Core method
	def policy(self, qpt):
		'''
		Given the history, choose an action for the current step that would have the highest utility
		'''
		test = True
		if (test == True):
			a_t_star = random.choice(qpt[2].A)
	
		#
		# for each t_f
		#	reconstruct rho
		#	max_util = -1; a_star = A[0]
		#	for each action a
		#		tot_util = 0
		#		for each prediction e
		#			find probability of e_pred as lambda_e_pred
		#			reconstruct rho_pred
		#			find utility u_pred as distance between rho and rho_pred
		#			tot_util = lambda_e_pred * u_pred
		#		if tot_util > max_util:
		# 			max_util = tot_util
		# 			a_star = a
		#

		rho_choi_t = qpt[1].est_choi(self.hist_a, self.hist_e)	# Current model of the environment [Get least cost while doing this]
		dTree = {}
		a_t_star = qpt[2].A[0]									# Optimal action for the agent determined by the policy at time step t	
		
		def futureCone(k, past_a, past_e, lambda_e_pred):
			nonlocal a_t_star
			if k < self.t+self.t_f:
				for a_k in qpt[2].A: 
					past_a_new = copy.deepcopy(past_a)
					past_a_new.append(['E', a_k])
					for e_pred_k in qpt[2].E:
						past_e_new = copy.deepcopy(past_e) 
						past_e_new.append(e_pred_k)
						lambda_e_pred_new = copy.deepcopy(lambda_e_pred) 
						lambda_e_pred_new.append(self.predict(qpt, past_a, a_k, past_e, e_pred_k))
						futureCone(k+1, past_a_new, past_e_new, lambda_e_pred_new)
			else:
				lambda_e_pred_m = 1									# Find total probability of sequence of predicted action-perception for t_f steps
				for lambda_e_pred_k in lambda_e_pred:
					lambda_e_pred_m *= lambda_e_pred_k
				rho_choi_m_pred = qpt[1].est_choi(past_a, past_e)	# Find predicted model based on predicted action-perception for t_f steps
				u_pred = self.Delta(rho_choi_m_pred, rho_choi_t)
				a_t = past_a[self.t]
				if a_t[1] in dTree:									# Cumulate weighted utility for same actions in current step
					dTree[a_t[1]] += lambda_e_pred_m*u_pred
				else:
					dTree[a_t[1]] = lambda_e_pred_m*u_pred

		futureCone(self.t, list(self.hist_a), list(self.hist_e), [])
		a_t_star = max(dTree, key=dTree.get)
		u_pred_star = max(dTree.values())
		
		return ['E', a_t_star], u_pred_star

	# Core method
	def mutate(self):
		# mutate current agent's gene
		genes_new = self.genes
		return genes_new

	# Core method
	def constructor(self, genes):
		f = open('src/'+self.newChildName+'.py', "w")
		# The expression below is used to automatically convert the embed the code within the f.write
		# QUINE = FALSE
		dna = '\
dna=%r\n\
\tf.write(dna%%(dna,genes))\n\
\tgenes = %r'
		f.write(dna%(dna,genes))
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
	def run(self):
		# Loop handled by Hypervisor
			
		self.t_p_max = self.t if (self.t-self.t_p) < 0 else self.t_p	 # How much historic data is available (adjusted for initial few steps) for calculating return
		
		qpt_star = []
		c_u_star = 10000000
		for qpt in self.qptPool:					# For each process reconstruction algorithm (qpt)
			
			# print("   QPT strategy: "+qpt[1].name, qpt,'\n')

			if self.t == 0:
				rho_choi_curr = qpt[1].est_choi(self.hist_a, self.hist_e)
				print("Initial estimated environment:\n")
				for line in rho_choi_curr:
					print ('  '.join(map(str, line)))
				print()			
		
			# Run policy to use that qpt to generate best action and prediction based on estimated utility
			a_t_star, u_pred_star = self.policy(qpt)		# Action chosen by the agent at time step t.
			c_least_est = self.c_est(qpt)
			if u_pred_star*c_least_est < c_u_star:			# Choose by weighted roulette?
				c_u_star = u_pred_star*c_least_est
				qpt_star = qpt

		print("Chosen QPT strategy for step",self.t," :",qpt_star[1].name)
		self.act(qpt_star[2], a_t_star)		# Action performed by the agent at time step t.
		e_t = self.perceive(qpt_star[2])	# Perception recorded by the agent at time step t.
		
		# Use that on qpt to get new model and access reward/return/utility
		rho_choi_curr = qpt_star[1].est_choi(self.hist_a, self.hist_e)
		self.hist_a.append(a_t_star)		# Update action history
		self.hist_e.append(e_t)				# Update perception history
		rho_choi_next = qpt_star[1].est_choi(self.hist_a, self.hist_e)
		u_t = self.Delta(rho_choi_next, rho_choi_curr)
		R_t = u_pred_star - u_t
		self.LOG_TEST.append(R_t)

		self.newChildName = ''
		if (self.R_t < self.R_R):								# Reproduce
			self.newChildName = 'agent_'+int(self.childCtr)
			self.childCtr += 1
			self.constructor(self.mutate())
			
		if (self.R_t < self.R_D or self.t == self.lifespan):	# Halt agent (die)
			self.halt()

		self.t += 1	