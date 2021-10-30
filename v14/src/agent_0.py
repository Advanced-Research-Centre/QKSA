from src.environment import environment
from src.least import least
from src.metrics import metrics
from src.qpt_1 import qpt

import importlib
import random
import copy
import numpy as np
from numpy.random import choice
import os	
from math import pi, floor, inf
from numpy_ringbuffer import RingBuffer		# pip install numpy_ringbuffer
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from progress.bar import Bar

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

	LOG_TEST_1 = []		
	LOG_TEST_2 = []		
	LOG_TEST_3 = []			
	LOG_TEST_4 = []			

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
			rho = qi.DensityMatrix.from_instruction(exp_env.qpCirc)           		# Actual Density Matrix
			self.qptPool.append([i,agtObj,exp_env,rho.data])

		self.newChildName 	= ''
		self.childCtr 		= 0
		self.alive			= True

		self.agt_life = Bar('Progress...', width = 64, fill='█', max=self.lifespan, suffix = '%(index)d/%(max)d steps [%(elapsed)s / %(eta)d sec]')
		self.LOG_TEST_1 = []		
		self.LOG_TEST_2 = []		
		self.LOG_TEST_3 = []			
		self.LOG_TEST_4 = []	

	# Core method
	def act(self, env, a_t_star):
		'''
		Pass action to the environment
		TBD: raise error if a_t_star is not a member of the set A
		'''
		env.action(a_t_star)
		return

	# Core method
	def perceive(self, env):
		'''
		Receive perception from the environment
		TBD: raise error if e_t is not a member of the set E
		'''
		e_t = env.perception(list(range(0,env.num_qb)))
		return e_t[0]

	# Core method
	def c_est(self, qpt):
		'''
		Estimate cost using LEAST metric and cost function 
		'''
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
		'''
		Distance function between elements in percept space
		'''
		return self.metrics.DeltaT(e_i, e_j)

	# Utility method
	def toStr(self,n,base):
		'''
		Convert a decimal number to base-n
		'''
		convertString = "0123456789ABCDEF"
		if n < base:
			return convertString[n]
		else:
			return self.toStr(n//base,base) + convertString[n%base]

	# Test method
	def loadHist(self, fname):
		'''
		Load history (from previous sessions) from file
		'''
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

	# Core method
	def predict(self, qpt, past_a, a_k, past_e, e_pred_k):
		'''
		Given hist of a,e, a_t and e_t, predict the probability of e_t
		'''
		test = False
		if (test == True):
			lambda_e = 1/len(qpt[2].E)
			return lambda_e

		rho_choi = qpt[1].est_choi(past_a, past_e) # Current estimated environment from history of action and perception till last step (t-1)
		pr_qcirc = QuantumCircuit(qpt[1].num_qb)
		prb = list(reversed(a_k))
		for i in range(0,len(prb)):
			if prb[i] == '1':
				pr_qcirc.ry(-pi/2,i) 	# for measuring along X
			elif prb[i] == '2':
				pr_qcirc.rx(pi/2,i) 	# for measuring along Y
		prU = qi.Operator(pr_qcirc)		# post-rotation unitary
		pr_rho_choi = np.matmul(prU.data,  np.matmul(rho_choi, prU.conjugate().transpose().data) )	# M rho M_dag
		dist_pred = np.diag(np.real(pr_rho_choi))
		lambda_e = dist_pred[int(e_pred_k,2)]
		return lambda_e

	# Core method
	def policy(self, qpt):
		'''
		Given the history, choose an action for the current step that would have the highest utility
		'''
		pbar = False
		test = False
		if (test == True):
			a_t_star = random.choice(qpt[2].A)
			u_pred_star = 1
			return ['E', a_t_star], u_pred_star
	
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
				if pbar == True:
					bar.next()
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
		 
		if pbar == True:
			d_tree_sz = (len(qpt[2].A)*len(qpt[2].E))**self.t_f
			bar = Bar('Progress...', width = 64, fill='█', max=d_tree_sz, suffix = '%(index)d/%(max)d steps [%(elapsed)s / %(eta)d sec]')
		futureCone(self.t, list(self.hist_a), list(self.hist_e), [])
		if pbar == True:
			bar.finish()

		# action with max utility (issue: never chooses certain actions, affects QPT reconstruction)
		# a_t_star = max(dTree, key=dTree.get)
		# u_pred_star = max(dTree.values())

		# action weighted by utility
		dlist = dTree.items()
		nullBias = 0.2												# Such that even Identity is chosen
		pdist = [nullBias+delem[1] for delem in dlist]
		draw = choice([delem[0] for delem in dlist], 1, p=pdist/sum(pdist))
		u_pred_star = dTree[draw[0]]

		return ['E', draw[0]], u_pred_star

	# Core method
	def mutate(self):
		'''
		Mutate current agent's gene
		'''
		genes_new = self.genes
		return genes_new

	# Core method
	def constructor(self, genes):
		'''
		Self-replicate agent code with mutated gene
		'''
		f = open('src/'+self.newChildName+'.py', "w")
		# The expression below is used to automatically convert the embed the code within the f.write
		# QUINE = 0
		dna = '\
dna=%r\n\
\tf.write(dna%%(dna,genes))\n\
\tgenes = %r'
		f.write(dna%(dna,genes))
		f.close()
		return

	# Core method
	def halt(self):
		'''
		Cleanup activities when agent is terminated automatically/manually
		'''
		for qpt in self.qptPool:
			print("\n   QPT strategy: "+qpt[1].name, '\n')
			rho_choi_curr = qpt[1].est_choi(self.hist_a, self.hist_e)
			print("Final estimated environment:\n")
			for line in rho_choi_curr:
				print ('  '.join(map(str, line)))
			print()		
		if hasattr(self, 'exp_env'):
			self.exp_env.suspendEnv()
		self.alive = False
		self.agt_life.finish()				# Progress Bar
		return

	# Core method
	def run(self):
		# Loop handled by Hypervisor
		showQPT = False	
		showLife = True
		self.t_p_max = self.t if (self.t-self.t_p) < 0 else self.t_p	 # How much historic data is available (adjusted for initial few steps) for calculating return
		
		qpt_star = []
		a_t_star = []
		u_pred_star = []
		c_u_star = 10000000
		for qpt in self.qptPool:					# For each process reconstruction algorithm (qpt)
			
			# print("   QPT strategy: "+qpt[1].name, qpt,'\n')

			if self.t == 0:
				rho_choi_curr = qpt[1].est_choi(self.hist_a, self.hist_e)
				print("Target environment:\n")
				for line in qpt[3]:
					print ('  '.join(map(str, line)))
				print()	
				print("Initial estimated environment:\n")
				for line in rho_choi_curr:
					print ('  '.join(map(str, line)))
				print()			
		
			# Run policy to use that qpt to generate best action and prediction based on estimated utility
			qpt_a_t, qpt_u_pred = self.policy(qpt)		# Action chosen by the agent at time step t.
			c_least_est = self.c_est(qpt)
			if qpt_u_pred * 2**(-c_least_est) < c_u_star:			# Choose by weighted roulette?
				c_u_star = qpt_u_pred * 2**(-c_least_est)
				qpt_star = qpt
				a_t_star = qpt_a_t
				u_pred_star = qpt_u_pred

		if showQPT == True:
			print("Chosen QPT strategy for step",self.t," :",qpt_star[1].name, a_t_star)
		elif showLife == True:
			self.agt_life.next()

		self.act(qpt_star[2], a_t_star)		# Action performed by the agent at time step t.
		e_t = self.perceive(qpt_star[2])	# Perception recorded by the agent at time step t.
		
		# Use that on qpt to get new model
		rho_choi_curr = qpt_star[1].est_choi(self.hist_a, self.hist_e)
		self.hist_a.append(a_t_star)		# Update action history
		self.hist_e.append(e_t)				# Update perception history
		rho_choi_next = qpt_star[1].est_choi(self.hist_a, self.hist_e)
		# Assess actual utility
		u_t = self.Delta(rho_choi_next, rho_choi_curr)
		# Calculate Knowledge Gain
		R_t = u_pred_star - u_t
		
		self.LOG_TEST_1.append(u_pred_star)
		self.LOG_TEST_2.append(u_t)
		self.LOG_TEST_3.append(R_t)
		self.LOG_TEST_4.append(self.Delta(rho_choi_curr,qpt_star[3]))

		self.newChildName = ''
		if (self.R_t < self.R_R):								# Reproduce
			self.newChildName = 'agent_'+int(self.childCtr)
			self.childCtr += 1
			self.constructor(self.mutate())
			
		if (self.R_t < self.R_D or self.t == self.lifespan):	# Halt agent (die)
			self.halt()

		self.t += 1	