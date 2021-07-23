import numpy as np
from pybdm import BDM
import pandas
from math import pi, floor, inf
from numpy_ringbuffer import RingBuffer		# pip install numpy_ringbuffer
import random
import copy
from qiskit import QuantumCircuit, qasm, Aer, execute
from qiskit import *
from qpt import qpt
from environment import environment
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt
from datetime import datetime

class agent:
				
	pi_t = 0		# policy for choosing an optimal action and maximizing return at time step t.
	trials = 20		# number of tomographic trials in growing phase
	history = []
	
	def __init__(self, env, genes):
				
		self.env = env
		self.genes = genes

		self.neighbours = genes[0]
		self.boundary = len(self.neighbours)

		self.define_A(self.boundary)						# Action space
		self.define_E(self.boundary)						# Percept space
		# self.history = np.empty((pow(3,self.boundary),self.trials),dtype='object')

		self.c_gene		= genes[1]
		self.wt_gene	= genes[2]

		# Initialize (optionally) mutable hyper-parameters
		self.l_max		= genes[3]	
		self.e_max		= genes[4]
		self.a_max		= genes[5]
		self.s_max		= genes[6]
		self.t_max		= genes[7]	
		self.m_c		= genes[8]	
		self.m			= genes[9]	
		self.n			= genes[10]	
		self.s_c		= genes[11]	
		self.t_p		= genes[12]	
		self.t_f		= genes[13]
		self.gamma		= genes[14]	
		self.R_D		= genes[15]	
		self.R_R		= genes[16]	
		self.lifespan	= genes[17]	

		self.t			= 0									# Age of agent in number of runStep calls
		self.R_t 		= self.R_R							# Cumulative discounted return at time step t.
		self.hist_a		= RingBuffer(capacity=self.t_p, dtype='object')						# History of actions
		self.hist_rho	= RingBuffer(capacity=self.t_p, dtype='U'+str(self.boundary))		# History of predictions (needed?)
		self.hist_e		= RingBuffer(capacity=self.t_p, dtype='object')						# History of perceptions
		self.hist_r		= RingBuffer(capacity=self.t_p)										# History of rewards

		print("Agent Alive and Ticking")
			
	def act(self, a_t_star):
		self.exp_env.action(a_t_star)
		return

	def perceive(self):
		# return self.E[random.randint(0, 2**self.boundary-1)]	# Test code
		e_t = self.exp_env.measure(list(range(0,self.exp.num_qb)))
		# check if e_t is a member of the set E, else raise error
		return e_t[0]

	def L_est(self, data):
		# Function to estimate the program length
		if len(data) < 13:
			return 0
		aprxKC = BDM(ndim=1)
		l_est = aprxKC.bdm(data)
		return l_est

	def E_est(self, data):
		# Function to estimate the thermodynamic (energy) cost
		e_est = 0
		return e_est

	def A_est(self, data):
		# Function to estimate the approximation margin
		a_est = 0
		return a_est

	def S_est(self, data):
		# Function to estimate the working memory (space)
		s_est = 0
		return s_est

	def T_est(self, data):
		# Function to estimate the run-time
		t_est = 0
		if len(data) < 13:	# LUT only till length 12, TBD: Sliding window Logical Depth
			ld_db = pandas.read_csv('data/logicalDepthsBinaryStrings.csv',names=['BinaryString', 'LogicalDepth'],dtype={'BinaryString': object,'LogicalDepth': int}) # https://github.com/algorithmicnaturelab/OACC/blob/master/data/logicalDepthsBinaryStrings.csv
			data_str = ""
			for b in data: data_str = data_str+str(b)
			t_est = ld_db[ld_db['BinaryString'].dropna().str.fullmatch(data_str)]['LogicalDepth'].values[0]
		return t_est

	def c_est(self, data):
		# Cost estimator using LEAST metric and cost function 
		est_least = [self.L_est(data), self.E_est(data), self.A_est(data), self.S_est(data), self.T_est(data)]
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

	def DecToBaseN(self, n, b, l):
		s = ''
		while n != 0:
			s = str(n%b)+s
			n = n//b
		return ('0'*(l-len(s)))+s

	def define_A(self, numQb):
		# define action space A as all 3-axis basis of numQb qubits
		self.A = []
		for i in range(3**numQb):
			self.A.append(str(self.DecToBaseN(i,3,numQb)))
		return

	def define_E(self, numQb):
		# Define percept space E as all binary strings of numQb qubits
		self.E = []
		for i in range(2**numQb):
			self.E.append(str(self.DecToBaseN(i,2,numQb)))
		return
	
	def Delta(self, e_i, e_j):
		# Distance function between elements in percept space
		dist_e_ij = 0
		for (i,j) in zip(e_i,e_j):
			if i!=j:
				dist_e_ij += 1					# Hamming distance
		dist_e_ij = len(e_i) - dist_e_ij
		return dist_e_ij

	'''
	Trace distance between two density matrices
	https://en.wikipedia.org/wiki/Trace_distance
	'''
	def DeltaDM(self, dm_i, dm_j):
		# Distance function between elements in percept space
		dist_dm_ij = np.real(np.trace(np.sqrt(np.matmul((dm_i - dm_j).conjugate().transpose(),dm_i - dm_j))) / 2)
		return dist_dm_ij

	'''
	Given the history, choose an action for the current step that would have the highest utility
	'''
	def policy(self):
		a_t = self.A[0]
		a_t_star = self.A[0]				# Optimal action for the agent determined by the policy at time step t	
		c_est_star = math.inf

		def futureCone(t_fc, past_a, past_e):
			nonlocal a_t, a_t_star, c_est_star
			if t_fc < self.t+self.t_f:
				for a in self.A: 
					past_a_new = copy.deepcopy(past_a)
					past_a_new.append(a)
					if t_fc == self.t:
						a_t = a
					for rho in self.E:
						past_e_new = copy.deepcopy(past_e) 
						past_e_new.append(rho)
						futureCone(t_fc+1, past_a_new, past_e_new)
			else:
				# ELSE clause to be defined by PID004
				'''
				Weigh current action based on hist_{a,e} pairs using least_est
					if multiplicity of any action is less that trail, do it
					if all actions are in history with >= trail
						for each action, find the c_est of filtered hist_e
						choose action with highest physical complexity 
				'''
				data = ''.join(map(str, past_a))	# for testing integration
				cost = self.c_est(np.array(list(data), dtype=int))
				if (cost >= 0) and (cost < c_est_star):
					c_est_star =  cost
					a_t_star = a_t
				# ELSE clause to be defined by PID004

		futureCone(self.t, list(self.hist_a), list(self.hist_e))
		
		return a_t_star

	'''
	Partial trace of the lower significant subsystem of 2 qubits
	'''
	def partialTrace1(self, dm):
		dm1 = np.zeros((2,2)) * 0j
		for i in range(0,2):
			for j in range(0,2):
				dm1[i][j] = dm[i][j]+dm[i+2][j+2]
		return dm1

	def predict(self, a_t_star):
		'''
		Use hist_{a,e} to predict rho_t_star from a_t_star
			if multiplicity of a_t_star is less that trail, return random rho_t_star
			else return rho_t_star based on probability of hist_e for a_t_star
		'''
		# for QPT, a_t_star comprises of a input density matrix and a measurement basis.
		# the rho_t_star is the output density matrix based on the current model

		# # Use the estimated Choi matrix to predict the output density matrix 
		# rho_inp = a_t_star[0]
		# process_qubits = 1					# TBD: Generalize to n-qubits
		# rho_out_choi_est = 2**process_qubits * self.partialTrace1( np.matmul( np.kron(np.transpose(rho_inp), np.eye(2**process_qubits) ), rho_choi ))
		# print("\nOutput Density Matrix using Estimated Choi Matrix")
		# print(rho_out_choi_est)

		# Current estimated environment from history of action and perception till last step (t-1)
		rho_choi = self.exp.est_choi(self.hist_a, self.hist_e)

		# Perception based on rho_choi and a_t
		pr_qcirc = QuantumCircuit(self.exp.num_qb)
		prb = list(reversed(a_t_star[1]))
		for i in range(0,len(prb)):
			if prb[i] == '1':
				pr_qcirc.ry(-pi/2,i) 	# for measuring along X
			elif prb[i] == '2':
				pr_qcirc.rx(pi/2,i) 	# for measuring along Y
		prU = qi.Operator(pr_qcirc)		# post-rotation unitary
		pr_rho_choi = np.matmul(prU.data,  np.matmul(rho_choi, prU.conjugate().transpose().data) )
		dist_pred = np.diag(np.real(pr_rho_choi))
		pred_rv = random.uniform(0, sum(dist_pred))
		j = 0
		for i in range(0,len(dist_pred)):
			j += dist_pred[i]
			if j > pred_rv:
				break
		rho_t_star = self.toStr(i,2).zfill(self.exp.num_qb)

		return rho_t_star

	def calcReturn(self):
		R_t = 0
		for i in range(self.t_p_max-1,-1,-1):
			R_t += self.hist_r[i] * (1 - self.gamma*(self.t_p_max-i))	
		return R_t

	def mutate(self):
		# mutate c_gene, wt_gene and other optional genes
		genes_new = self.genes
		return genes_new

	def constructor(self, genes):
		# add Quine code here
		# offload file execution to OS
		return

	'''
	Convert a decimal number to base-n
	'''
	def toStr(self,n,base):
		convertString = "0123456789ABCDEF"
		if n < base:
			return convertString[n]
		else:
			return self.toStr(n//base,base) + convertString[n%base]

	def loadHist(self, fname):

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

	def log(self):
		fname = open("results/runlog.txt", "a")
		now = datetime.now()
		fname.write("\n"+str(now)+"\n")
		for r in self.hist_r:
			fname.write(str(r)+"\n")
		fname.close()
		return

	def run(self):

		# Create Quantum Process Tomography Object and Environment
		self.exp = qpt(self.env.num_qb)
		qptCirc = self.exp.aaqpt(self.env.qpCirc)
		self.exp_env = environment(self.exp.num_qb)
		self.exp_env.createEnv(qptCirc)		

		rho_choi_pred = self.exp.est_choi(self.hist_a, self.hist_e)
		print("Initial estimated environment:\n")#,rho_choi_pred)
		for line in rho_choi_pred:
			print ('  '.join(map(str, line)))

		while (self.t < self.lifespan):

			if (self.R_t < self.R_D):			# Halt agent (die)
				print("The input and program conspired against my eternal life")
				break	

			self.t_p_max = self.t if (self.t-self.t_p) < 0 else self.t_p	 # How much historic data is available (adjusted for initial few steps) for calculating return

			a_t_star = self.exp.policy()			# Action chosen by the agent at time step t.
			self.act(a_t_star)						# Action performed by the agent at time step t.

			rho_t_star = self.predict(a_t_star)

			e_t = self.perceive()					# Perception recorded by the agent at time step t.

			self.hist_a.append(a_t_star)
			hist_e_rho = copy.deepcopy(self.hist_e)
			hist_e_rho.append(rho_t_star)
			rho_choi_pred = self.exp.est_choi(self.hist_a, hist_e_rho)
			self.hist_e.append(e_t)
			rho_choi = self.exp.est_choi(self.hist_a, self.hist_e)

			# This is the reward/utility (knowledge gain), thus, higher the knowledge gain (error in prediction) the better
			# When knowledge gain falls below a limit, learning is finished and QKSA reproduces with mutated cost fn.
			r_t = self.DeltaDM(rho_choi_pred,rho_choi)	# Reward based on Hamming/KL/Trace distance between perception and prediction
			# r_t = self.Delta(rho_t_star, e_t)	
			
			self.hist_r.append(r_t)
			self.R_t = self.calcReturn()				# Sum of knowledge gain so far

			# print("Age =",self.t,"\t--> Action :",a_t_star,"Prediction :",rho_t_star,"Perception :",e_t,"Reward :",r_t,"Return :",self.R_t)

			if (self.R_t < self.R_R):					# Reproduce
				genes_new = self.mutate(genes)
				self.constructor(genes_new)
			self.t += 1

		print("Lived life to the fullest")

		rho_choi = self.exp.est_choi(self.hist_a, self.hist_e)
		print("Learnt environment:\n")#,rho_choi)
		for line in rho_choi:
			print ('  '.join(map(str, line)))

		# self.log()

		plt.plot(list(self.hist_r))
		plt.ylabel('trace distance')
		plt.ylim(0,1)
		plt.show()

		# print("Target environment:\n",rho_choi_analytical)

		# self.loadHist("data/AAQPT_full.txt")						
		# rho_choi_full = self.exp.est_choi(self.hist_a, self.hist_e)
		# print("Target environment:\n",rho_choi_full)

		# kg = self.DeltaDM(rho_choi_full,rho_choi)
		# print("Remaining Knowledge Gap:",kg)

		return

	def test(self):

		# Codes for unit tests
		'''
		print(self.genes)
		print(self.neighbours)
		print(self.A)
		print(self.E)
		print((self.E[0],self.E[5]),self.Delta(self.E[0],self.E[5]))

		data = np.random.randint(low=0, high=2, dtype=int, size=21)
		print(self.L_est(data))
		print(self.c_est(data))

		self.t_f = 2
		self.hist_rho = np.random.randint(low=0, high=2, dtype=int, size=21)
		print(self.policy())
		'''

		# self.run()

		return