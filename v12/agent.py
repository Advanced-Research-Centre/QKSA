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
		self.env.setBasis(a_t_star) # check compatibility
		return

	def perceive(self):
		return self.E[random.randint(0, 2**self.boundary-1)]	# Test code
		e_t = self.env.measure(self.neighbours)
		# check if e_t is a member of the set E, else raise error
		return e_t

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
		# Legacy code from v10

		basis = [0] * self.boundary			
		reqdTrials = pow(3,self.boundary)
		# measure in all-Z for first time
		if self.age == 0:
			pass
		# sequentially try out all Pauli basis combinations of neighbouring qubits for 10 cycles
		elif self.age < reqdTrials*self.trials:
			pb = np.base_repr(floor(self.age/self.trials), base=3, padding=self.boundary)
			pb = pb[len(pb)-self.boundary:]
			for i in range(0,len(pb)):
				basis[i] = int(pb[i])
		# select winning policy as the one with the lowest K-complexity of the percept history 
		else:
			minLambda = 10000
			action_best = basis
			for i in range(0,reqdTrials):
				PBhist = ''.join(self.history[i])
				data = np.array(list(PBhist)).astype(np.int)
				estLambda = self.Lambda(data)	# TBD: append predicted action
				print(PBhist, estLambda)
				if estLambda < minLambda:
					minLambda = estLambda
					action_best = i
			pb = np.base_repr(action_best, base=3, padding=self.boundary)
			pb = pb[len(pb)-self.boundary:]
			for i in range(0,len(pb)):
				basis[i] = int(pb[i])
			print("Best basis : ",basis)
		return basis
		'''

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
		rho_t_star = self.E[0]				# Prediction of the perception e_t made at time step t
		'''
		Use hist_{a,e} to predict rho_t_star from a_t_star
			if multiplicity of a_t_star is less that trail, return random rho_t_star
			else return rho_t_star based on probability of hist_e for a_t_star
		'''

		# for QPT, a_t_star comprises of a input density matrix and a measurement basis.
		# the rho_t_star is the output density matrix based on the current model

		rho_inp = a_t_star[0]
		process_qubits = 1					# TBD: Generalize to n-qubits

		# Use the estimated Choi matrix to predict the output density matrix 
		rho_out_choi_est = 2**process_qubits * self.partialTrace1( np.matmul( np.kron(np.transpose(rho_inp), np.eye(2**process_qubits) ), rho_choi ))
		print("\nOutput Density Matrix using Estimated Choi Matrix")
		print(rho_out_choi_est)

		return rho_t_star

	def calcReturn(self, hist_r):
		R_t = 0
		for i in range(self.t_p_max-1,-1,-1):
			R_t += hist_r[i] * (1 - self.gamma*(self.t_p_max-i))
		return R_t

	def mutate(self):
		# mutate c_gene, wt_gene and other optional genes
		genes_new = self.genes
		return genes_new

	def constructor(self, genes):
		# add Quine code here
		# offload file execution to OS
		return

	def run(self):
		while (self.t < self.lifespan):
			if (self.R_t < self.R_D):			# Halt agent (die)
				print("The input and program conspired against my eternal life")
				return	
			self.t_p_max = self.t if (self.t-self.t_p) < 0 else self.t_p	 # How much historic data is available (adjusted for initial few steps)

			a_t_star = self.policy()
			self.hist_a.append(a_t_star)
			self.act(a_t_star)

			rho_t_star = self.predict(a_t_star)
			self.hist_rho.append(rho_t_star)

			e_t = self.perceive()				# Perception recorded by the agent at time step t.
			self.hist_e.append(e_t)
			
			r_t = self.Delta(rho_t_star, e_t)	# Reward based on Hamming distance between perception and prediction
			self.hist_r.append(r_t)
			
			self.R_t = self.calcReturn(self.hist_r)

			print("Age =",self.t,"\t--> Action :",a_t_star,"Prediction :",rho_t_star,"Perception :",e_t,"Reward :",r_t,"Return :",self.R_t)

			if (self.R_t < self.R_R):			# Reproduce
				genes_new = self.mutate(genes)
				self.constructor(genes_new)
			self.t += 1
		print("Lived life to the fullest")
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

		# Codes for v12 QPT tests

		# Create Quantum Process Tomography Object and Environment
		p_qpt = qpt(self.env.num_qb)
		qptCirc = p_qpt.aaqpt(self.env.qpCirc)
		env_qpt = environment(p_qpt.num_qb)
		env_qpt.createEnv(qptCirc)
		
		# Boost learning (optional)
		# self.loadHist("data/AAQPT_full.txt")						

		rho_choi_pred = p_qpt.est_choi(self.hist_a, self.hist_e)
		print("Initial estimated environment:\n",rho_choi_pred)
		
		# Given hist_a[rho_in, M] and hist_e[bitstr] for every timestep 0:t-1
		# Select QPT algos that are within c_bound
		# Pass hist_ae to each QPT algo
		# Each QPT returns:
		# (1) least cost estimate 
		# (2) estimated rho_choi
		# (3) action for timestep t
		# Use a_t and rho_choi to predict e_t
		# Use current history and a_t and predicted e_t to make predicted rho_choi for next step
		# Get actual e_t by running the QPT env.
		# Calculate distance between rho_choi and predicted rho_choi_old as the knowledge gain
		
		knowledge = []

		for step in range(0,2000):
			rho_choi = p_qpt.est_choi(self.hist_a, self.hist_e)
			# print("Current estimated environment:\n",rho_choi)

			a_t = p_qpt.policy()
			# print("Chosen action:", a_t)
			self.hist_a.append(a_t)

			pr_qcirc = QuantumCircuit(p_qpt.num_qb)
			prb = list(reversed(a_t[1]))
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
			rho_t = self.toStr(i,2).zfill(p_qpt.num_qb)
			# print("Perception based on rho_choi and a_t:",rho_t)
							
			hist_e_rho = copy.deepcopy(self.hist_e)
			hist_e_rho.append(rho_t)
			rho_choi_pred = p_qpt.est_choi(self.hist_a, hist_e_rho)

			e_t = env_qpt.measure(list(range(0,p_qpt.num_qb)),list(reversed(a_t[1])))
			# print("Perception from environment:",e_t[0])
			self.hist_e.append(e_t[0])

			# This is the reward/utility (knowledge gain), thus, higher the knowledge gain (error in prediction) the better
			# When knowledge gain falls below a limit, learning is finished and QKSA reproduces with mutated cost fn.
			kg = self.DeltaDM(rho_choi_pred,rho_choi)
			# print("Reward/Utility:",kg)
			knowledge.append(kg)

		rho_choi = p_qpt.est_choi(self.hist_a, self.hist_e)
		print("Learnt environment:\n",rho_choi)

		self.loadHist("data/AAQPT_full.txt")						
		rho_choi_full = p_qpt.est_choi(self.hist_a, self.hist_e)
		print("Target environment:\n",rho_choi_full)

		kg = self.DeltaDM(rho_choi_full,rho_choi)
		print("Remaining Knowledge Gap:",kg)

		plt.plot(knowledge)
		plt.ylabel('trace distance')
		plt.show()

		return