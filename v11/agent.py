import numpy as np
from pybdm import BDM
import pandas
from math import pi, floor
from numpy_ringbuffer import RingBuffer		# pip install numpy_ringbuffer
import random
import copy

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
		self.hist_a		= RingBuffer(capacity=self.t_p, dtype='U'+str(self.boundary))		# History of actions
		self.hist_rho	= RingBuffer(capacity=self.t_p, dtype='U'+str(self.boundary))		# History of predictions
		self.hist_e		= RingBuffer(capacity=self.t_p, dtype='U'+str(self.boundary))		# History of perceptions
		self.hist_r		= RingBuffer(capacity=self.t_p)										# History of rewards

		print("Agent Alive and Ticking")
			
	def act(self, a_t_star):
		self.env.setBasis(a_t_star)
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
			ld_db = pandas.read_csv('logicalDepthsBinaryStrings.csv',names=['BinaryString', 'LogicalDepth'],dtype={'BinaryString': object,'LogicalDepth': int}) # https://github.com/algorithmicnaturelab/OACC/blob/master/data/logicalDepthsBinaryStrings.csv
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

	def policyLegacy(self):

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

	def policy(self):
		a_t = self.A[0]
		a_t_star = self.A[0]				# Optimal action for the agent determined by the policy at time step t	
		rho_t = self.E[0]
		rho_t_star = self.E[0]				# Prediction of the perception e_t made at time step t
		c_est_star = 111110					# some large number

		def futureCone(t_fc, past_a, past_rho):
			nonlocal a_t, a_t_star, rho_t, rho_t_star, c_est_star
			if t_fc < self.t+self.t_f:
				for a in self.A: 			#[0:4]:		# for testing
					past_a_new = copy.deepcopy(past_a)
					past_a_new.append(a)
					if t_fc == self.t:
						a_t = a
					for rho in self.E:		#[0:4]: 	# for testing
						if t_fc == self.t:
							rho_t = rho
						past_rho_new = copy.deepcopy(past_rho) 
						past_rho_new.append(rho)
						futureCone(t_fc+1, past_a_new, past_rho_new)
			else:
				data = ''.join(map(str, past_rho))	# currently only rho considered
				cost = self.c_est(np.array(list(data), dtype=int))
				# print("Past_a :",list(past_a),"Past_rho :",list(past_rho),"c_est :",cost)
				if (cost >= 0) and (cost < c_est_star):
						c_est_star =  cost
						a_t_star = a_t
						rho_t_star = rho_t

		futureCone(self.t, list(self.hist_a), list(self.hist_rho))
		
		return [a_t_star, rho_t_star]

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
			if (self.R_t < self.R_D):	# Halt agent (die)
				print("The input and program conspired against my eternal life")
				return
			self.t_p_max = self.t if (self.t-self.t_p) < 0 else self.t_p	 # How much historic data is available (adjusted for initial few steps)

			[a_t_star, rho_t_star] = self.policy()
			
			self.hist_a.append(a_t_star)
			self.act(a_t_star)

			self.hist_rho.append(rho_t_star)

			e_t = self.perceive()				# Perception recorded by the agent at time step t.
			self.hist_e.append(e_t)
			
			r_t = self.Delta(rho_t_star, e_t)	# Reward based on Hamming distance between perception and prediction
			self.hist_r.append(r_t)
			
			self.R_t = self.calcReturn(self.hist_r)

			print("Age =",self.t,"\t--> Action :",a_t_star,"Prediction :",rho_t_star,"Perception :",e_t,"Reward :",r_t,"Return :",self.R_t)

			if (self.R_t < self.R_R):	# Reproduce
				genes_new = self.mutate(genes)
				self.constructor(genes_new)
			self.t += 1
		print("Lived life to the fullest")
		return

	def test(self):
		# Codes for unit tests

		# print(self.genes)
		# print(self.neighbours)
		# print(self.A)
		# print(self.E)
		# print((self.E[0],self.E[5]),self.Delta(self.E[0],self.E[5]))

		# data = np.random.randint(low=0, high=2, dtype=int, size=21)
		# print(self.L_est(data))
		# print(self.c_est(data))

		# self.t_f = 2
		# self.hist_rho = np.random.randint(low=0, high=2, dtype=int, size=21)
		# print(self.policy())

		self.run()

		return
	
##############
# How to run #
##############

from environment import environment
from agent import agent

if __name__ == "__main__":
	
	env = environment("env.qasm")

	neighbours 	= [0, 1, 2]					# Qubit ids of neighbours
	c_gene		= "F0F0F0V0V1V2F0V3V4" 		# Initial Seed AI simple cost function
	wt_gene		= [1, 0, 0, 0, 0]			# Weight assigned to EAIT metrics in current LEAST c_function. Consider only program length for now
	l_max		= 120	
	e_max		= 0							# Currently not considered
	a_max		= 0							# Currently not considered
	s_max		= 0							# Currently not considered
	t_max		= 120	
	m_c			= 0.18
	m			= 2							# Alphabet size of the UTM that the agent uses for modeling. Binary
	n			= 5							# State size of the UTM that the agent uses for modeling. Based on ACSS specification for BDM
	s_c			= 0							# Currently not considered	
	t_p			= 10						# Number of time steps in the past considered by the agent at each point in time.
	t_f			= 1							# Number of time steps the agent predicts in the future. Single step	
	gamma		= 0.05						# Reward discount that is proportional to the time span between the reward step and the current time step. Linear function
	R_D			= 0							# Reward threshold for death. If R_t < R_D the agent halts (dies).
	R_R			= 0							# Reward threshold for reproduction. If R_D < R_t < R_R, the agent self-replicates with mutation in genes
	lifespan	= 20						# Max age of agent before death

	genes = [neighbours, c_gene, wt_gene, l_max, e_max, a_max, s_max, t_max, m_c, m, n, s_c, t_p, t_f, gamma, R_D, R_R, lifespan]
	agt = agent(env, genes)

	agt.test()