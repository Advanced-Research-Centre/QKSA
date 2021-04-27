'''
	##############
	# How to run #
	##############

		# from environment import environment
		# env = environment("env.qasm")
		
		# from agent import agent
		# genes = [...] 							# add initial value of genes
		# agt = agent(env,genes)

		# agt.run()

'''

import numpy as np
from pybdm import BDM
import pandas
from math import pi, floor

class agent:

	genes = []

	env = []
	neighbours = []	# qubit ids of neighbours
	boundary = 0	# number of neighbours

	c_gene = ""
	wt_gene = []

	R_D = 0			# reward threshold for death. If R_t < R_D the agent halts (dies).
	R_R = 0			# reward threshold for reproduction. If R_D < R_t < R_R, the agent self-replicates with some stochastic mutation in its hyper-parameters like t_p,m,A,Lambda_U,gamma,R_D,R_R.
						
	R_t = 0			# cumulative discounted return at time step t.
						
	t_p = 0 		# number of time steps in the past that is considered by the agent at each point in time.
	t_f = 0			# number of time steps that the agent predicts in the future.
						
	n	= 0 		# alphabet size of the UTM that the agent uses for modeling.
	m 	= 0			# state size of the UTM that the agent uses for modeling.
						
	a_t = 0			# action performed by the agent at time step t.
	A = []			# action space.
						
	e_t = 0			# perception recorded by the agent at time step t.
	E = []			# percept space. all bitstrings of the same size as the number of entries in neighbours
						
	nu = 0			# a model of the environment.
	w_nu = 0		# weight assigned to a model based on EAIT metric.
	
	rho_t = 0		# prediction of the perception e_t made at time step t-1.
	pi_t = 0		# policy for choosing an optimal action and maximizing return at time step t.
	gamma_t = 0		# reward discount that is proportional to the time span between the reward step and the current time step.
	r_t = 0			# reward at time step t based on a distance measure.
	
	h_t = []		# total history of actions, predictions and perceptions in the time window t-t_p to t-1
		
	t = 0			# age of agent in number of runStep calls
	lifespan = 0	# max age of agent before death
	
	trials = 20		# number of tomographic trials in growing phase
	history = []
	
	def __init__(self, env, *genes):
				
		self.env = env
		self.genes = genes

		self.neighbours = genes[0]
		self.boundary = len(self.neighbours)
		self.history = np.empty((pow(3,self.boundary),self.trials),dtype='object')

		self.c_gene		= "F0F0F0V0V1V2F0V3V4" 		# genes[1] 	# Initial Seed AI simple cost function
		self.wt_gene	= [1, 0, 0, 0, 0]			# genes[2]	# Consider only program length for now

		# Initialize (optionally) mutable hyper-parameters
		self.l_max		= genes[3]	
		self.e_max		= 0							# genes[4]	# Currently not considered
		self.a_max		= 0							# genes[5]	# Currently not considered
		self.s_max		= 0							# genes[6]	# Currently not considered
		self.t_max		= genes[7]	
		self.m_c		= genes[8]	
		self.m			= 2							# genes[9]	# binary alphabet
		self.n			= 5							# genes[10]	# based on ACSS specification for BDM
		self.s_c		= 0							# genes[11]	# Currently not considered	
		self.t_p		= genes[12]	
		self.t_f		= 1							# genes[13]	# Single step in the future	
		self.gamma		= genes[14]	
		self.R_D		= genes[15]	
		self.R_R		= genes[16]	
		self.lifespan	= genes[17]	

		self.t		= 0
		self.R_t 	= (self.R_D + self.R_R) / 2
		self.rhist	= []
		self.hist	= []

		print("Agent Alive and Ticking")
	
	def act(self):
		# self.a_t = self.A[0]
		# self.h_t.append(self.a_t)	# delete old history
		basis = self.policy()
		self.env.setBasis(basis)

	def perceive(self):
		self.e_t = self.env.measure(self.neighbours)
		# check if e_t is a member of the set E, else raise error
		if self.age < pow(3,self.boundary)*self.trials: 
			self.history[floor(self.age/self.trials)][self.age%self.trials] = self.e_t[0]

	def L_est(self, data):
		# Function to estimate the program length
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
		for i in range(3**numQb):
			self.A.append(self.DecToBaseN(i,3,numQb))

	def define_E(self, numQb):
		# Define percept space E as all binary strings of numQb qubits
		for i in range(2**numQb):
			self.E.append(self.DecToBaseN(i,2,numQb))
	
	def Delta(self, e_i, e_j):
		# Distance function between elements in percept space
		dist_e_ij = 0
		for (i,j) in zip(e_i,e_j):
			if i!=j:
				dist_e_ij += 1					# Hamming distance
		return dist_e_ij

	def policy(self):

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

	def calcReturn(self):
		# \item[] $r_t=-\Delta(\rho^*_t,e_t)$
        # \item[] hist.\textit{append}$(a^*_t,e_t)$
        # \item[] $r_{hist}.$\textit{append}$(r_t)$
        # \item[] $R_t = 0$ 
        # \item[] for $i$ in range$(t_{min},t)$:
        # \begin{itemize}[noitemsep,nolistsep]
        #     \item[] $R_t += \gamma_i * r_{hist}[i]$
        # \end{itemize}
		return

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
			if (self.R_D < self.R_t):	# Halt agent (die)
				break					
			self.policy()
			self.act()
			self.perceive()
			self.calcReturn()
			if (self.R_t < self.R_R):	# Reproduce
				genes_new = self.mutate(genes)
				self.constructor(genes_new)
			self.t += 1
				
##############
# Unit tests #
##############

# from environment import environment

# env = environment("env.qasm")

# agt = agent(env,(0,1),30,100,0.2,10,0.1,0,0,100)

# data = np.random.randint(low=0, high=2, dtype=int, size=21)
# print(agt.L_est(data))
# print(agt.c_est(data))

# numQb = 3
# agt.define_A(numQb)
# print(agt.A)
# agt.define_E(numQb)
# print(agt.E)
# print((agt.E[0],agt.E[5]),agt.Delta(agt.E[0],agt.E[5]))
