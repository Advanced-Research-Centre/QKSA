import numpy as np
from pybdm import BDM
import pandas


class agent:

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
	E = []			# percept space.
						
	nu = 0			# a model of the environment.
	w_nu = 0		# weight assigned to a model based on EAIT metric.
	
	rho_t = 0		# prediction of the perception e_t made at time step t-1.
	pi_t = 0		# policy for choosing an optimal action and maximizing return at time step t.
	gamma_t = 0		# reward discount that is proportional to the time span between the reward step and the current time step.
	r_t = 0			# reward at time step t based on a distance measure.
	
	h_t = []		# total history of actions, predictions and perceptions in the time window t-t_p to t-1
		
	age = 0			# age of agent in number of runStep calls
	lifeSpan = 0	# max age of agent before death
	
	trials = 20		# number of tomographic trials in growing phase
	
	env = []

	def __init__(self, genes, env):
		
		# Hardcoded Genes
		self.n = 2			# binary alphabet
		self.m = 5			# based on ACSS specification for BDM
		self.A = ['000']
		
		# Mutable Genes
		self.R_D = genes[0]
		self.R_R = genes[1]
		self.E = genes[2]
		self.lifeSpan = genes[3]
		
		self.env = env
		
		self.R_t = self.R_R # for testing
		
		print("Agent Alive and Ticking")

	def policy(step, percept):

		global neighbours, trials
		reqdTrials = pow(3,neighbours)
		basis = [0] * neighbours
		# there is always a first time
		if percept == []: 
			global history
			history = np.empty((reqdTrials,trials),dtype='object')
		# sequentially try out all Pauli basis combinations of neighbouring qubits for 10 cycles
		elif step < reqdTrials*trials:
			pb = np.base_repr(floor(step/trials), base=3, padding=neighbours)
			pb = pb[len(pb)-neighbours:]
			for i in range(0,len(pb)):
				basis[i] = int(pb[i])
		# select winning policy as the one with the lowest K-complexity of the percept history 
		else:
			minKC = 10000
			for i in range(0,reqdTrials):
				PBhist = ''.join(history[i])
				data = np.array(list(PBhist)).astype(np.int)
				aprxKC = BDM(ndim=1)
				nu_l = aprxKC.bdm(data)
				print(PBhist,nu_l)
				if nu_l < minKC:
					minKC = nu_l
					basis_best = i
			pb = np.base_repr(basis_best, base=3, padding=neighbours)
			pb = pb[len(pb)-neighbours:]
			for i in range(0,len(pb)):
				basis[i] = int(pb[i])
			print("Best basis : ",basis)
		return basis
	
	def act(self):
	
		# self.a_t = self.A[0]
		# self.h_t.append(self.a_t)	# delete old history
		basis = policy(step, percept)
		env.setBasis(basis)
	
	def predict(self):
	
		# self.rho_t = self.E[0]
		# self.h_t.append(self.rho_t) # delete old history
		
	def perceive(self):
	
		# self.e_t = input("\nPerceive Environment: ")
		# check if e_t is a member of the set E, else raise error
		percept = env.measure(neighbours)
		if step < pow(3,neighbours)*trials: 
			history[floor(step/trials)][step%trials] = percept[0]
	
	def Delta(self):
	
		# HammingDist(self.e_t, self.rho_t)
		reward = len(self.e_t)
		for (i,j) in zip(self.e_t, self.rho_t):
			if i!=j:
				reward -= 1
		self.r_t = reward

	def Return(self):
	
		self.R_t = self.r_t

	def Lambda(self):
	
		# function of runtime, working memory, prediction accuracy and program length respectively, defined for the specific UTM (ULBA).
		# self.w_nu = nu_c * nu_s * nu_a * nu_l
		
		# Create dataset, initialize 1-dim BDM object (at least of length 12), compute BDM and LD, predict based on lowest BDM
		h_e = np.random.randint(low=0, high=2, dtype=int, size=11)
		aprxKC = BDM(ndim=1)
		minKC = 10000
		ld_db = pandas.read_csv('logicalDepthsBinaryStrings.csv',names=['BinaryString', 'LogicalDepth'],dtype={'BinaryString': object,'LogicalDepth': int}) # https://github.com/algorithmicnaturelab/OACC/blob/master/data/logicalDepthsBinaryStrings.csv
		for i in self.E:
			data = np.append(h_e, i)
			nu_l = aprxKC.bdm(data)
			if nu_l < minKC:
				minKC = nu_l
				self.rho_t = i
			data_str = ""
			for b in data: data_str = data_str+str(b)
			ld = ld_db[ld_db['BinaryString'].dropna().str.fullmatch(data_str)]['LogicalDepth'].values[0]
			print(data,"BDM:",nu_l,"LD:",ld)
		print("Prediction:",self.rho_t)
		
	def runStep(self):
						
		print("Life goes on... a step at a time ",self.age)
		
		self.act()
		# self.predict()
		self.perceive()
		# self.Delta()
		# self.Return()
		
		self.age += 1
		if self.age == self.lifeSpan:
			return [True, self.R_t < self.R_R]
		return [self.R_t < self.R_D, self.R_t < self.R_R] 