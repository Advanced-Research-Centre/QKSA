class agent:

	R_D = 0		# reward threshold for death. If R_t < R_D the agent halts (dies).
	R_R = 0		# reward threshold for reproduction. If R_D < R_t < R_R, the agent self-replicates with some stochastic mutation in its hyper-parameters like t_p,m,A,Lambda_U,gamma,R_D,R_R.
						
	R_t = 0		# cumulative discounted return at time step t.
					
	t_p = 0 	# number of time steps in the past that is considered by the agent at each point in time.
	t_f = 0		# number of time steps that the agent predicts in the future.
					
	n	= 0 	# alphabet size of the UTM that the agent uses for modeling.
	m 	= 0		# state size of the UTM that the agent uses for modeling.
					
	a_t = 0		# action performed by the agent at time step t.
	A = []		# action space.
					
	e_t = 0		# perception recorded by the agent at time step t.
	E = []		# percept space.
					
	nu = 0		# a model of the environment.
	w_nu = 0	# weight assigned to a model based on EAIT metric.

	rho_t = 0	# prediction of the perception e_t made at time step t-1.
	pi_t = 0	# policy for choosing an optimal action and maximizing return at time step t.
	gamma_t = 0	# reward discount that is proportional to the time span between the reward step and the current time step.
	r_t = 0		# reward at time step t based on a distance measure.

	h_t = []	# total history of actions, predictions and perceptions in the time window t-t_p to t-1

	def __init__(self, genes):
		
		# Hardcoded Genes
		self.n = 2			# binary alphabet
		self.m = 5			# based on ACSS specification for BDM
		self.A = ['000']
		
		# Mutable Genes
		self.R_D = genes[0]
		self.R_R = genes[1]
		self.E = genes[2]
		
		print("Agent Alive and Ticking")
	
	def act(self):
	
		self.a_t = self.A[0]
		# self.h_t.append(self.a_t)	# delete old history
	
	def predict(self):
	
		self.rho_t = self.E[0]
		# self.h_t.append(self.rho_t) # delete old history
		
	def perceive(self):
	
		self.e_t = input("\nPerceive Environment: ")
		# check if e_t is a member of the set E, else raise error
	
	def Delta(self):
	
		# HammingDist(self.e_t, self.rho_t)
		reward = len(self.e_t)
		for (i,j) in zip(self.e_t, self.rho_t):
			if i!=j:
				reward -= 1
		self.r_t = reward

	def Return(self):
	
		self.R_t = self.r_t

	def Lambda(self, nu):
	
		# function of runtime, working memory, prediction accuracy and program length respectively, defined for the specific UTM (ULBA).
		self.w_nu = nu_c * nu_s * nu_a * nu_l
		
	def runStep(self):
	
		self.act()
		self.predict()
		self.perceive()
		self.Delta()
		self.Return()
		print("Life goes on... a step at a time")
		return [self.R_t < self.R_D, self.R_t < self.R_R] 