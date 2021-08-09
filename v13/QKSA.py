import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
from environment import environment
from agent import agent

# Create Quantum Process Environment
num_qb = int(input("Number of qubits : ") or "1")
env = environment(num_qb)
env.createEnv()

# Create Seed QKSA gene
neighbours 	= list(range(0,num_qb))				    # Qubit ids of neighbours (currently full environment is visible to every agent)
c_gene		= "F0F0F0V0V1V2F0V3V4" 					# Initial Seed AI simple cost function. Addition of 5 LEAST estimates.
wt_gene		= [1, 0, 0, 0, 0]						# Weight assigned to LEAST metrics in current c_function. Consider only program length for now
l_max		= 120				
e_max		= 0										# Currently not considered
a_max		= 0										# Currently not considered
s_max		= 0										# Currently not considered
t_max		= 120				
m_c			= 0.18			
m			= 2										# Alphabet size of the UTM that the agent uses for modeling. Binary
n			= 5										# State size of the UTM that the agent uses for modeling. Based on ACSS specification for BDM
s_c			= 0										# Currently not considered	
t_p			= 16384									# Number of time steps in the past considered by the agent at each point in time.
t_f			= 1										# Number of time steps the agent predicts in the future. Single step	
gamma		= 0.00									# Reward discount that is proportional to the time span between the reward step and the current time step. Linear function
R_D			= 0										# Reward threshold for death. If R_t < R_D the agent halts (dies).
R_R			= 0										# Reward threshold for reproduction. If R_D < R_t < R_R, the agent self-replicates with mutation in genes
lifespan	= 2000									# Max age of agent before death
genes = [neighbours, c_gene, wt_gene, l_max, e_max, a_max, s_max, t_max, m_c, m, n, s_c, t_p, t_f, gamma, R_D, R_R, lifespan]

# Create Seed QKSA
agt = agent(env, genes)

# Run QKSA	
agt.run()