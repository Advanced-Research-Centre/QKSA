qksa = '\n\n\
###########################################################################################################\n\
                                                                                                           \n\
        QQQQQQQQQ        KKKKKKKKK    KKKKKKKK       SSSSSSSSSSSSS                     ____                         \n\
     QQ:::::::::QQ       K:::::::K    K:::::KK     SS::::::::::::::S                 / \\   \\                      \n\
   QQ:::::::::::::QQ     K:::::::K    K::::::K    S:::::SSSSSS::::::S               /   \\___\\                     \n\
  Q:::::::QQQ:::::::Q    K:::::::K   K::::::K     S:::::S     SSSSSSS              _\\   /   /__                    \n\
  Q::::::O   Q::::::Q    KK::::::K  K:::::KK      SS::::S                        / \\ \\ /_/ \\   \\                \n\
  Q:::::O     Q:::::Q    K::::::K K:::::KK          SS::::::SS                  /   \\___/   \\___\\                \n\
  Q:::::O     Q:::::Q    K::::::K K:::::K            SSS:::::::SS              _\\   /   \\   /   /__               \n\
  Q:::::O     Q:::::Q    K::::::K:::::K               SSSSSS::::SS           / \\ \\ /___/ \\ /_/ \\   \\           \n\
  Q:::::O  QQQQ:::::Q    K::::::KK:::::KK                  S:::::SS         /   \\___\\       /   \\___\\           \n\
  Q::::::O Q::::::::Q    KK::::::K  K:::::KK                 S:::::S       _\\   /   /__     \\__ /  _/__           \n\
  Q:::::::QQ::::::::Q    K:::::::K   K::::::K    SSSSSSS    SS:::::S     / \\ \\ /___/   \\ / \\   \\ / \\   \\     \n\
   QQ::::::::::::::Q     K:::::::K    K:::::K    S::::::SSSSSS:::::S    /   \\___\\   \\___\\   \\___\\   \\___\\   \n\
     QQ:::::::::::Q      K:::::::K    K:::::KK    S:::::::::::::::SS    \\   /   /   /   /   /   /   /   /          \n\
      QQQQQQQQ::::QQ     KKKKKKKKK    KKKKKKKK     SSSSSSSSSSSSSSS       \\ /___/ \\ /___/ \\ /___/ \\ /___/        \n\
               Q:::::Q                                                                                              \n\
                QQQQQQ                                                                  \u00a9 Aritra Sarkar        \n\
                                                                                                           \n\
###########################################################################################################\n\n'    

print(qksa)

''' 
===========================================================================================================
Create Quantum Process Environment 
===========================================================================================================
'''

from src.environment import environment

num_qb = int(input("===> Specify number of qubits [Default: 1]: ") or "1")
env = environment(num_qb)
env.createEnv()

''' 
===========================================================================================================
Create Seed QKSA genome 
===========================================================================================================
'''

# Mutable hyper-parameters genes

c_gene		= 'F0F0F0V0V1V2F0V3V4' 				# Initial Seed AI simple cost function. Addition of 5 LEAST estimates.
wt_gene		= [0.0001, 0, 0.1, 0.00005, 1000]	# Weight assigned to LEAST metrics in current c_function.
l_max		= 10000				
e_max		= 0									# Currently not considered
a_max		= 10
s_max		= 20000	
t_max		= 5				

m_c			= 0.18

# Immutable hyper-parameters genes

neighbours 	= list(range(0,num_qb))				# Qubit ids of neighbours (currently full environment is visible to every agent)

t_p			= 16384								# Number of time steps in the past considered by the agent at each point in time.
t_f			= 1									# Number of time steps the agent predicts in the future. Single step	
gamma		= 0.00								# Reward discount that is proportional to the time span between the reward step and the current time step. Linear function
R_R			= 0									# Reward threshold for reproduction. If R_t < R_R, the agent self-replicates with mutation in genes
R_D			= 0									# Reward threshold for death. If R_t < R_D the agent halts (dies).
lifespan	= 1024								# Max age of agent before death

genes = [c_gene, wt_gene, l_max, e_max, a_max, s_max, t_max, m_c, neighbours, t_p, t_f, gamma, R_R, R_D, lifespan]

''' 
===========================================================================================================
Run Quine Hypervisor 
===========================================================================================================
'''

# Initial versions of Quine Hypervisor code developed as part of QWorld QIntern 2021
# Developer: Dhanvanth Balakrishnan & Aritra Sarkar
# https://gitlab.com/qworld/qresearch/qintern2021/QKSA_QIntern/-/blob/master/SP04_alife/QIntern04/hypervisor.py


from collections import deque
import importlib

biosphere = []				# Agents currently running
agt_waitlist = deque()		# Queue containing agents that are created but yet to be executed
run_log = []				# Agents which have already executed and died
aborted = []
abort = 0					# Flag to terminate all processes
max_thread = 1				# How many agents can be handled in parallel by the hypervisor threads
max_queue = 2

agt_waitlist.append('agent_0')

while (len(biosphere) + len(agt_waitlist) > 0):									# No agents alive, every agent serviced!

	if abort != 2:
		print("\nHypervisor status --- \n\tRunning\t\t:",biosphere,"\n\tWaitlist\t:",list(agt_waitlist),"\n\tDead\t\t:",run_log)
		abort = int(input("===> 0: Continue  1: Abort  2: Auto [Default: 0]: ") or 0)
		
	if abort == 1:																# User gets to choose each cycle (world clock tick) to abort or continue
		for agt in biosphere:
			print("Aborting agent:",agt[0])
			agt[1].halt()  									            		# Run one perception cycle for the agent
			aborted.append(agt)
			biosphere.remove(agt)		
		break
	
	if len(biosphere) < max_thread and len(agt_waitlist) > 0:					# If biosphere can support more agents, bring one alive from the waitlist
		agtName = agt_waitlist.popleft()
		agtClass = getattr(importlib.import_module('src.'+agtName), "agent")	# Filename changes for each quine while class name remains same
		agtObj = agtClass(agtName, env, genes)									# Agent gets instantiated here
		biosphere.append([agtName,agtObj])
		
	for agt in biosphere:
		if abort != 2:
			print("Running agent:",agt[0])
		agt[1].run()  									                		# Run one perception cycle for the agent
		if agt[1].alive == False:												# If enough hazard has been encountered
			run_log.append(agt)
			biosphere.remove(agt)
		if agt[1].newChildName != '':											# If agent reproduced in this cycle add child to the waitlist
			if len(agt_waitlist) < max_queue:
				agt_waitlist.append(agt[1].newChildName)
			else:
				print("Queue full, child not queued")

''' 
===========================================================================================================
Visualize Results 
===========================================================================================================
'''

print("\nFinal status --- \n\tRunning\t\t:",biosphere,"\n\tWaitlist\t:",list(agt_waitlist),"\n\tDead\t\t:",run_log,"\n\tAborted\t\t:",aborted)

import matplotlib.pyplot as plt

for agt in run_log:
	ax1 = plt.subplot(2,1,1)
	plt.plot(list(agt[1].LOG_TEST_1))
	plt.ylabel('utility difference')
	plt.ylim(-1.5,1.5)

	ax1 = plt.subplot(2,1,2, sharex=ax1)
	plt.plot(list(agt[1].LOG_TEST_2))
	plt.ylabel('predicted utility')
	plt.ylim(-0.5,1.5)
	plt.show()
	break