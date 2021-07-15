from agent import agent

def goedel_machine():
	genes = [1, 2, [0,1]]
	agt = agent(genes)
	agt.Lambda()
	#while True:
	#	[die, rpd] = agt.runStep()
	#	if die:
	#		print("Bye bye, cruel world!")
	#		break
	#	if rpd:
	#		print("Mutants, assemble!")
	#		genes[2] = [str(agt.e_t)]
	#		# mutate other genes here

goedel_machine()