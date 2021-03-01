from agent import agent
import inspect

myName = inspect.getframeinfo(inspect.currentframe()).filename
childCtr = 0

def constructor(genes):
	global childCtr
	childCtr += 1
	childName = myName[0:-3]+"c"+str(childCtr)+".py"
	f = open(childName, "w")
	dna='from agent import agent\nimport inspect\n\nmyName = inspect.getframeinfo(inspect.currentframe()).filename\nchildCtr = 0\n\ndef constructor(genes):\n\tglobal childCtr\n\tchildCtr += 1\n\tchildName = myName[0:-3]+"c"+str(childCtr)+".py"\n\tf = open(childName, "w")\n\tdna=%r\n\tf.write(dna%%(dna,genes))\n\tf.close()\n\ndef goedel_machine():\n\tgenes = %r\n\tagt = agent(genes)\n\twhile True:\n\t\t[die, rpd] = agt.runStep()\n\t\tif die:\n\t\t\tprint("Bye bye, cruel world!")\n\t\t\tbreak\n\t\tif rpd:\n\t\t\tprint("Mutants, assemble!")\n\t\t\tgenes[2] = [str(agt.e_t)]\n\t\t\tconstructor(genes)\n\ngoedel_machine()'
	f.write(dna%(dna,genes))
	f.close()

def goedel_machine():
	genes = [1, 2, ['011']]
	agt = agent(genes)
	while True:
		[die, rpd] = agt.runStep()
		if die:
			print("Bye bye, cruel world!")
			break
		if rpd:
			print("Mutants, assemble!")
			genes[2] = [str(agt.e_t)]
			constructor(genes)

goedel_machine()