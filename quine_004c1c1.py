import inspect

myName = inspect.getframeinfo(inspect.currentframe()).filename
childCtr = 0
hazardCtr = 0
ageCtr = 0

def constructor(p):
	global childCtr
	childCtr += 1
	childName = myName[0:-3]+"c"+str(childCtr)+".py"
	f = open(childName, "w")
	dna='import inspect\n\nmyName = inspect.getframeinfo(inspect.currentframe()).filename\nchildCtr = 0\nhazardCtr = 0\nageCtr = 0\n\ndef constructor(p):\n\tglobal childCtr\n\tchildCtr += 1\n\tchildName = myName[0:-3]+"c"+str(childCtr)+".py"\n\tf = open(childName, "w")\n\tdna=%r\n\tf.write(dna%%(dna,p))\n\tf.close()\n\ndef goedel_machine():\n\tprediction = str(%r)\n\tglobal ageCtr, childCtr, hazardCtr\n\twhile hazardCtr < 3:\n\t\tageCtr += 1\n\t\tp = input("Perceive Environment: ")\n\t\treward = 3\n\t\tfor (i,j) in zip(prediction,p):\n\t\t\tif i!=j:\n\t\t\t\treward -= 1\n\t\tif reward == 0:\n\t\t\tconstructor(p)\n\t\t\thazardCtr += 1\n\t\tprint(ageCtr,childCtr,hazardCtr)\n\ngoedel_machine()'
	f.write(dna%(dna,p))
	f.close()

def goedel_machine():
	prediction = str('000')
	global ageCtr, childCtr, hazardCtr
	while hazardCtr < 3:
		ageCtr += 1
		p = input("Perceive Environment: ")
		reward = 3
		for (i,j) in zip(prediction,p):
			if i!=j:
				reward -= 1
		if reward == 0:
			constructor(p)
			hazardCtr += 1
		print(ageCtr,childCtr,hazardCtr)

goedel_machine()