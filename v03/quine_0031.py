import inspect
myName = inspect.getframeinfo(inspect.currentframe()).filename
print(myName)
childCtr = 0
childCtr+= 1
childName = myName[0:-3]+str(childCtr)+".py"
print(childName)
f = open(childName, "w")
dna='import inspect\nmyName = inspect.getframeinfo(inspect.currentframe()).filename\nprint(myName)\nchildCtr = 0\nchildCtr+= 1\nchildName = myName[0:-3]+str(childCtr)+".py"\nprint(childName)\nf = open(childName, "w")\ndna=%r\nf.write(dna%%dna)\nf.close()'
f.write(dna%dna)
f.close()