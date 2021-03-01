def AIXI_tl():
	return "AIXI_tl"
agt = AIXI_tl()
c='def AIXI_tl():\n\treturn "AIXI_tl"\nagt = AIXI_tl()\nc=%r\nprint(c%%c,end="")'
print(c%c,end="")