from pybdm import BDM
import pandas

class least:

	def L_est(self, data):
		# Function to estimate the program length
		if len(data) < 13:
			return 0
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
			ld_db = pandas.read_csv('data/logicalDepthsBinaryStrings.csv',names=['BinaryString', 'LogicalDepth'],dtype={'BinaryString': object,'LogicalDepth': int}) # https://github.com/algorithmicnaturelab/OACC/blob/master/data/logicalDepthsBinaryStrings.csv
			data_str = ""
			for b in data: data_str = data_str+str(b)
			t_est = ld_db[ld_db['BinaryString'].dropna().str.fullmatch(data_str)]['LogicalDepth'].values[0]
		return t_est