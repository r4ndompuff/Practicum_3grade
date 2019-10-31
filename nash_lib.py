import numpy as np
from fractions import fraction as frc
 
def correct_output(a):
	r, c = a.shape 
	for i in range(r):
		print(end=" | ")
		for j in range(c):
			print(a[i][j], end=" | ")
		print()
	
		
	


mtr_1 = np.array([[4, 0, 6, 2, 2, 1],
                  [3, 8, 4, 10, 4, 4],
				  [1, 2, 6, 5, 0, 0],
				  [6, 6, 4, 4, 10, 3],
				  [10, 4, 6, 4, 0, 9],
				  [10, 7, 0, 7, 9, 8]])
				  
spectre_1 = np.array([0, 0.2281488, 1, 0.13371335])

def spectre_output(s):
	n = s.shape
	
	for i in range(n)
		print(frc(s[i]), end=" | ")
	

correct_output(mtr_1)
