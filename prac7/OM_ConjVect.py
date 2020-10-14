import numpy as np
#import numdifftools as nd
from math import sqrt
from scipy.optimize import minimize_scalar

notations = True
dim = 3
#gr = (sqrt(5) - 1)/2

def BrentComb(a, c, func, eps):
	
	"""
	Комбинированный метод Брента минимизации ф-ии одной переменной
	 (применяется для фунцкий f_k(alpha) = J(u_k - alpha*p_k)
	"""
	K = (3 - sqrt(5)) / 2
	x = (a + c)/2
	w = (a + c)/2
	v = (a + c)/2
	u = 2*c
	f_x = func(x)
	f_w = func(w)
	f_v = func(v)
	while abs(u - x) >= eps:
		if ((x != w and w != v and x != v) and (f_x != f_w and f_w != f_v and f_x != f_v)):
			u = x - (((x - v)**2)*(f_x - f_w) - ((x - w)**2)*(f_x - f_v))/(2*((x - v)*(f_x - f_w) - (x - w)*(f_x - f_v)))
		if ((u >= a + eps) and (u <= c - eps) and abs(u - x) < g/2):
			d = abs(u - x)
		else:
			if x < (c - a)/2:
				u = x + K*(c - x) # золотое сечение на [x, c]
				d = c - x
			else:
				u = x - K*(x - a) # золотое сечение на [a, x]
				d = x - a
		if abs(u - x) < eps:
			u = x + eps*np.sign(u - x)
		f_u = func(u)
		if f_u <= f_x:
			if u >= x:
				a = x
			else:
				c = x
			v = w
			w = x
			x = u
			f_v = f_w
			f_w = f_x
			f_x = f_u
		else:
			if u >= x:
				c = u
			else:
				a = u
			if ((f_u <= f_w) or (w == x)):
				v = w
				w = u
				f_v = f_w
				f_w = f_u
			elif ((f_u <= f_v) or (v == x) or (v == w)):
				v = u
				f_v = f_u
	return u, f_u

def GSS(a, b, f, eps):
	"""
	Метод золотого сечения для минимизации функции одной переменной
	(применяется для фунцкий f_k(alpha) = J(u_k - alpha*p_k)
	
	"""
	
	c = b - (b - a) / gr
	d = a + (b - a) / gr
	
	while abs(c - d) > eps:
		if f(c) < f(d):
			b = d
		else:
			a = c

		c = b - (b - a) / gr
		d = a + (b - a) / gr

	return (b + a) / 2
	
def makeF(J, u, alph, p):
	return 
	
def Gradient(func, point):
	grad = []
	h = 0.0001
	for pos, num in enumerate(point):
		dim_f = np.array(point)
		dim_b = np.array(point)
		dim_f[pos] += h
		dim_b[pos] -= h
		grad.append((func(dim_f) - func(dim_b)) / (2 * h))
	return np.array(grad)

	
def conj_vect(J, eps, u_0):
	"""
	Метод сопряженных векторов для минимизации функции многих переменных

	u_k+1 = u_k - alpha_k*p_k
		,где: 
	u_0 - начальное приближение
	p_0 = grad(J)(u_0)
	p_k = grad(J)(u_k) - beta_k*u_(k-1) 
	
	alpha_k >= 0 ; 
	alpha_k = argmin(f_k(alpha) | alpha > 0), 
		where f_k(alpha) = J(u_k - alpha*p_k)
		
	beta_k = - |grad(J)(u_k)|**2/
				|grad(J)(u_k-1)|**2
				
	"""
    
	#grad = nd.Gradient(J)
	u_old = np.array(u_0)
	p = Gradient(J, u_old)
	f = lambda alph: J(u_old - alph*p)
	print(f(1))
	print(f(0.5))
	#alpha = BrentComb(0, 2, f, 0.0001)[0]
	alpha = minimize_scalar(f, bounds=(0, 1), method='bounded').x
	beta = 0.
	if (notations == True):
			print("Шаг 1")
			print("  Alpha | Beta  - ", alpha, " | ", beta)
			print(" Значение аргумента - ", u_old)
			print(" Значение направления - ", p)
	k = 1
	while(np.linalg.norm(Gradient(J, u_old))>= eps):
		if (notations == True):
			print("Шаг ", k)
		u_new = u_old - alpha*p

		if (k % dim == 0):
			beta = 0.
		else:
			beta = - np.linalg.norm(Gradient(J, u_new))**2 / (np.linalg.norm(Gradient(J, u_old))**2)
		p = Gradient(J, u_new) - beta * u_old
		f = lambda alph: J(u_old - alph*p)
		print(f(1))
		#alpha = BrentComb(0, 1, f, 0.0001)[0]
		alpha = minimize_scalar(f, bounds=(0, 1), method='bounded').x
		
		u_old = u_new
		if (notations == True):
			print("  Alpha | Beta  - ", alpha, " | ", beta)
			print(" Значение аргумента - ", u_new)
			print(" Значение направления - ", p)
			print ("Минимум функции: ", J(u_old))
		k += 1
	print ("Результат получен через ", k, "шагов")
	print ("Значение аргумента:", np.array(u_old))
	print ("Минимум функции: ", J(u_old))
	return(u_new)

J =	lambda u: (u[0] + 3)**2 + 3*(u[1] - 1)**2 + 4*(u[2] - 2)**2  - 3
eps = 0.001
u_0 = [4.0, -2.0, 4.0]
	
conj_vect(J, eps, u_0)	
	
	
	
	
	
	
	
	
	
	