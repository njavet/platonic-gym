
import sympy as sp
import numpy as np
from scipy.interpolate import CubicSpline


# task1 
x, y = sp.symbols('x y')
f = sp.Matrix([x + y - 4,
               x**2 + y**2 - 9])

df = f.jacobian((x, y))
fl = sp.lambdify((x, y), f, 'numpy')
dfl = sp.lambdify((x, y), df, 'numpy')

x0 = np.array([0., 3.])
xn_true = np.array([(4 - np.sqrt(2)) / 2, ((np.sqrt(2) - 4) / 2) + 4])

x_diff = np.linalg.norm(x0 - xn_true, 2)
iterations = 0
eps = 1e-4
while eps <= x_diff:
    d0 = np.linalg.solve(dfl(*x0), -fl(*x0))
    xn = x0 + d0.reshape(-1)
    iterations += 1
    x0 = xn
    x_diff = np.linalg.norm(x0 - xn_true, 2)


# task2 
tk = np.array([0, 0.5, 2, 3])
xk = np.array([1, 2, 2.5, 0])

cs = CubicSpline(tk, xk)


A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0.5, 0, 0, 0.5**2, 0, 0, 0.5**3, 0, 0],
              [0, 1, 0, 0, 1.5, 0, 0, 1.5**2, 0, 0, 1.5**3, 0],
              [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, -1, 0, 2*0.5, 0, 0, 3*0.5**2, 0, 0],
              [0, 0, 0, 1, -1, 0, 2*1.5, 0, 0, 3*1.5**2, 0, 0],
              [0, 0, 0, 0, 0, 0, 2, -2, 0, 6*0.5, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 6*1.5, 0],
              [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 6*1]])

y = np.array([1, 2, 2.5, 2, 2.5, 0, 0, 0, 0, 0, 0, 0])
coefs = np.linalg.solve(A, y)
print(coefs)

ttt = coefs[1] + coefs[4] * (1 - 0.5) + coefs[7] * (1 - 0.5)**2 + coefs[10] * (1-0.5)**3
print('t = 1', ttt)
print('t = 1', cs(1))

