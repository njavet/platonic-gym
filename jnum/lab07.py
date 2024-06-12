import sympy as sp
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import fmin, leastsq
from chap6 import gauss_newton_d


sp.init_printing()

x = np.array([0.1, 
              0.3, 
              0.7, 
              1.2, 
              1.6, 
              2.2, 
              2.7, 
              3.1, 
              3.5, 
              3.9], dtype=np.float64)

y = np.array([0.558, 
              0.569, 
              0.176, 
              -0.207, 
              -0.133, 
              0.132, 
              0.055, 
              -0.090, 
              -0.069, 
              0.027], dtype=np.float64)


lam0 = np.array([1, 2, 2, 1], dtype=np.float64)
lam1 = np.array([2, 2, 2, 2], dtype=np.float64)
p = sp.symbols('p:{n:d}'.format(n=lam0.size))

def f(t, p):
    return p[0] * sp.exp(-p[1]*t) * sp.sin(p[2]*t + p[3])

g = sp.Matrix([y[i] - f(x[i], p) for i in range(len(x))])
Dg = g.jacobian(p)
g = sp.lambdify([p], g, 'numpy')
Dg = sp.lambdify([p], Dg, 'numpy')

lam, _ = gauss_newton_d(g, Dg, lam0, damped=False)
lam_2, _ = gauss_newton_d(g, Dg, lam1, damped=False)
t = sp.symbols('t')
F = f(t, lam)
F = sp.lambdify([t], F, 'numpy')
F2 = f(t, lam_2)
F2 = sp.lambdify([t], F2, 'numpy')

lam_d, _ = gauss_newton_d(g, Dg, lam0)
lam_d_2, _ = gauss_newton_d(g, Dg, lam1)
H = f(t, lam_d)
H = sp.lambdify([t], H, 'numpy')
H2 = f(t, lam_d_2)
H2 = sp.lambdify([t], H2, 'numpy')

fig = plt.figure(figsize=(32, 16))
xs = np.linspace(np.min(x) - 1, np.max(x) + 1, 1000)
ax1 = fig.add_subplot(121)
ax1.scatter(x, y, color='red', edgecolor='black', zorder=5, label='data points')
ax1.plot(xs, F(xs), color='orange', label='undamped function')
ax1.plot(xs, H(xs), color='cyan', label='damped function')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.scatter(x, y, color='red', edgecolor='black', zorder=5, label='data points')
ax2.plot(xs, F2(xs), color='orange', label='undamped function')
ax2.plot(xs, H2(xs), color='cyan', label='damped function')
ax2.legend()

# plt.show()
def err_func(x):
    return np.linalg.norm(g(x)) ** 2


xopt = fmin(err_func, np.array([2, 2, 2, 2]))


x = np.array([2., 
              2.5, 
              3., 
              3.5, 
              4., 
              4.5, 
              5., 
              5.5, 
              6., 
              6.5, 
              7., 
              7.5, 
              8.,
              8.5, 
              9., 
              9.5])

y = np.array([159.57209984, 
              159.8851819 , 
              159.89378952, 
              160.30305273,
              160.84630757, 
              160.94703969, 
              161.56961845, 
              162.31468058,
              162.32140561, 
              162.88880047, 
              163.53234609, 
              163.85817086,
              163.55339958, 
              163.86393263, 
              163.90535931, 
              163.44385491])


lam0 = np.array([100, 120, 3, -1], dtype=np.float64)
p = sp.symbols('p:{n:d}'.format(n=lam0.size))
xi = sp.symbols('xi')

def f(x, p):
    a = p[0] + p[1]*10**(p[2] + p[3]*x) 
    b = 1 + 10**(p[2] + p[3]*x)
    return a / b


g = sp.Matrix([y[i] - f(x[i], p) for i in range(len(x))])
Dg = g.jacobian(p)
g = sp.lambdify([p], g, 'numpy')
Dg = sp.lambdify([p], Dg, 'numpy')

lam, _ = gauss_newton_d(g, Dg, lam0)
F = f(xi, lam)
F = sp.lambdify([xi], F, 'numpy')

xs = np.linspace(np.min(x) - 1, np.max(x) + 1, 1000)

plt.clf()
plt.scatter(x, y)
plt.plot(xs, F(xs))
plt.show()


def error_func(lambdas, x, y):
    return f(x, lambdas) - y


start_vector = [100, 120, 3, -1]
result = leastsq(error_func, start_vector, args=(x, y))
# undamped
result2 = leastsq(error_func, start_vector, args=(x, y), ftol=0)

fit_params = result[0]
fit_params2 = result2[0]

plt.plot(x, f(x, fit_params), color='red', label='damped')
plt.plot(x, f(x, fit_params2), color='blue', label='undamped')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 1b)
# for this problem the undamped method also converges 

# 1c)
plt.clf()
fit_params_fmin = fmin(lambda lambdas: np.sum(error_func(lambdas, x, y)**2),
                       start_vector)

plt.scatter(x, y, label='Data')
plt.plot(x, f(x, fit_params), color='red', label='damped')
plt.plot(x, f(x, fit_params_fmin), color='blue', label='fmin')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print('damped params:', fit_params)
print('fmin params:', fit_params_fmin)

# the solutions are nearly the same:
# damped params: [163.88257136 159.47424249   2.172233    -0.42934182]
# fmin params: [163.88256553 159.47427156   2.17225694  -0.4293443 ]

