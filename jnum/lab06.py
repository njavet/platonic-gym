import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl


T = np.array([0, 
              10., 
              20., 
              30., 
              40., 
              50., 
              60., 
              70., 
              80., 
              90., 
              100.])

rho = np.array([999.9, 
                999.7, 
                998.2, 
                995.7, 
                992.2, 
                988.1, 
                983.2, 
                977.8, 
                971.8,
                965.3, 
                958.4])


def f(x, c):
    return c[0] + c[1] * x + c[2]*x**2


A = np.vstack([np.ones_like(T), T, T**2]).T
c0 = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, rho))
print('condition of A.T*A', np.linalg.cond(np.dot(A.T, A)))

Q, R = np.linalg.qr(A)
c1 = np.linalg.solve(R, np.dot(Q.T, rho))
print('condition of R', np.linalg.cond(R))

xs = np.linspace(-1, 101)
ys0 = np.array([c0[0] + c0[1]*x + c0[2]*x**2 for x in xs])
ys1 = np.array([c1[0] + c1[1]*x + c1[2]*x**2 for x in xs])

p = np.polyfit(T, rho, 2)
print('coef for c0', c0)
print('coef for c1', c1)
print('coef for p', p)

e0 = np.linalg.norm(rho - f(T, c0), 2)
e1 = np.linalg.norm(rho - f(T, c1), 2)
e2 = np.linalg.norm(rho - np.polyval(p, T), 2)

print('error functional for c0', e0)
print('error functional for c1', e1)
print('error functional for p', e2)

plt.scatter(T, rho, color='red')
plt.plot(xs, ys0, color='green')
plt.plot(xs, ys1, color='blue')
plt.plot(xs, np.polyval(p, xs), color='cyan')
plt.show()


