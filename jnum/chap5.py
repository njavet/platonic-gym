import numpy as np
import sympy as sp
import matplotlib.pyplot as plt



#### 
# newton method for systems (O(n**2), f 3xcon differentiable, Df regular, x0
# close)
####
# can converge to local minima != 0 => Df not regular
# select start vector close to solution

# 1D 
# f(x1) = f(x0) + f'(x0) * (x1 - x0)
# x1 = x0 - f(x0) / f'(x0)

# multivariate
# f(x1) = f(x0) + Df(x0) * (x1 - x0)
# x1 = x0 - Df(x0).inv * f(x0)

# d0 = -Df(x0).inv * f(x0)
# solve linear system
# Df(x0) * d0 = -f(x0)
# x1 = x0 + d0

def newton_for_systems(f, df, x0, n_iter=None, eps=None):

    for i in range(n_iter):
        print(f'x{i} = {x0}')
        d0 = np.linalg.solve(df(*x0), -f(*x0))
        xn = x0 + d0.reshape(-1)

        # stop criterias
        x_norm = np.linalg.norm(xn - x0, 2)
        cond0 = x_norm <= np.linalg.norm(xn, 2) * eps
        cond1 = x_norm <= eps
        cond2 = np.linalg.norm(f(*xn), 2) <= eps
        if cond0:
            print('cond0 reached, xn = ', xn)
        if cond1:
            print('cond1 reached, xn = ', xn)
        if cond2:
            print('cond2 reached, xn = ', xn)

        x0 = xn

    return xn


def damped_newton(f, df, x0, kmax=4, n_iter=None, eps=None):
    for i in range(n_iter):
        print(f'x{i} = {x0}')

        fnorm = np.linalg.norm(f(*x0), 2)
        d0 = np.linalg.solve(df(*x0), -f(*x0)).reshape(-1)
        xn = x0 + d0
        for k in range(kmax):
            # if no k will be found
            xd = x0 + d0 / (2**k)
            if np.linalg.norm(f(*xd), 2) <= fnorm:
                xn = xd
                break
        x0 = xn
    return xn




x1, x2 = sp.symbols('x1 x2')
f = sp.Matrix([2*x1 + 4*x2, 
               4*x1 + 8*x2**3])
df = f.jacobian([x1, x2])

f = sp.lambdify([x1, x2], f, 'numpy')
df = sp.lambdify([x1, x2], df, 'numpy')

newton_for_systems(f, df, np.array([4, 2]), n_iter=5, eps=1e-4)
damped_newton(f, df, np.array([4, 2]), n_iter=5, eps=1e-4)


