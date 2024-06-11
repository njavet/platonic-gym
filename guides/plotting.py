
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x1, x2 = sp.symbols('x1 x2')

f = sp.Matrix([x1**2 + x2 - 11, 
               x1 + x2**2 -7])

f = sp.lambdify((x1, x2), f)
xs = np.linspace(-8, 8)
ys = f(xs)

plt.plot(xs, ys)


