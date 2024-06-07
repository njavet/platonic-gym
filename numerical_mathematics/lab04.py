import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


x_points = np.array([0, 2500., 5000, 1e5])
y_points = np.array([1013., 747., 540., 226.])


x, x0, x1, x2, x3 = sp.symbols('x x0 x1 x2 x3')

sp.init_printing()
l0 = ((x - x1) * (x - x2) * (x - x3)) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
print(l0.subs([(x0, 0), (x1, 2500), (x2, 5000), (x3, 1e5)]).expand())


