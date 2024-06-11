
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# interpolation
def lagrange_polynomial(x, x_points, y_points):
    n = len(x_points)
    p = 0 
    for i in range(n):
        p += lagrange_basis_li(i, x, x_points) * y_points[i]
    return p


def lagrange_basis_li(i, x, x_points):
    n = len(x_points) 
    li = 1
    for j in range(n):
        if i != j:
            li *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return li
    


xs = np.array([8., 10., 12., 14.])

ys = np.array([11.2, 13.4, 15.3, 19.5])
print(lagrange_polynomial(11, xs, ys))


