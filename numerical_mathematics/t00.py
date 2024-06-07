# interpolate given datapoints
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def lagrange_basis(xsym, i, x_points):
    basis = 1
    for j in range(len(x_points)):
        if j != i:
            basis *= (xsym - x_points[j]) / (x_points[i] - x_points[j])
    return sp.simplify(basis)


def lagrange_interpolation_polynomial(x_points, y_points):
    x = sp.symbols('x')
    p = 0 
    for i, (xi, yi) in enumerate(zip(x_points, y_points)):
        p += lagrange_basis(x, i, x_points) * yi
    p = sp.simplify(p)
    return sp.lambdify(x, p, 'numpy')


if __name__ == '__main__':
    x_points = np.array([1, 2.5, 3, 5, 13, 18, 20])
    y_points = np.array([2, 3, 4, 5, 7, 6, 3.])
    p = lagrange_interpolation_polynomial(x_points, y_points)

    xs = np.arange(0, 21, 0.01)
    ys = p(xs)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.grid()
    ax.plot(xs, ys)
    for xi, yi in zip(x_points, y_points):
        ax.scatter(xi, yi, color='red', edgecolor='black', zorder=5)
        ax.plot([xi, xi], [0, yi], linestyle='--', color='gray')

    ax.axvline(0, color='black', linewidth=2)
    ax.axhline(0, color='black', linewidth=2)

    plt.show()

