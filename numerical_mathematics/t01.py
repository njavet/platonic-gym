# interpolate given datapoints
import numpy as np
import sympy as sp
from scipy import integrate
import matplotlib.pyplot as plt


def integrate_poly(p):
    ip = np.zeros(len(p) + 1)
    for i, coef in enumerate(p):
        deg = i+1
        ip[deg] = (1 / deg) * coef
    return ip


def middlepoint_sum(p, a, b, n):
    h = (b - a) / n
    for i in range(n):
        xi = a + i*h
        res += np.polyval(p, (xi + h/2))
    return res * h


def plot_sums(ax, p, a, b, n):
    ax.grid()
    ax.axvline(0, color='black', linewidth=2)
    ax.axhline(0, color='black', linewidth=2)
    xs = np.arange(0, 8, 0.01)
    ys = np.polyval(p, xs)
    ax.plot(xs, ys)

    h = (b - a) / n
    for i in range(n):
        xi = a + i*h
        fxi = np.polyval(p, xi + h/2)
        # middle point plot
        ax.scatter(xi + h/2, fxi, color='red', edgecolor='black', zorder=5)
        ax.plot([xi +h/2, xi+h/2], [0, fxi], linestyle='--', color='gray')
        # riemann bars
        ax.plot([xi, xi], [0, fxi], linestyle='-', color='black')
        ax.plot([xi, xi + h], [fxi, fxi], linestyle='-', color='black')
        ax.plot([xi+h, xi+h], [0, fxi], linestyle='-', color='black')




if __name__ == '__main__':
    x_points = np.array([1, 2.5, 3, 5, 13, 18, 20])
    y_points = np.array([2, 3, 4, 5, 7, 6, 3.])
    p = np.polyfit(x_points, y_points, deg=6)
    pi = integrate_poly(p)
    # TODO find and correct mistake
    print(f'true int', np.polyval(pi, 5) - np.polyval(pi, 0))
    pi2 = np.polyint(p)
    print(f'true int numpy', np.polyval(pi2, 5) - np.polyval(pi2, 0))


    fig = plt.figure(figsize=(30, 20))
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        plot_sums(ax, p, 0, 5, 2**i)

    plt.show()

