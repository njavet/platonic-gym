import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import jint


def plot_func(f, a, b, ax=None):
    x_axis = np.arange(a-1, b+1, 0.01)
    y_axis = f(x_axis)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

    # zero axes highlight
    ax.axvline(0, color='black', linewidth=2)
    ax.axhline(0, color='black', linewidth=2)

    ax.grid()
    ax.plot(x_axis, y_axis)
    plt.show()


def plot_func_integral(f, a, b, ax=None):
    x_axis = np.arange(a-1, b+1, 0.01)
    y_axis = f(x_axis)
    if ax is None:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)

    jint.sum_trapezoid(f, a, b, h=1)
    jint.sum_trapezoid(f, a, b, h=2)
    jint.sum_trapezoid(f, a, b, h=4)
    h = 1
    for i in range(4):
        xi = a + i*h
        x1 = a + (i+1)*h
        fxi = f(xi)
        fx1 = f(x1)
        ax.scatter(xi, fxi, color='red', edgecolors='black', zorder=5)
        ax.scatter(x1, fx1, color='red', edgecolors='black', zorder=5)
        ax.plot([xi, xi], [0, fxi], linestyle='-', color='black')
        ax.plot([xi, x1], [fxi, fx1], linestyle='-', color='black')
        ax.plot([x1, x1], [0, fx1], linestyle='-', color='black')

    jint.sum_trapezoid(f, a, b, h=1/8)

    ax.grid()
    ax.plot(x_axis, y_axis)
    plt.show()



def f(x):
    return 6*x**2 - 2*x

plot_func_integral(f, 0, 4)



