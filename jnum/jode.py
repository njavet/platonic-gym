import numpy as np


def euler(f, a, b, n, y0):
    h = (b - a) / n
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0

    for i in range(n):
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h*f(x[i], y[i])

    return x, y


def midpoint(f, a, b, n, y0):
    h = (b - a) / n
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0

    for i in range(n):
        xh2 = x[i] + h/2
        yh2 = y[i] + (h/2)*f(x[i], y[i])
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h*f(xh2, yh2)

    return x, y


def modeuler(f, a, b, n, y0):
    h = (b - a) / n
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0

    for i in range(n):
        x[i+1] = x[i] + h
        yi1 = y[i] + h*f(x[i], y[i])
        k1 = f(x[i], y[i])
        k2 = f(x[i+1], yi1)
        y[i+1] = y[i] + (h/2) * (k1 + k2)

    return x, y


def runge_kutta_k4(f, a, b, n, y0):
    h = (b - a) / n
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0

    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h/2), y[i] + (h/2)*k1)
        k3 = f(x[i] + (h/2), y[i] + (h/2)*k2)
        k4 = f(x[i] + h, y[i] + h*k3)
        x[i+1] = x[i] + h
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    return x, y


