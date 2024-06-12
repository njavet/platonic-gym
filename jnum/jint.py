import numpy as np
import sympy as sp


def sum_midpoint(f, a, b, h=1):
    n = int((b - a) / h)
    
    res = 0
    for i in range(n):
        res += f(a + i*h + h/2)
    return h * res


def trapezoid_points(f, a, b, h):
    n = int((b - a) / h)
    tmp = (f(a) + f(b) / 2)
    for i in range(1, n):
        xi = a + i*h
        tmp += f(xi)

        
def sum_trapezoid(f, a, b, h=1):
    n = int((b - a) / h)

    tmp = (f(a) + f(b) / 2)
    for i in range(1, n):
        xi = a + i*h
        tmp += f(xi)
    return h * tmp


def sum_simpson(f, a, b, h=1):
    n = int((b - a) / h)

    res = 0.5*f(a) + 0.5*f(b)

    for i in range(1, n):
        res += f(a + i*h)

    for i in range(1, n+1):
        x0 = a + (i-1)*h
        xi = a + i*h
        res += 2*f((x0 + xi) / 2)
    return (h/3) * res


def romberg(f, a, b, m):
    T = {}
    for j in range(m+1):
        T[j, 0] = sum_trapezoid(f, a, b, 2**j)

    for k in range(1, m+1):
        for j in range(0, m+1-k):
            T[j, k] = ((4**k) * T[j+1, k-1] - T[j, k-1]) / (4**k - 1)

    return T


T = romberg(lambda x: 6*x**2 - 2*x, 0, 4, 2)
for k, v in T.items():
    print(f'k = {k}, v = {v}')


print(sum_simpson(lambda x: 6*x**2 - 2*x, 0, 4, 8))
print(sum_simpson(lambda x: 6*x**2 - 2*x, 0, 4, 10))


