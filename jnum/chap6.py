import numpy as np
from scipy.interpolate import CubicSpline as CS
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


class LagrangePoly:
    def __init__(self, x_points, y_points):
        sp.init_printing()
        assert len(x_points) == len(y_points)
        self.n = len(x_points) 
        self.x_points = x_points
        self.y_points = y_points
        self.sp_poly = 0
        self.sp_vector = np.zeros(self.n)
        self.f_poly = self.construct_sym_poly()

    def construct_sym_poly(self):
        x = sp.symbols('x')
        self.sp_poly = 0
        self.sp_vector = np.zeros(self.n)
        for i in range(self.n):
            li = sp.simplify(self.li(i, x) * self.y_points[i])
            self.sp_poly += li
        return sp.lambdify(x, self.sp_poly, 'numpy')

    def li(self, i, x):
        res = 1
        for j in range(self.n):
            if i != j:
                res *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
        return res

    def eval_poly(self, x):
        p = 0 
        for i in range(self.n):
            p += self.li(i, x) * self.y_points[i]
        return p


class CubicSpline:
    def __init__(self, x_points, y_points):
        assert len(x_points) == len(y_points)
        self.n = len(x_points) - 1
        self.x_points = x_points
        self.y_points = y_points
        self.h = [self.x_points[i + 1] - self.x_points[i] for i in range(self.n)]
        self.c = np.zeros((4, self.n))
        self.compute_coefficients()

    def __call__(self, x):
        for i in range(self.n):
            if self.x_points[i] <= x <= self.x_points[i + 1]:
                y = np.ones(self.n + 1)
                for j in range(self.n, 0, -1):
                    y[j] = (x - self.x_points[i]) ** j
                return np.dot(self.c[:, i], y)

    def compute_cy_i(self, i):
        tmp0 = 3 * (self.y_points[i+1] - self.y_points[i]) / self.h[i]
        tmp1 = 3 * (self.y_points[i] - self.y_points[i-1]) / self.h[i-1]
        return tmp0 - tmp1

    def compute_b_i(self, i):
        tmp0 = (self.y_points[i + 1] - self.y_points[i]) / self.h[i]
        try:
            tmp1 = (self.h[i] / 3) * (self.c[1][i + 1] + 2 * self.c[1][i])
        except IndexError:
            tmp1 = (self.h[i] / 3) * 2 * self.c[1][i]
        return tmp0 - tmp1

    def compute_d_i(self, i):
        try:
            res = 1 / (3 * self.h[i]) * (self.c[1][i + 1] - self.c[1][i])
        except IndexError:
            res = 1 / (3 * self.h[i]) * (- self.c[1][i])
        return res

    def compute_coefficients(self):
        self.c[3] = self.y_points[:-1]

        mat = np.zeros((self.n - 1, self.n - 1))
        mat[0][0:2] = np.array([2*(self.h[0] + self.h[1]), self.h[1]])
        mat[-1][-3:] = np.array([self.h[-1], 2*(self.h[-1] + self.h[-2])])

        cx = np.zeros(self.n - 1)
        cx[0] = self.compute_cy_i(1)
        cx[1] = self.compute_cy_i(2)
        for i in range(2, self.n-1):
            mat[i, i-1] = self.h[i-1]
            mat[i, i] = 2 * (self.h[i-1] + self.h[i])
            mat[i, i+1] = self.h[i]
            cx[i] = self.compute_cy_i(i-1)
        self.c[1][1:] = np.linalg.solve(mat, cx)

        for i in range(self.n):
            self.c[2][i] = self.compute_b_i(i)
            self.c[0][i] = self.compute_d_i(i)


def solve_normal_system(A, y):
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, np.dot(Q.T, y))


def gauss_newton(g, Dg, lam0, max_iter=8, eps=1e-4):
    i = 0
    lam = np.copy(lam0)
    increment = eps + 1
    err_func = np.linalg.norm(g(lam)) ** 2

    while i < max_iter and eps < increment:
        Q, R = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R, np.dot(-Q.T, g(lam))).flatten()
        lam += delta
        err_func = np.linalg.norm(g(lam)) ** 2
        increment = np.linalg.norm(delta)
        i += 1
    return lam, i


def gauss_newton_d(g, Dg, lam0, max_iter=32, eps=1e-5, pmax=5, damped=True):
    if not damped:
        return gauss_newton(g, Dg, lam0, max_iter, eps)

    i = 0
    lam = np.copy(lam0)
    increment = eps + 1
    err_func = np.linalg.norm(g(lam)) ** 2

    while i < max_iter and eps < increment:
        Q, R = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R, -Q.T.dot(g(lam))).flatten()

        p = 0
        while p <= pmax and err_func <= np.linalg.norm(g(lam + delta / 2**p)) ** 2:
            p += 1
        if p <= pmax:
            lam += delta / 2**p
        else:
            lam += delta

        err_func = np.linalg.norm(g(lam)) ** 2
        increment = np.linalg.norm(delta)
        i += 1
        #print('iteration = ', i)
        #print('lambda = ', lam)
        #print('increment = ', increment)
        #print('error functional = ', err_func)

    return lam, i
