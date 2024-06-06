import numpy as np
from functools import wraps


def input_check(func):
    @wraps(func)
    def check(A, b):
        try:
            m, n = A.shape
        except ValueError:
            raise Exception('input must be a square 2d array')
        if m != n:
            raise Exception('input must be a square 2d array')
        try:
            mb, nb = b.shape
        except ValueError:
            raise Exception(f'{b} must be a {n} x 1 vector')
        if mb != n and nb != 1:
            raise Exception(f'{b} must be a {n} x 1 vector')
        return func(A, b)
    return check


class Solver:
    def __init__(self, A):
        self.P, self.L, self.U = self.lu_dec_with_pivoting(A)

    def lu_dec_with_pivoting(self, A):
        n = A.shape[0]
        L = np.eye(n)
        P = np.eye(n)
        U = np.copy(A)
        for i in range(n-1):
            print(f'before U = \n{U}')
            max_ind = np.argmax(np.abs(U[i:, i]))
            P[[i, max_ind + i]] = P[[max_ind + i, i]]
            U[[i, max_ind + i]] = U[[max_ind + i, i]]
            print(f'after U = \n{U}')
            for j in range(i+1, n):
                L[j, i] = U[j, i] / U[i, i]
                print(f'fac = {L[j, i]}')
                print(f'U{j} = {U[j]}')
                print(f'U{i} = {U[i]}')
                print(f'facU[i] = {L[j, i] * U[i]}')
                aa = L[j, i] * U[i]
                print(f'a = {aa}')
                print(f'U[j] = {U[j]}')
                print(f'u[j] - aa {U[j] - aa}')
                U[j] = U[j] - L[j, i] * U[i]
                print(f'final U{j} = {U[j]}')
        return P, L, U

    def solve(self, b):
        y = forward_substitution(self.L, np.dot(self.P, b))
        return backward_substitution(self.U, y)


def solve_linear_system(A, b):
    L, U = lu_decomposition_without_pivoting(A)
    y = forward_substitution(L, b)
    return backward_substitution(U, y)


def lu_decomposition_without_pivoting(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A)
    for i in range(n-1):
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j] = U[j] - L[j, i] * U[i]
    return L, U


@input_check
def gauss_without_pivoting(A, b):
    Aext = np.hstack((A, b))
    n = A.shape[0]
    for i in range(n-1):
        for j in range(i+1, n):
            fac = Aext[j, i] / Aext[i, i]
            Aext[j] = Aext[j] - fac*Aext[i]
    return backward_substitution(Aext[:, 0:n], Aext[:, n])


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - np.dot(y[0:i], L[i, 0:i])
    return y


def backward_substitution(R, y):
    n = R.shape[0]
    x = np.zeros_like(y)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(x[i+1:], R[i, i+1:])) / R[i, i]
    return x

