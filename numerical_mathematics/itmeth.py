import numpy as np


def jacobi(A, b, x0):
    L = np.tril(A)
    np.fill_diagonal(L, 0)
    Dinv = np.diag(1 / np.diag(A))
    R = np.triu(A)
    np.fill_diagonal(R, 0)

    for _ in range(5):
        xn = -np.dot(Dinv, np.dot(L + R, x0)) + np.dot(Dinv, b)
        x0 = xn
        print(f'{xn}')
    return xn

