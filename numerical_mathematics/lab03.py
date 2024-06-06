import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def newton_method(f, df, x0, n_iter=5):

    for _ in range(n_iter):
        x1 = x0[0]
        x2 = x0[1]
        A = np.array(df(x1, x2), dtype=np.float64)
        b = np.array(-f(x1, x2), dtype=np.float64)
        d0 = np.linalg.solve(A, b).reshape(2,)
        x1 = x0 + d0
        print(f'x = {x0}')
        print('norm f(x)', np.linalg.norm(-b.flatten(), ord=2))
        print('norm x1 - x0:', np.linalg.norm(x1 - x0, ord=2))
        x0 = x1


def task1():
    x1, x2 = sp.symbols('x1 x2')
    f1 = 20 - 18*x1 - 2*x2**2
    f2 = -4*x2 * (x1 - x2**2)
    f = sp.Matrix([f1, f2])
    df = f.jacobian((x1, x2))

    newton_method(sp.lambdify((x1, x2), f),
                  sp.lambdify((x1, x2), df),
                  np.array([1.1, 0.9]),
                  n_iter=3)


def damped_newton_method(f, df, x0, n_iter=5):
    pass


def task3():
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    f1 = x1 + x2**2 - x3**2 - 13
    f2 = sp.ln(x2/4) + sp.exp(0.5*x3 - 1) - 1
    f3 = (x2 - 3)**2 - x3**3 + 7
    f = sp.Matrix([f1, f2, f3])
    df = f.jacobian((x1, x2, x3))

    damped_newton_method(sp.lambdify((x1, x2, x3), f),
                         sp.lambdify((x1, x2, x3), df),
                         np.array([1.5, 3, 2.5]))


def main():
    # symbolic computations setup
    sp.init_printing()

    # task1()
    task3()



if __name__ == '__main__':
    main()

