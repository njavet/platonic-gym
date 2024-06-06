import numpy as np
import sympy as sp
import matplotlib.pyplot as plt




def main():
    # symbolic computations setup
    sp.init_printing()

    # define symbols 
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    x = sp.symbols('x')

    # define functions
    f1 = x1 + x2**2 - x3**2 - 13
    f2 = sp.ln(x2/4) + sp.exp(0.5*x3 - 1) - 1
    f3 = (x2 - 3)**2 - x3**3 + 7

    # define a function matrix style
    f = sp.Matrix([f1, f2, f3])

    # get jacobian matrix 
    df = f.jacobian((x1, x2, x3))

    # substitute variables with numbers 
    fx0 = f.subs([(x1, 1.5), (x2, 3), (x3, 2.5)])
    df0 = df.subs([(x1, 1.5), (x2, 3), (x3, 2.5)])

    # lambdify functions to use them like normal functions
    fl = sp.lambdify((x1, x2, x3), f)
    dfl = sp.lambdify((x1, x2, x3), df)

    # linearize function
    dx = sp.Matrix([x1 - 1.5, x2 - 3, x3 - 2.5])
    g = fx0 + np.dot(df0, dx)


if __name__ == '__main__':
    main()

