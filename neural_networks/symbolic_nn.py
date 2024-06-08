import numpy as np
import sympy as sp


class SymNeuralNet:
    def __init__(self, seed=0x101):
        self.rng = np.random.default_rng(seed)
        # hidden layer bias and weights
        self.beta0 = sp.MatrixSymbol('beta0', 3, 1)
        self.omega0 = sp.MatrixSymbol('omega0', 3, 2)
        # output layer bias and weights
        self.beta1 = sp.MatrixSymbol('beta1', 1, 1)
        self.omega1 = sp.MatrixSymbol('omega1', 1, 3)
        # initial values
        self.omega0_values = self.rng.standard_normal((3, 2))
        self.beta0_values = self.rng.standard_normal((3, 1))
        self.beta1_values = self.rng.standard_normal((1, 1))
        self.omega1_values = self.rng.standard_normal((1, 3))

    def compute_gradient_beta0(self, x):
        grad_beta0 = np.ones_like(self.beta0_values)
        tmp = self.beta0_values + np.dot(self.omega0_values, x)
        if tmp < 0:
            return np.zeros_like(self.beta0_values)
        else:
            return np.dot(self.omega1_values, grad_beta0)

    def compute_gradient_beta1(self):
        return np.ones_like(self.beta1_values)

    def compute_gradient_omega0(self, x):
        tmp = self.beta0_values + np.dot(self.omega0_values, x)
        if tmp < 0:
            return np.zeros_like(self.omega0_values)
        else:
            xx = np.array([x.reshape(1, -1), x.reshape(1, -1), x.reshape(1, -1)])
            return np.dot(self.omega1_values, xx)

    def compute_gradient_omega1(self, x):
        return np.max(0, self.beta0_values + np.dot(self.omega0_values, x))

    def predict(self, x):
        # ReLU
        h1 = np.max(0, (self.beta0 + np.dot(self.omega0_values, x)))
        return self.beta1 + np.dot(self.omega1_values, h1)

    def fit(self, inputs, targets, alpha=0.1, n_iter=8):
        for _ in range(n_iter):

            for xi, yi in zip(inputs, targets):
                print('xi=', xi, xi.shape)
                ypred = self.predict(xi.reshape(-1, 1))
                deru = 2 * (ypred - yi)
                self.beta0_values -= alpha * deru * self.compute_gradient_beta0(xi)
                self.beta1_values -= alpha * deru * self.compute_gradient_beta1()
                self.omega0_values -= alpha * deru * self.compute_gradient_omega0(xi)
                self.omega1_values -= alpha * deru * self.compute_gradient_omega1(xi)


if __name__ == '__main__':
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

    y = np.array([0., 1., 1., 0.])
    sn = SymNeuralNet()
    sn.fit(X, y)

    #print(sn.predict(X))

