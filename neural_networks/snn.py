"""
understanding deep learning book, shallow neural networks implementation

"""
from typing import Callable
import numpy as np
import sympy as sp


class Neuron:
    def __init__(self,
                 input_dim: int,
                 activation: Callable,
                 random_seed: int = 0x101):
        self.input_dim = input_dim
        self.activation = activation
        self.rng = np.random.default_rng(random_seed)
        self.bias = self.rng.standard_normal()
        self.weights = self.rng.standard_normal(self.input_dim)

    def forward(self, inputs):
        net_sum = self.bias + np.dot(inputs, self.weights)
        return self.activation(net_sum)

    def predict(self, x):
        pass

    def fit(self, inputs, targets, alpha=0.1, epochs=512):
        for _ in range(epochs):
            for x, y in zip(inputs, targets):
                err = self.forward(x) - y
                self.bias -= alpha * err
                self.weights -= alpha * err * x


class SNN:
    def __init__(self,
                 d_layer: int,
                 d_in: int,
                 d_out: int,
                 act: Callable,
                 der_act: Callable,
                 random_seed: int = 0x101):
        self.d_layer = d_layer
        self.d_in = d_in
        self.d_out = d_out
        self.act = act
        self.der_act = der_act
        self.beta0: np.ndarray
        self.beta1: np.ndarray
        self.omega0: np.ndarray
        self.omega1: np.ndarray
        self.init_network(random_seed)

    def init_network(self, random_seed):
        rng = np.random.default_rng(random_seed)
        self.beta0 = rng.standard_normal(self.d_layer)
        self.beta1 = rng.standard_normal(self.d_out)
        self.omega0 = rng.standard_normal(self.d_layer * self.d_in).reshape(
            (self.d_layer, self.d_in)
        )
        self.omega1 = rng.standard_normal(self.d_layer * self.d_out).reshape(
            (self.d_out, self.d_layer)
        )

    def forward(self, inputs: np.ndarray):
        h1 = self.act(self.beta0 + np.dot(self.omega0, inputs))
        return self.beta1 + np.dot(self.omega1, h1)

    def predict(self, inputs):
        pass

    def compute_grad_beta0(self, pred, x, y):
        grad_b0 = np.zeros_like(self.beta0)
        for i, b in enumerate(self.beta0):
            tmp = np.dot(self.omega0[i], x) + b
            grad_b0[i] = 2 * (pred - y) * self.der_act(tmp)
        return grad_b0

    def compute_grad_beta1(self, pred, y):
        return np.full(self.d_out, 2 * (pred - y))

    def compute_grad_omega0(self, pred, x, y):
        grad_o0 = np.zeros_like(self.omega0)
        for i, w in self.omega0:
            tmp = self.beta0[i] + np.dot()
            grad_o0[i] = 2 * (pred - y) * self.der_act(tmp)

    def compute_grad_omega1(self, pred, x, y):
        grad_o1 = np.zeros_like(self.omega1)
        for i, w in self.omega1:
            tmp = self.beta0[i] + np.dot(self.omega0[:, i], x)
            grad_o1[i] = 2 * (pred - y) * self.act(tmp)
        return grad_o1

    def fit(self, inputs, targets, alpha=0.01, epochs=512):
        for _ in range(epochs):

            for x, y in zip(inputs, targets):
                pred = self.forward(x)
                grad_b1 = self.compute_grad_beta1(pred, y)
                grad_b0 = self.compute_grad_beta0(pred, x, y)
                grad_o0 = self.compute_grad_omega0(pred, x, y)
                grad_o1 = self.compute_grad_omega1(pred, x, y)


