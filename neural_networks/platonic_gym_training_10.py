import numpy as np
from sklearn.datasets import make_blobs


class MLP:
    def __init__(self, n_hidden, seed=0x101):
        self.rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        self.d_in = 0
        self.d_out = 0
        self.b0 = None
        self.b1 = None
        self.w0 = None
        self.w1 = None
        # saving during forward pass
        self.z1 = None
        self.a1 = None

    def _init_weights(self):
        self.b0 = np.zeros((self.n_hidden, 1))
        self.b1 = np.zeros((self.d_out, 1))
        self.w0 = self.rng.standard_normal((self.n_hidden, self.d_in))
        self.w1 = self.rng.standard_normal((self.d_out, self.n_hidden))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def df_relu(x):
        return (0. < x).astype(float)

    def forward_pass(self, x):
        # z(1) = b(0) + w(0) * a(0)
        self.z1 = self.b0 + np.dot(self.w0, x)
        # a(1) = relu(z(1))
        self.a1 = self.relu(self.z1)
        # z(2) = b(1) + w(1) * a(1)
        return self.b1 + np.dot(self.w1, self.a1)

    def fit(self, inputs, targets, alpha=0.01, epochs=8):
        n_samples, self.d_in = inputs.shape
        if targets[0].shape:
            self.d_out = len(targets[0])
        else:
            self.d_out = 1
        self._init_weights()
        print(self.w0.shape)

        for _ in range(epochs):
            losses = []
            loss = 0
            for x, y in zip(inputs, targets):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                z2 = self.forward_pass(x)
                loss += np.mean((z2 - y)**2)
                # multivariate output
                dlz2 = 2 * np.mean(z2 - y)
                adlz2 = alpha * dlz2
                self.b1 -= adlz2
                print('w1', self.w1)
                print('adlz2 * np.trans', adlz2 * np.transpose(self.a1))
                self.w1 -= adlz2 * np.transpose(self.a1)
                self.b0 -= adlz2 * np.dot(self.w1, self.df_relu(self.z1))
                x_rep = np.tile(x.reshape(-1), (self.n_hidden, 1))
                self.w0 -= adlz2 * np.dot(self.w1, self.df_relu(self.z1)) * x_rep


            losses.append(loss)


if __name__ == '__main__':
    inputs, targets = make_blobs(n_samples=16,
                                 n_features=2)
    nn = MLP(n_hidden=3)
    nn.fit(inputs, targets)

