import numpy as np

# project imports
import snn


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def der_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def main():
    # inputs = np.linspace(-2 * np.pi, 2 * np.pi, 101)
    # targets = np.array([np.sin(x) for x in inputs])
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    targets = np.array([0, 1, 1, 0])

    nn = snn.SNN(d_layer=3, d_in=2, d_out=1, act=sigmoid, der_act=der_sigmoid)
    print(nn.forward(np.array([0, 0])))


if __name__ == '__main__':
    main()
