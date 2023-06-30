import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def sinusoidal(x):
    return np.sin(x)


def linear(x):
    return x


def hard_limit(x):
    return 1 if x >= 0 else 0


def relu(x):
    return np.maximum(0, x)


def leakyrelu(x):
    return np.maximum(0.1 * x, x)


functions = [
    sigmoid,
    tanh,
    sinusoidal,
    linear,
    hard_limit,
    relu,
    leakyrelu,
]
