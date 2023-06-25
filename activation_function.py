import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear(x):
    return x


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.maximum(0, x)


def leakyrelu(x):
    return np.maximum(0.1 * x, x)


functions = [
    sigmoid,
    linear,
    softmax,
    relu,
    leakyrelu,
]
