
import numpy as np


def ReLU(x: np.array) -> np.array:
    return np.maximum(0, x)

def Sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))

def Tanh(x: np.array) -> np.array:
    return np.tanh(x)

def GELU(x: np.array) -> np.array:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def Softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # for numerical stability
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)