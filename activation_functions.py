
import numpy as np



class Activation:
    def forward(self, z):
        raise NotImplementedError
    
    def backward(self, d_out):
        raise NotImplementedError
    
    def __call__(self, z):
        return self.forward(z)
    

class ReLU(Activation):
    def forward(self, z):
        self.z = z # cache for backward
        return np.maximum(0, z)

    def backward(self, d_out):
        return d_out * (self.z > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, z):
        self.out = 1 / (1 + np.exp(-z)) # cache for backward
        return self.out

    def backward(self, d_out):
        return d_out * self.out * (1 - self.out)



def Tanh(x: np.array) -> np.array:
    return np.tanh(x)

def GELU(x: np.array) -> np.array:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def Softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # for numerical stability
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)