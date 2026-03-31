
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
        # d_out * partial derivative of activation with respect to z
        return d_out * (self.z > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, z):
        self.out = 1 / (1 + np.exp(-z)) # cache for backward
        return self.out

    def backward(self, d_out):
        # d_out * partial derivative of activation with respect to z
        return d_out * self.out * (1 - self.out) # = d_out * dy_pred/dz = d_z


class Softmax(Activation):
    def forward(self, z):
        z = z - np.max(z, axis=-1, keepdims=True)  # for numerical stability
        e_z = np.exp(z)
        self.out = e_z / e_z.sum(axis=-1, keepdims=True)  # cache for backward
        return self.out

    def backward(self, d_out):
        dot = (d_out * self.out).sum(axis=-1, keepdims=True)  # sum(d_out_j * s_j)
        return self.out * (d_out - dot)                        # s_i * (d_out_i - dot)



def Tanh(x: np.array) -> np.array:
    return np.tanh(x)

def GELU(x: np.array) -> np.array:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
