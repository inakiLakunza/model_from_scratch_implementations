

import numpy as np
from activation_functions import ReLU, Sigmoid
from losses import BCELoss


allowed_activations = {
    "ReLU": ReLU,
    "Sigmoid": Sigmoid
}



class Layer():
    def __init__(self, input_dim = 8, output_dim = 8):
        self.input_dim = input_dim
        self.output_dim = output_dim


class HiddenLayer(Layer):
    def __init__(self, input_dim = 8, output_dim = 8, activation = "ReLU"):
        super().__init__(input_dim, output_dim)
        self.X = np.random.uniform(-0.5, 0.5, size=[output_dim, input_dim])
        self.b = np.random.uniform(-0.5, 0.5, size=[output_dim, 1])
        self.activation = allowed_activations[activation]()

    def forward(self, y_in):
        self.input = y_in
        z = self.X @ y_in + self.b
        return self.activation.forward(z)   # activation stores z internally

    def backward(self, d_out):
        d_z = self.activation.backward(d_out)   # activation handles its own derivative
        self.dX = d_z @ self.input.T
        self.db = d_z
        return self.X.T @ d_z
    
    def update(self, lr):
        self.X -= lr * self.dX
        self.b -= lr * self.db


class InputLayer(Layer):
    def __init__(self, dim):
        super().__init__(dim, dim)

    def forward(self, x):
        return x
    
    # No backward here, this layer is dummy
    

class OutputLayer(Layer):

    def __init__(self, input_dim):
        super().__init__(input_dim, 1)
        self.W = np.random.uniform(-0.5, 0.5, size=(1, input_dim))
        self.b = np.random.uniform(-0.5, 0.5, size=(1, 1))
        self.activation = Sigmoid()

    def forward(self, x):
        self.input = x # cache for backward
        self.z = self.W @ x + self.b
        return self.activation.forward(self.z)
    
    def backward(self, y_true, y_pred):
        d_z = y_pred - y_true
        self.dW = d_z @ self.input.T
        self.db = d_z
        return self.W.T @ d_z
    
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Model():

    def __init__(self, input_dim = 4, hidden_dims = [8, 16, 32, 16, 8]):
        self.input_layer = InputLayer(input_dim)
        self.hidden_layers = []
        prev = input_dim
        for dim in hidden_dims:
            self.hidden_layers.append(HiddenLayer(prev, dim))
            prev = dim
        self.output_layer = OutputLayer(prev)
        self.loss_fn = BCELoss()

    def forward(self, x):
        x = self.input_layer.forward(x)
        for layer in self.hidden_layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)
    
    def backward(self, y_true, y_pred):
        d = self.output_layer.backward(y_true, y_pred)
        for layer in reversed(self.hidden_layers):
            d = layer.backward(d) # get backward output, and pass it to the prev layer

    def update(self, lr=0.01):
        self.output_layer.update(lr)
        for layer in self.hidden_layers:
            layer.update(lr)

    def compute_loss(self, y_true, y_pred):
        return self.loss_fn.forward(y_true, y_pred)




def test():
    
    input = np.random.random([4, 1])
    model = Model()
    model.forward(input)



if __name__ == "__main__":
    test()