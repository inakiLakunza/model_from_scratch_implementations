

import numpy as np
from activation_functions import *


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
        self.X = np.zeros(shape = [output_dim, input_dim])
        self.b = np.zeros(shape = [output_dim, 1])
        self.activation = allowed_activations.get(activation)
        assert activation in allowed_activations.keys()
        self.init_random()

    def init_random(self):
        self.X = np.random.uniform(low=-0.5, high=0.5, size=self.X.shape)
        self.b = np.random.uniform(low=-0.5, high=0.5, size=self.b.shape)


    def forward(self, y_in):
        print(f"Input: {y_in}")
        x = y_in
        print("shapes: ", x.shape, self.X.shape, self.b.shape)
        x = self.X @ y_in + self.b
        x = self.activation(x)
        print(f"Output: {x}")
        return x



class InputLayer(Layer):

    def __init__(self, dim):
        super().__init__(dim, dim)

    def forward(self, x):
        return x
    

class OutputLayer(Layer):

    def __init__(self, input_dim):
        super().__init__(input_dim, 1)
        self.W = np.random.uniform(-0.5, 0.5, size=(1, input_dim))
        self.b = np.random.uniform(-0.5, 0.5, size=(1, 1))


    def forward(self, x):
        logits = self.W @ x + self.b
        return Sigmoid(logits)




class Model():

    def __init__(self, input_dim = 4, hidden_dims = [8, 16, 32, 16, 8]):
        self.input_layer = InputLayer(input_dim)
        self.hidden_layers = []
        prev = input_dim
        for dim in hidden_dims:
            self.hidden_layers.append(HiddenLayer(prev, dim))
            prev = dim
        self.output_layer = OutputLayer(prev)

    def forward(self, x):
        print(f"Model input: {x}\n\n")
        x = self.input_layer.forward(x)
        for layer in self.hidden_layers:
            x = layer.forward(x)
        x = self.output_layer.forward(x)
        print(f"Model output: {x}")
        return x




def test():
    
    input = np.random.random([4, 1])
    model = Model()
    model.forward(input)



if __name__ == "__main__":
    test()