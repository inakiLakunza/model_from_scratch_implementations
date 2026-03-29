

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
        self.input = y_in # save for backward
        z = self.X @ y_in + self.b
        return self.activation.forward(z)   # activation stores z internally

    def backward(self, d_out):
        '''
        dL/dW = dL/dy_pred * dy_pred/dz * dz/dW
            1. dL/dy_pred = -y/y_pred + (1-y)/(1-y_pred)
            2. dy_pred/dz = sigmoid(z) * (1 - sigomoid(z)) -> Computed in activations_functions.py
            3. dz/dW = x

            d_out == dL/dy_pred -> Computed in losses.py or taken from previous layer 
            d_z == dL/dy_pred * dy_pred/dz

            
        dL/db = dL/dy_pred * dy_pred/dz * dz/db
            1. dL/dy_pred = -y/y_pred + (1-y)/(1-y_pred)
            2. dy_pred/dz = sigmoid(z) * (1 - sigomoid(z)) -> Computed in activations_functions.py
            3. dz/db = 1

        What we send to the prev layer, the contrubtion of the input x to the loss dL/dx
        dL/dx = dL/dy_pred * dy_pred/dz * dz/dx
            1. dL/dy_pred = -y/y_pred + (1-y)/(1-y_pred)
            2. dy_pred/dz = sigmoid(z) * (1 - sigomoid(z)) -> Computed in activations_functions.py
            3. dz/dx = W
        
        WE WILL STORE AS SELF VARIABLES dX (dX == dW) and db and return dL/dx SO THAT THE PREV LAYER CAN TAKE IT
        '''
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
    
    def backward(self, d_loss):
        d_z = self.activation.backward(d_loss)
        self.dW = d_z @ self.input.T # dL/dW used to update weights
        self.db = d_z # dL/db used to update bias
        return self.W.T @ d_z # dL/dx passed back to the previous layer
    
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Model():

    def __init__(self, input_dim = 4, hidden_dims = [8, 16, 32, 64, 32, 16, 8], loss=BCELoss):
        self.input_layer = InputLayer(input_dim)
        self.hidden_layers = []
        prev = input_dim
        for dim in hidden_dims:
            self.hidden_layers.append(HiddenLayer(prev, dim))
            prev = dim
        self.output_layer = OutputLayer(prev)
        self.loss_fn = loss()

    def forward(self, x):
        x = self.input_layer.forward(x)
        for layer in self.hidden_layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)
    
    def backward(self):
        d_loss = self.loss_fn.backward()
        d = self.output_layer.backward(d_loss)
        for layer in reversed(self.hidden_layers):
            d = layer.backward(d) # get backward output, and pass it to the prev layer

    def update(self, lr=0.01):
        self.output_layer.update(lr)
        for layer in self.hidden_layers:
            layer.update(lr)

    def compute_loss(self, y_true, y_pred):
        return self.loss_fn.forward(y_true, y_pred)




def test_forward():
    input = np.random.random([8, 4, 1])
    model = Model()
    out = model.forward(input)
    print(out)



def test_train():
    model = Model()
    lr = 0.01
    n_epochs = 100
    
    X_data = [np.random.random([4, 1]) for _ in range(20)]
    y_data = [np.array([[float(np.random.randint(2))]]) for _ in range(20)]

    for epoch in range(n_epochs):
        total_loss = 0
        for x, y in zip(X_data, y_data):
            y_pred = model.forward(x)
            total_loss += model.compute_loss(y, y_pred)
            model.backward()
            model.update(lr)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss/len(X_data):.4f}")

    print(f"Model finished training for {n_epochs}  | Loss: {total_loss/len(X_data):.4f}")




class Train:
    def __init__(self, model, n_epochs=100, lr=0.01, split_instances=[1000, 10, 10]):
        self.model = model
        
        self.n_epochs = n_epochs
        self.lr = lr
        
        self.X_train = [np.random.random([4, 1]) for _ in range(split_instances[0])]
        self.y_train = [np.array([[1.0 if x.sum() > 2 else 0.0]]) for x in self.X_train]

        self.X_val = [np.random.random([4, 1]) for _ in range(split_instances[1])]
        self.y_val =  [np.array([[1.0 if x.sum() > 2 else 0.0]]) for x in self.X_val]

        self.X_test = [np.random.random([4, 1]) for _ in range(split_instances[2])]
        self.y_test =  [np.array([[1.0 if x.sum() > 2 else 0.0]]) for x in self.X_test]

    def validate(self):
        total_loss = 0
        for x, y in zip(self.X_val, self.y_val):
            y_pred = self.model.forward(x)
            total_loss += self.model.compute_loss(y, y_pred)
        print(f"Validation Loss: {total_loss/len(self.X_val):.4f}")

    def test(self):
        total_loss = 0
        for x, y in zip(self.X_test, self.y_test):
            y_pred = self.model.forward(x)
            total_loss += self.model.compute_loss(y, y_pred)
        print(f"Test Loss: {total_loss/len(self.X_test):.4f}")

    def train(self):
        for epoch in range(self.n_epochs):
            total_loss = 0
            for x, y in zip(self.X_train, self.y_train):
                y_pred = self.model.forward(x)
                total_loss += self.model.compute_loss(y, y_pred)
                self.model.backward()
                self.model.update(self.lr)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {total_loss/len(self.X_train):.4f}")
                self.validate()

        
        print("\n-----------------------------------")
        print(f"Model finished training for {self.n_epochs}")
        print(f"Train Loss: {total_loss/len(self.X_train):.4f}")
        self.validate()
        self.test()
        print("-----------------------------------")


if __name__ == "__main__":
    test_forward()
    print("\n\n\n")
    test_train()
    print("\n\n\n")
    
    model = Model()
    train = Train(model)
    train.train()
