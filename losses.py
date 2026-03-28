import numpy as np

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)


class BCELoss(Loss):
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        eps = 1e-8
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def backward(self):
        eps = 1e-8
        return -(self.y_true / (self.y_pred + eps)) + (1 - self.y_true) / (1 - self.y_pred + eps)
