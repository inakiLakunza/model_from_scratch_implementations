
import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # for numerical stability
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
    



class Linear:

    def __init__(self, in_features, out_features, bias=True):
        # self.in_features = in_features
        # self.out_features = out_features
        # self.bias = bias

        self.W = np.random.uniform(-0.5, 0.5, [out_features, in_features])
        self.b = np.random.uniform(-0.5, 0.5, [out_features]) if bias else np.zeros(out_features)

    def forward(self, x):
        # x: [B, block, dim]
        # W: [dim, dim]
        out = x @ self.W.T
        out = out + self.b
        return out



class Head:

    def __init__(self, dim, n_heads):
        self.dim = dim
        self.head_dim = dim // n_heads

        self.Wq = np.random.uniform(-0.5, 0.5, [self.dim, self.head_dim])
        self.Wk = np.random.uniform(-0.5, 0.5, [self.dim, self.head_dim])
        self.Wv = np.random.uniform(-0.5, 0.5, [self.dim, self.head_dim])


    def forward(self, x: np.array) -> np.array:

        # x: [B, block, dim]

        # q: [B, block, head_dim]
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        e = q @ k.transpose(0, 2, 1)
        e = e / np.sqrt(self.head_dim)
        # e: [B, block, block]
        
        # softmax
        e = softmax(e)

        # attention
        # e: [B, block, block]
        # v = [B, block, head_dim]
        a = e @ v

        return a
    

class MultiHead:
    def __init__(self, dim, n_heads):
        self.dim = dim
        self.n_heads = n_heads
        self.heads = [Head(dim, n_heads) for _ in range(n_heads)]
        self.linear = Linear(dim, dim)

    def forward(self, x):
        x = np.concatenate([head.forward(x) for head in self.heads], axis=-1)
        x = self.linear.forward(x)
        return x




