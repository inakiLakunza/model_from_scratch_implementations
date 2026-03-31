
import numpy as np

from activation_functions import Softmax, ReLU



class TokenEmbedding:

    def __init__(self, vocab_size, dim):
        self.embedding = np.random.uniform(-0.5, 0.5, [vocab_size, dim])
    
    def forward(self, x):
        # x: [B, block]
        return self.embedding[x] # [B, block, dim]
    

class PositionalEmbedding:

    def __init__(self, block_size, dim):
        pe = np.zeros([block_size, dim])

        pos = np.arange(block_size).reshape(-1, 1)
        i = np.arange(0, dim, 2)
        div = np.power(10000, i / dim)

        pe[:, 0::2] = np.sin(pos / div) # even dims -> sin
        pe[:, 1::2] = np.cos(pos / div) # odd dims  -> cos

        self.embedding = pe

    def forward(self, x):
        B, T, D = x.shape
        return self.embedding[:T, :]


class Embedding:

    def __init__(self, vocab_size, block_size, dim):
        self.token_emb = TokenEmbedding(vocab_size, dim)
        self.pos_emb = PositionalEmbedding(block_size, dim)

    def forward(self, x):
        # x: [B, block size]
        tok = self.token_emb.forward(x)
        pos = self.pos_emb.forward(tok)
        return tok + pos


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
        self.x = x # cache for backward
        out = x @ self.W.T
        out = out + self.b
        return out
    
    def backward(self, d_out):
        pass


class FeedForward:

    def __init__(self, dim):
        self.upscale = Linear(dim, dim * 4)
        self.downscale = Linear(dim * 4, dim)
        self.relu = ReLU()

    def forward(self, x):
        # x: [B, block, dim=512]
        out = self.upscale.forward(x)
        # out: [B, block, 2048]
        out = self.relu.forward(out)
        out = self.downscale.forward(out)
        # out: [B, block, 2048]
        return out


class Head:

    def __init__(self, dim, n_heads):
        self.dim = dim
        self.head_dim = dim // n_heads
        self.softmax = Softmax()

        self.Wq = np.random.uniform(-0.5, 0.5, [self.dim, self.head_dim])
        self.Wk = np.random.uniform(-0.5, 0.5, [self.dim, self.head_dim])
        self.Wv = np.random.uniform(-0.5, 0.5, [self.dim, self.head_dim])


    def forward(self, x: np.array, mask: bool = True) -> np.array:

        # x: [B, block, dim]

        # q: [B, block, head_dim]
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        e = q @ k.transpose(0, 2, 1)
        e = e / np.sqrt(self.head_dim)
        # e: [B, block, block]
        
        if mask:
            T = e.shape[-1]
            causal_mask = np.triu(np.ones((T, T)), k=1).astype(bool)
            e = np.where(causal_mask, -np.inf, e)

        # softmax
        e = self.softmax.forward(e)

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
    

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)   # scale — initialized to 1
        self.beta  = np.zeros(dim)  # shift — initialized to 0
        self.eps   = eps

    def forward(self, x):
        # x: [B, block, dim]
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta  # [B, block, dim]


class Block:

    def __init__(self, dim):
        self.mha = MultiHead(dim=512, n_heads=8)
        self.ff = FeedForward(dim=512)
        self.ln1 = LayerNorm(dim=512)
        self.ln2 = LayerNorm(dim=512)

    def forward(self, x):
        x = x + self.mha.forward(self.ln1.forward(x))
        x = x + self.ff.forward(self.ln2.forward(x))
        return x


class Model:

    def __init__(self, vocab_size, block_size, dim=512, n_blocks=6):
        self.block_size = block_size
        self.embed = Embedding(vocab_size, block_size, dim)
        self.blocks = [Block(dim) for _ in range(n_blocks)]
        self.norm = LayerNorm(dim)
        self.linear = Linear(dim, vocab_size)
        self.softmax = Softmax()

    def forward(self, x):
        # x: [B, block]
        x = self.embed.forward(x)
        # x: [B, block, dim]
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        x = self.linear.forward(x)
        return x


    def generate(self, tokens, max_new_tokens):

        for _ in range(max_new_tokens):
            context = tokens[:, -self.block_size:]
            
            logits = self.forward(context)
            logits = logits[:, -1, :] # last token

            probs = self.softmax.forward(logits)
            next_token = np.random.choice(len(probs[0]), p=probs[0])

            tokens = np.concatenate(
                [tokens, np.array([[next_token]])], axis=1
            )

            return tokens




def cross_entropy_loss(logits, targets):
    # logits: [B, T, vocab_size]
    # targets: [B, T]

    # shift by 1: input predicts next token
    logits = logits[:, :-1, :] # [B, T-1, vocab_size]
    targets = targets[:, 1:] # [B, T-1]

    B, T, V = logits.shape
    N = B * T

    logits_flat = logits.reshape(N, V)
    targets_flat = targets.reshape(N)

    logits_flat -= logits_flat.max(axis=-1, keepdims=True)

    exp = np.exp(logits_flat)
    probs = exp / exp.sum(axis=-1, keepdims=True) # [N, V]

    loss = -np.log(probs[np.arange(N), targets_flat] + 1e-9).mean()

    dlogits = probs.copy()
    dlogits[np.arange(N), targets_flat] = -1
    dlogits /= N

    dlogits = dlogits.reshape(B, T, V)

    return loss, dlogits



