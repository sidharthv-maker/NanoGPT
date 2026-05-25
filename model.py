import torch
import torch.nn as nn
import torch.nn.functional as F
class MHSelfAttention(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len):
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.head_dim = emb_dim//head_num

        self.WQ = nn.Linear(emb_dim, emb_dim)
        self.WK = nn.Linear(emb_dim, emb_dim)
        self.WV = nn.Linear(emb_dim, emb_dim)

        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        Q.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        K.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        V.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        scores = Q @ K.transpose(-2,-1)
        score = score/(self.emb_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(score, -1)

        out = weights @ V
        out = out.transpose(1,2).contiguous()
        out = out.view(B, T, self.emb_dim)
        out = self.W_O(out)
        return out

class Transformer(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len):
        self.attn = MHSelfAttention(emb_dim, head_num)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.layer = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
       out = self.attn(x)
       x = x + out
       x = self.norm1(x)
       out = self.ff(x)
       x = x+out
       x = self.norm2(x)
       return x

class TinyGPT(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.head_dim = emb_dim//head_num

        self.emb = nn.Embedding(28, emb_dim)
        self.posemb = nn.Embedding(seq_len, emb_dim)
        self.mod = nn.ModuleList([Transformer(emb_dim, head_num, seq_len) for _ in range(20)])
        self.norm = nn.LayerNorm(emb_dim)
        self.lin = nn.Linear(emb_dim, 28)

    def forward(self, x):
        tok = self.emb(x)
        _ , T, _ = tok.shape
        pos = self.posemb(torch.arange(T))
        x = tok + pos  
        for block in self.mod:
            x = block(x)
        x = self.norm(x)
        out = self.lin(x)
        return out

with open("shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

cha = {}
idx = {}
for i, ch in enumerate(chars):
    cha[ch] = i
for ch, i in cha.items():
    idx[i] = ch

data = torch.tensor([cha[ch] for ch in text])

X, y = []
seq_len = 256
for i in range(0, len(data) - seq_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+1:i+seq_len+1])

X = torch.stack(X)
y = torch.stack(y)

dset = TensorDataset(X,y)
loader = DataLoader(dset, 32, shuffle=True)