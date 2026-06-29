import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
with urllib.request.urlopen(url) as f:
    text = f.read().decode("utf-8")

class MHSelfAttention(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.head_dim = emb_dim//head_num

        self.WQ = nn.Linear(emb_dim, emb_dim)
        self.WK = nn.Linear(emb_dim, emb_dim)
        self.WV = nn.Linear(emb_dim, emb_dim)
        self.WO = nn.Linear(emb_dim, emb_dim)

        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        Q = Q.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        K = K.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        V = V.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        score = Q @ K.transpose(-2,-1)
        score = score/(self.emb_dim ** 0.5)
        score = score.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(score, -1)

        out = weights @ V
        out = out.transpose(1,2).contiguous()
        out = out.view(B, T, self.emb_dim)
        out = self.WO(out)
        return out

class Transformer(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len):
        super().__init__()
        self.attn = MHSelfAttention(emb_dim, head_num, seq_len)
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
       out = self.layer(x)
       x = x+out
       x = self.norm2(x)
       return x

class TinyGPT(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len, vocab_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.seq_len = seq_len
        self.head_dim = emb_dim//head_num

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.posemb = nn.Embedding(seq_len, emb_dim)
        self.mod = nn.ModuleList([Transformer(emb_dim, head_num, seq_len) for _ in range(4)])
        self.norm = nn.LayerNorm(emb_dim)
        self.lin = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = x[:, :self.seq_len]
        tok = self.emb(x)
        _ , T, _ = tok.shape
        pos = self.posemb(torch.arange(T, device=x.device))
        x = tok + pos
        for block in self.mod:
            x = block(x)
        x = self.norm(x)
        out = self.lin(x)
        return out

chars = sorted(set(text))
vocab_size = len(chars)

cha = {}
idx = {}
for i, ch in enumerate(chars):
    cha[ch] = i
for ch, i in cha.items():
    idx[i] = ch

data = torch.tensor([cha[ch] for ch in text])

X, y = [], []
seq_len = 256
for i in range(0, len(data) - seq_len, seq_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+1:i+seq_len+1])

X = torch.stack(X)
y = torch.stack(y)

dset = TensorDataset(X, y)
loader = DataLoader(dset, 512, shuffle=True)

model = TinyGPT(64, 8, seq_len, vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lossfn = nn.CrossEntropyLoss()

losses = []

for epoch in range(150):
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        out = model(x_batch)
        out = out.view(-1, vocab_size)
        y_batch = y_batch.view(-1)
        loss = lossfn(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/150  Loss: {avg_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, 151), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
prompt = "ROMEO:"
context = torch.tensor([[cha[ch] for ch in prompt]]).to(device)

model.eval()
for _ in range(500):
    context_crop = context[:, -seq_len:]
    out = model(context_crop)
    out = out[:, -1, :]
    probs = F.softmax(out/0.8, dim=-1)
    next_token = torch.multinomial(probs, 1)
    context = torch.cat([context, next_token], dim=1)

output = "".join([idx[i.item()] for i in context[0]])
print(output)