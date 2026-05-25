import pandas as pd
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
