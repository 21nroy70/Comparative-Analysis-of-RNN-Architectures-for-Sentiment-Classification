'''Nothing here to run lol, just defining the RNN class with functions and stuff which will be used in train.py to train and see models/performance'''


import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        arch: str = "rnn",           # rnn | lstm | bilstm
        dropout: float = 0.5,
        head_activation: str = "relu"
    ):
        super().__init__()
        self.arch = arch.lower()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        if self.arch == "rnn":
            self.encoder = nn.RNN(
                emb_dim, hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout, nonlinearity="tanh"
            )
            feat_dim = hidden_size
        elif self.arch == "lstm":
            self.encoder = nn.LSTM(
                emb_dim, hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout, bidirectional=False
            )
            feat_dim = hidden_size
        elif self.arch == "bilstm":
            self.encoder = nn.LSTM(
                emb_dim, hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout, bidirectional=True
            )
            feat_dim = hidden_size * 2
        else:
            raise ValueError("arch must be one of rnn|lstm|bilstm")

        Act = ACTIVATIONS[head_activation.lower()]
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_size),
            Act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, T]
        emb = self.emb(x)  # [B,T,E]
        if self.arch == "rnn":
            out, h = self.encoder(emb)  # h: [L,B,H]
            feat = h[-1]                # last layer hidden state [B,H]
        elif self.arch in ("lstm", "bilstm"):
            out, (h, c) = self.encoder(emb)
            if self.arch == "bilstm":
                # concat last layer forward + backward
                h_last = h[-2:]  # [2,B,H]
                feat = torch.cat([h_last[0], h_last[1]], dim=1)  # [B,2H]
            else:
                feat = h[-1]  # [B,H]
        prob = self.head(feat).squeeze(1)  # [B]
        return prob
