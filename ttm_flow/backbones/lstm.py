# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class _LSTMHead(nn.Module):
    """
    입력:  x [B, C, T]
    출력:  (logits [B, 1], feats [B, T, H] or None)
    - 파이프라인이 기대하는 (pred, feats) 인터페이스를 맞춰서 반환합니다.
    """
    def __init__(self, in_channels: int, hidden: int = 128, num_layers: int = 2, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        y, _ = self.rnn(x)         # [B, T, H]
        last = y[:, -1, :]         # [B, H]
        logits = self.head(last)   # [B, out_dim]
        return logits, y           # (pred, feats)
        

def build_lstm_head(in_channels: int, hidden: int = 128, num_layers: int = 2, out_dim: int = 1, dropout: float = 0.0):
    return _LSTMHead(in_channels=in_channels, hidden=hidden, num_layers=num_layers, out_dim=out_dim, dropout=dropout)
