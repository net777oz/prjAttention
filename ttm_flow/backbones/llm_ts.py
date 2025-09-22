# -*- coding: utf-8 -*-
# LLM-style causal Transformer for multivariate time series
# Inputs:  x [B, C, T]  (C=채널/변량, T=시간)
# Outputs: pred_next [B, C] (t=T -> t+1 예측), feats [B, T, D] (임베딩)

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Positional Encoding (sinusoidal)
# -----------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

def build_causal_attn_mask(T: int, device) -> torch.Tensor:
    # nn.MultiheadAttention(batch_first=True)에서 float mask는
    # 0=통과, -inf=차단. 하삼각만 통과되도록 구성.
    mask_bool = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    attn_mask = (~mask_bool).float() * -1e9  # 상삼각을 -inf로 마스킹
    return attn_mask  # [T, T]

# -----------------------------
# GPT-style Block
# -----------------------------
class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x

# -----------------------------
# Main Model
# -----------------------------
class LLMTimeSeries(nn.Module):
    """
    입력:  x [B, C, T]
    출력:  pred_next [B, C], feats [B, T, D]
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, d_model)
        self.pe = SinusoidalPE(d_model, max_len=max_len)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_head, mlp_ratio, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, in_channels)  # 다변량 next-step 회귀

    @torch.no_grad()
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → 임베딩 [B, D] (T 평균 풀링)
        feats = self._forward_features(x)  # [B, T, D]
        return feats.mean(dim=1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T] → [B, T, C] → Linear → PE → Blocks
        x = x.permute(0, 2, 1).contiguous()
        x = self.proj_in(x)
        x = self.pe(x)
        T = x.size(1)
        attn_mask = build_causal_attn_mask(T, x.device)  # [T, T]
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.ln_f(x)  # [B, T, D]
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self._forward_features(x)   # [B, T, D]
        last = feats[:, -1, :]              # [B, D]
        pred_next = self.head(last)         # [B, C]
        return pred_next, feats
