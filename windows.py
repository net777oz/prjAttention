# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: windows.py                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  롤링 윈도우(Xw,Yw,groups,W) 생성                                 ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   build_windows_dataset(X:[N,1,T], L:int)                                 ║
║     -> (Xw:[N*W,1,L], Yw:[N*W,1], groups:[N*W], W:int)                    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ NOTES  L은 [1,T-1]로 자동 보정                                            ║
╠───────────────────────────────────────────────────────────────────────────╣
║ DEPENDENCY GRAPH  (pipeline, evaler, viz) → windows                       ║
╚════════════════════════════════════════════════════════════════════════════╝

windows.py — 롤링 윈도우 생성
"""
import torch
from typing import Tuple

def build_windows_dataset(X: torch.Tensor, L:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    X: [N,1,T] → Xw:[N*W,1,L], Yw:[N*W,1], groups:[N*W], W=T-L
    L은 [1, T-1]로 보정됨.
    """
    N, C, T = X.shape
    L = max(1, min(L, T-1))
    W = T - L
    Xw_full = X.unfold(dimension=2, size=L, step=1)   # [N,1,W+1,L]
    Xw = Xw_full[:, :, :W, :]                         # [N,1,W,L]
    Yw = X[:, :, L:]                                  # [N,1,W]
    Xw = Xw.permute(0,2,1,3).contiguous().view(N*W, C, L)
    Yw = Yw.permute(0,2,1).contiguous().view(N*W, C)
    groups = torch.arange(N).repeat_interleave(W)
    return Xw, Yw, groups, W
