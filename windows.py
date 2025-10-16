# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: windows.py                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  롤링 윈도우(Xw,Yw,groups,W) 생성                                 ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   build_windows_dataset(X:[N,C,T], L:int)                                 ║
║     -> (Xw:[N*W,C,L], Yw:[N*W], groups:[N*W], W:int)                      ║
╠───────────────────────────────────────────────────────────────────────────╣
║ NOTES  L은 [1,T-1]로 자동 보정, 라벨은 항상 첫 CSV(ch0)만 사용             ║
╠───────────────────────────────────────────────────────────────────────────╣
║ DEPENDENCY GRAPH  (pipeline, evaler, viz) → windows                       ║
╚════════════════════════════════════════════════════════════════════════════╝

windows.py — 롤링 윈도우 생성 (라벨=첫 CSV 고정)
"""
import torch
from typing import Tuple

def build_windows_dataset(X: torch.Tensor, L:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    X: [N, C, T]
      - C: CSV/채널 수 (첫 채널 ch0가 라벨 생성의 기준)
    반환:
      Xw:     [N*W, C, L]     입력 윈도우 (모든 채널 그대로)
      Yw:     [N*W]           타깃(회귀/분류 공통) — 항상 ch0만 사용 (1D)
      groups: [N*W]           원본 row 인덱스
      W:      int             타임 윈도우 개수 (T - L)

    L은 [1, T-1]로 보정됨.
    """
    assert X.dim() == 3, f"X must be [N,C,T], got {tuple(X.shape)}"
    N, C, T = X.shape
    L = max(1, min(L, T - 1))
    W = T - L

    # 입력 윈도우: unfold → [N, C, W+1, L] 중 앞 W개만 사용
    Xw_full = X.unfold(dimension=2, size=L, step=1)   # [N, C, W+1, L]
    Xw = Xw_full[:, :, :W, :]                         # [N, C, W, L]
    # 타깃(다음 시점 값): ch0만 사용
    Yw_full = X[:, 0:1, L:]                           # [N, 1, W]  ← ch0만 슬라이스

    # 배치 차원으로 평탄화
    Xw = Xw.permute(0, 2, 1, 3).contiguous().view(N * W, C, L)   # [N*W, C, L]
    Yw = Yw_full.permute(0, 2, 1).contiguous().view(N * W)       # [N*W] (1D)

    groups = torch.arange(N, device=X.device).repeat_interleave(W)  # [N*W]
    return Xw, Yw, groups, W
