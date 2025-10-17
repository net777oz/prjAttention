# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: windows.py                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  롤링 윈도우(Xw, Yw, groups, W) 생성                               ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   build_windows_dataset(X:[N,C,T], L:int, label_offset:int=1, step:int=1) ║
║     -> (Xw:[N*W,C,L], Yw:[N*W], groups:[N*W], W:int)                      ║
╠───────────────────────────────────────────────────────────────────────────╣
║ NOTES                                                                     ║
║  • 기본 Δ=+1 → 윈도우 끝(t = s+L-1)의 다음 시점(t+Δ)을 라벨로 사용            ║
║  • Δ 변경은 label_offset으로 설정(예: 동시시점 라벨이면 Δ=0)                 ║
║  • 모든 row가 동일한 T를 가진다고 가정, 라벨은 항상 ch0 기준                ║
║  • 경계(T = L + Δ)에서도 W=1이 되도록 오프바이원(+1) 처리 완료               ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Tuple
import torch


def build_windows_dataset(
    X: torch.Tensor,
    L: int,
    label_offset: int = 1,   # Δ (기본: 다음 시점)
    step: int = 1,           # 윈도우 시작 간격
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    입력:
      X: [N, C, T]  (torch.Tensor)
         - C: 채널 수 (라벨은 ch0에서 생성)
      L: 윈도우 길이 (context_len)
      label_offset(Δ): 윈도우 끝에서 Δ 스텝 뒤 시점을 라벨로 사용 (기본 1)
      step: 시작 인덱스 간격 (기본 1)

    반환:
      Xw:     [N*W, C, L]
      Yw:     [N*W]           (float/long은 후단에서 변환)
      groups: [N*W]           (row grouping id)
      W:      int             윈도우 개수

    정의:
      - 윈도우 i의 시작 s_i = i * step
      - 윈도우 범위: [s_i, s_i + L - 1]
      - 라벨 인덱스: t_i = s_i + (L - 1) + Δ
      - 유효 조건: t_i < T
        → s_i ≤ T - (L + Δ)  → 최대 시작 s_max = T - (L + Δ)
        → W = floor(s_max/step) + 1  (s_max ≥ 0일 때), 그렇지 않으면 W ≤ 0
    """
    # 기본 체크
    if X.dim() != 3:
        raise ValueError(f"X must be [N,C,T], got {tuple(X.shape)}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if label_offset < 0:
        raise ValueError(f"label_offset(Δ) must be >= 0, got {label_offset}")

    N, C, T = X.shape

    # L이 너무 크면(T-Δ보다 크면) 줄여서라도 동작하게 보정
    if L > T - label_offset:
        L = max(1, T - label_offset)

    # 시작 인덱스의 최대값 s_max = T - (L + Δ)
    s_max = T - (L + label_offset)
    # W 계산(경계 T = L + Δ일 때 W = 1)
    W = (s_max // step) + 1 if s_max >= 0 else 0

    if W <= 0:
        raise ValueError(
            f"Invalid window count W={W}. Need T ≥ L + Δ. "
            f"(T={T}, L={L}, Δ={label_offset})"
        )

    # 입력 윈도우 전개: unfold로 모든 시작점 전개 후, Δ 고려하여 앞 W개만 사용
    # unfold 결과 W_all = 1 + floor((T - L) / step)
    Xw_all = X.unfold(dimension=2, size=L, step=step)  # [N, C, W_all, L]
    W_all = Xw_all.size(2)
    if W > W_all:
        # 이론상 W ≤ W_all이어야 하나, 수치 엣지 대비 방어
        W = W_all
    Xw = Xw_all[:, :, :W, :]  # Δ로 인해 뒤쪽 일부를 버림

    # 라벨 인덱스: t_i = s_i + (L - 1) + Δ, where s_i = i*step, i=0..W-1
    base = (L - 1) + label_offset
    starts = torch.arange(W, device=X.device) * step            # [W]
    label_idx = base + starts                                   # [W]
    if label_idx[-1].item() >= T:
        # 아주 드문 수치 엣지: label_idx가 T-1을 넘어가면 W를 조정
        valid = (label_idx < T)
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            raise ValueError(
                f"No room for labels with L={L}, Δ={label_offset}, T={T}, step={step}"
            )
        W = valid_count
        Xw = Xw[:, :, :W, :]
        label_idx = label_idx[:W]

    # ch0에서 라벨 뽑기 → [N, W]
    Yw_mat = X[:, 0, label_idx]  # [N, W]

    # 평탄화
    Xw = Xw.permute(0, 2, 1, 3).contiguous().view(N * W, C, L)  # [N*W, C, L]
    Yw = Yw_mat.contiguous().view(N * W)                        # [N*W]
    groups = torch.arange(N, device=X.device).repeat_interleave(W)  # [N*W]

    return Xw, Yw, groups, W
