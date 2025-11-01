# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: windows.py                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  롤링 윈도우(Xw, Yw, groups, W) 생성                               ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   build_windows_dataset(X:[N,C,T], L:int, label_offset:int=1, step:int=1  ║
║                        , label_src_ch:int=0, exclude_channels=None)       ║
║     -> (Xw:[N*W,C_eff,L], Yw:[N*W], groups:[N*W], W:int)                  ║
╠───────────────────────────────────────────────────────────────────────────╣
║ NOTES                                                                     ║
║  • 기본 Δ=+1 → 윈도우 끝(t = s+L-1)의 다음 시점(t+Δ)을 라벨로 사용            ║
║  • Δ 변경은 label_offset으로 설정(예: 동시시점 라벨이면 Δ=0)                 ║
║  • 모든 row가 동일한 T를 가진다고 가정                                     ║
║  • 라벨은 label_src_ch에서 추출, 필요 시 입력특징에서는 해당 채널 제외        ║
║  • ENV 지원:                                                                ║
║     - AP_LABEL_SRC_CH, AP_DROP_LABEL_FROM_X(1/0)                           ║
║    (함수 인자가 주어지지 않으면 ENV를 읽어 동일 동작을 보장)                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Tuple, Optional, Iterable, Set
import os
import torch

def _env_label_src_ch(default: int = 0) -> int:
    try:
        return int(os.environ.get("AP_LABEL_SRC_CH", str(default)))
    except Exception:
        return default

def _env_drop_flag() -> bool:
    return os.environ.get("AP_DROP_LABEL_FROM_X", "0") == "1"

def _normalize_exclude(exclude_channels: Optional[Iterable[int]]) -> Set[int]:
    if exclude_channels is None:
        return set()
    return set(int(c) for c in exclude_channels)

def build_windows_dataset(
    X: torch.Tensor,
    L: int,
    label_offset: int = 1,   # Δ (기본: 다음 시점)
    step: int = 1,           # 윈도우 시작 간격
    label_src_ch: Optional[int] = None,
    exclude_channels: Optional[Iterable[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    입력:
      X: [N, C, T]  (torch.Tensor)
      L: 윈도우 길이 (context_len)
      label_offset(Δ): 윈도우 끝에서 Δ 스텝 뒤 시점을 라벨로 사용 (기본 1)
      step: 시작 인덱스 간격 (기본 1)
      label_src_ch: 라벨 소스 채널(기본 None→ENV→0)
      exclude_channels: 입력 특징에서 제외할 채널 목록(기본 None→ENV drop이면 {label_src_ch})

    반환:
      Xw:     [N*W, C_eff, L]  (라벨 채널 제외 반영)
      Yw:     [N*W]            (float/long은 후단에서 변환)
      groups: [N*W]            (row grouping id)
      W:      int              윈도우 개수
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

    # ENV/인자 병합
    if label_src_ch is None:
        label_src_ch = _env_label_src_ch(0)
    if not (0 <= label_src_ch < C):
        label_src_ch = 0  # 안전 클램프

    exclude = _normalize_exclude(exclude_channels)
    if not exclude and _env_drop_flag():
        exclude = {label_src_ch}

    # L 보정(동작 보장)
    if L > T - label_offset:
        L = max(1, T - label_offset)

    # 시작 인덱스의 최대값 s_max = T - (L + Δ)
    s_max = T - (L + label_offset)
    W = (s_max // step) + 1 if s_max >= 0 else 0
    if W <= 0:
        raise ValueError(
            f"Invalid window count W={W}. Need T ≥ L + Δ. "
            f"(T={T}, L={L}, Δ={label_offset})"
        )

    # 라벨 소스 시계열 (원본 X에서 분리)
    # 라벨 인덱스: t_i = s_i + (L - 1) + Δ, where s_i = i*step, i=0..W-1
    base = (L - 1) + label_offset
    starts = torch.arange(W, device=X.device) * step            # [W]
    label_idx = base + starts                                   # [W]
    if label_idx[-1].item() >= T:
        valid = (label_idx < T)
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            raise ValueError(
                f"No room for labels with L={L}, Δ={label_offset}, T={T}, step={step}"
            )
        W = valid_count
        label_idx = label_idx[:W]
    Yw_mat = X[:, label_src_ch, label_idx]  # [N, W]

    # 입력 특징 채널 선택(필요 시 라벨 채널 제외)
    if exclude:
        keep = [c for c in range(C) if c not in exclude]
        if len(keep) == 0:
            raise ValueError(f"All channels excluded ({sorted(list(exclude))}). Need at least one feature channel.")
        X_feat = X[:, keep, :]
        C_eff = len(keep)
    else:
        X_feat = X
        C_eff = C

    # 입력 윈도우 전개 (특징만 전개)
    Xw_all = X_feat.unfold(dimension=2, size=L, step=step)  # [N, C_eff, W_all, L]
    W_all = Xw_all.size(2)
    if W > W_all:
        W = W_all
    Xw = Xw_all[:, :, :W, :]

    # 평탄화
    Xw = Xw.permute(0, 2, 1, 3).contiguous().view(N * W, C_eff, L)  # [N*W, C_eff, L]
    Yw = Yw_mat.contiguous().view(N * W)                             # [N*W]
    groups = torch.arange(N, device=X.device).repeat_interleave(W)   # [N*W]

    return Xw, Yw, groups, W
