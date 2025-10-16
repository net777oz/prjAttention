# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: windows.py                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  롤링 윈도우(Xw,Yw,groups,W) 생성                                 ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   build_windows_dataset(X:[N,C,T], L:int, label_offset:int=1, step:int=1) ║
║     -> (Xw:[N*W,C,L], Yw:[N*W], groups:[N*W], W:int)                      ║
╠───────────────────────────────────────────────────────────────────────────╣
║ NOTES                                                                      ║
║  • 기본값은 Δ=+1 (윈도우 바로 다음 시점 라벨) → 기존 코드와 동작 동일         ║
║  • Δ를 바꾸고 싶으면 label_offset=0(동시시점) 등으로 호출 가능                ║
║  • 모든 소스는 동일한 T를 가진다고 가정하며, 라벨은 항상 ch0(main) 기준       ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Tuple
import torch


def build_windows_dataset(
    X: torch.Tensor,
    L: int,
    label_offset: int = 1,   # Δ=+1 (기본: 기존 코드와 동일)
    step: int = 1,           # 슬라이딩 간격
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    X: [N, C, T]
      - C: 채널/CSV 수 (라벨은 항상 ch0에서 생성)
    반환:
      Xw:     [N*W, C, L]     입력 윈도우
      Yw:     [N*W]           라벨(1D, ch0 기준)
      groups: [N*W]           원본 row 인덱스
      W:      int             윈도우 개수 (T - (L + Δ))

    동작 개요:
      • 각 row에 대해 시작 s = 0..W-1
      • 입력: X[:, :, s : s+L]
      • 라벨: X[:, 0,  s+L+Δ-1+1] == X[:, 0, s+L+Δ] (end-exclusive 관점)
             → 구현은 벡터 슬라이스로 X[:, 0:1, L+Δ : L+Δ+W]
    """
    # 기본 형태 체크
    if X.dim() != 3:
        raise ValueError(f"X must be [N,C,T], got {tuple(X.shape)}")
    N, C, T = X.shape

    if label_offset < 0 or label_offset >= T:
        raise ValueError(f"label_offset({label_offset}) out of range for T={T}")

    # Δ를 반영한 L 보정: L ∈ [1, T-Δ]
    L = max(1, min(L, T - label_offset))

    # 윈도우 개수: s+L+Δ-1 < T  ⇔  s ≤ T-(L+Δ)-1  ⇔  W = T - (L+Δ)
    W = T - (L + label_offset)
    if W <= 0:
        raise ValueError(
            f"Invalid window count W={W}. Need T ≥ L + Δ. "
            f"(T={T}, L={L}, Δ={label_offset})"
        )

    # 입력 윈도우 전개: [N, C, (T-L+1), L]
    Xw_full = X.unfold(dimension=2, size=L, step=step)
    # 안전장치: unfold로 나온 총 윈도우 수가 기대 이상인지 확인
    # (step>1에서도 동작하도록 Xw_full.size(2) >= W 보장만 체크)
    if Xw_full.size(2) < W:
        # step>1인 경우 마지막 일부 윈도우가 잘릴 수 있으므로 W를 재산정
        W = Xw_full.size(2)
    # Δ를 고려해 뒤쪽 윈도우를 잘라 Δ만큼의 레이블 여유를 확보
    Xw = Xw_full[:, :, :W, :]                         # [N, C, W, L]

    # 라벨 슬라이스: ch0에서 L+Δ ~ L+Δ+W-1
    start_label = L + label_offset
    end_label = start_label + W
    if end_label > T:
        # 이 경우도 실질적으로 W를 줄여 정합을 맞춤
        W = T - start_label
        if W <= 0:
            raise ValueError(
                f"No room for labels: start_label={start_label}, T={T}"
            )
        Xw = Xw[:, :, :W, :]
        end_label = start_label + W

    Yw_full = X[:, 0:1, start_label:end_label]        # [N, 1, W]

    # 추가 안전장치
    if Xw.shape[2] != Yw_full.shape[2]:
        raise RuntimeError(
            f"window/label count mismatch: Xw.W={Xw.shape[2]} vs Yw.W={Yw_full.shape[2]}"
        )
    if Xw.shape[-1] != L:
        raise RuntimeError(
            f"window length mismatch: got {Xw.shape[-1]} vs L={L}"
        )

    # 배치 평탄화
    Xw = Xw.permute(0, 2, 1, 3).contiguous().view(N * W, C, L)   # [N*W, C, L]
    Yw = Yw_full.permute(0, 2, 1).contiguous().view(N * W)       # [N*W]

    # 그룹 인덱스
    groups = torch.arange(N, device=X.device).repeat_interleave(W)  # [N*W]

    return Xw, Yw, groups, W
