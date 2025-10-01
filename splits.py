# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: splits.py                                                           ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  split-mode 구현(group/item/time/window)                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   make_splits(Xw, Yw, groups, N, T, L, W, mode, val_ratio, seed)          ║
║     -> (train_index:list, val_index:list)                                  ║
╠───────────────────────────────────────────────────────────────────────────╣
║ SIDE EFFECTS  없음                                                         ║
║ EXCEPTIONS   sklearn 미존재 시 item split으로 폴백                         ║
║ DEPENDENCY   pipeline → splits                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

splits.py — 데이터 분할 로직
"""
import math, numpy as np, torch
from typing import List, Tuple

def make_splits(Xw_all: torch.Tensor, Yw_all: torch.Tensor, groups: torch.Tensor,
                N:int, T:int, L:int, W:int, mode:str, val_ratio:float, seed:int) -> Tuple[list, list]:
    """
    mode:
      - window: 윈도우 전체에서 랜덤 split (누수 가능)
      - item: 아이템(행) 단위 split → 각 집합의 윈도우만 선택
      - time: 각 아이템별 윈도우 축을 앞/뒤로 분할 (겹침 없음)
      - group: GroupShuffleSplit (기본)
    """
    g = torch.Generator().manual_seed(seed)

    if mode == "window":
        total = Xw_all.shape[0]
        idx = torch.randperm(total, generator=g)
        val_size = int(total * val_ratio)
        val_idx = idx[:val_size]; trn_idx = idx[val_size:]
        return trn_idx.tolist(), val_idx.tolist()

    if mode == "item":
        n_items = N
        n_val = int(max(1, round(n_items * val_ratio)))
        perm = torch.randperm(n_items, generator=g)
        val_items = set(perm[:n_val].tolist()); trn_items = set(perm[n_val:].tolist())
        trn_idx = [i for i, gg in enumerate(groups.tolist()) if gg in trn_items]
        val_idx = [i for i, gg in enumerate(groups.tolist()) if gg in val_items]
        return trn_idx, val_idx

    if mode == "time":
        trn_idx, val_idx = [], []
        cutoff_w = int(max(1, math.floor(W * (1 - val_ratio))))
        for k in range(N):
            start = k * W
            trn_idx.extend(range(start, start + cutoff_w))
            val_idx.extend(range(start + cutoff_w, start + W))
        return trn_idx, val_idx

    # group (기본)
    try:
        from sklearn.model_selection import GroupShuffleSplit
    except Exception as e:
        print(f"[WARN] sklearn not available ({e}); fallback to item split.")
        return make_splits(Xw_all, Yw_all, groups, N, T, L, W, mode="item", val_ratio=val_ratio, seed=seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    trn_idx, val_idx = next(gss.split(np.zeros(len(groups)), np.zeros(len(groups)), groups.numpy()))
    return trn_idx.tolist(), val_idx.tolist()
