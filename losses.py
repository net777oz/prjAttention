# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: losses.py                                                           ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  분류 손실(SoftF1, BCE+pos_weight) 및 pos_weight 헬퍼             ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   class SoftF1Loss(nn.Module)                                             ║
║   build_bce(pos_weight_mode:str, device, global_pos_weight:float|None)    ║
║      -> (bce_loss_fn, pos_weight_tensor|None)                             ║
║   compute_pos_weight_from_labels(y_bin:Tensor) -> float                   ║
╠───────────────────────────────────────────────────────────────────────────╣
║ SIDE EFFECTS  없음                                                         ║
║ DEPENDENCY   trainer/metrics(재노출) → losses                              ║
╚════════════════════════════════════════════════════════════════════════════╝

losses.py — 분류용 손실 및 pos_weight 도우미
"""
import torch, torch.nn.functional as F

class SoftF1Loss(torch.nn.Module):
    """1 - SoftF1 (배치·채널 합 기준)"""
    def __init__(self, eps=1e-8):
        super().__init__(); self.eps = eps
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        p = torch.sigmoid(logits); y = targets
        tp = (p * y).sum(); fp = (p * (1 - y)).sum(); fn = ((1 - p) * y).sum()
        soft_f1 = (2 * tp) / (2 * tp + fp + fn + self.eps)
        return 1.0 - soft_f1

def build_bce(pos_weight_mode:str, device, global_pos_weight:float|None):
    if pos_weight_mode == "none":
        return torch.nn.BCEWithLogitsLoss(reduction="mean"), None
    pw = torch.tensor([global_pos_weight if global_pos_weight is not None else 1.0],
                      dtype=torch.float32, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="mean"), pw

def compute_pos_weight_from_labels(y_bin: torch.Tensor) -> float:
    pos = float((y_bin > 0.5).sum()); neg = float((y_bin <= 0.5).sum())
    return (neg / pos) if pos > 0 else 1.0
