# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: metrics.py                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  분류/회귀 지표 계산 및 τ(F1 최대) 탐색                           ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   metrics_from_probs(y_true, y_prob, threshold=0.5) -> dict               ║
║   find_best_threshold_for_f1(y_true, y_prob, step=0.001)                  ║
║     -> (best_tau:float, best_f1:float)                                    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ NOTES  MSE/MAE는 확률 vs 라벨(0/1) 기준 참고치                            ║
║ DEPENDENCY   pipeline/evaler → metrics; losses.compute_pos_weight 재노출   ║
╚════════════════════════════════════════════════════════════════════════════╝

metrics.py — 분류/회귀 지표 및 임계값 탐색
"""
import numpy as np, torch

@torch.no_grad()
def metrics_from_probs(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5, eps=1e-8):
    y_true = y_true.float().view(-1); y_prob = y_prob.float().view(-1)
    y_pred = (y_prob >= threshold).float()
    tp = (y_pred * y_true).sum().item()
    tn = ((1 - y_pred) * (1 - y_true)).sum().item()
    fp = (y_pred * (1 - y_true)).sum().item()
    fn = ((1 - y_pred) * y_true).sum().item()
    acc  = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(eps, (tp + fp))
    rec  = tp / max(eps, (tp + fn))
    f1   = 2 * prec * rec / max(eps, (prec + rec))
    mse  = torch.mean((y_prob - y_true) ** 2).item()
    mae  = torch.mean(torch.abs(y_prob - y_true)).item()
    return {"threshold": float(threshold), "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Accuracy": float(acc), "Precision": float(prec), "Recall": float(rec), "F1": float(f1),
            "MSE": float(mse), "MAE": float(mae)}

@torch.no_grad()
def find_best_threshold_for_f1(y_true: torch.Tensor, y_prob: torch.Tensor, step: float = 0.001):
    y_true = y_true.float().view(-1); y_prob = y_prob.float().view(-1)
    taus = np.arange(0.0, 1.0 + 1e-12, step, dtype=np.float32)
    best_tau, best_f1 = 0.5, -1.0
    for tau in taus:
        y_pred = (y_prob >= tau).float()
        tp = (y_pred * y_true).sum().item()
        fp = (y_pred * (1 - y_true)).sum().item()
        fn = ((1 - y_pred) * y_true).sum().item()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau, best_f1

# 재노출(파이프라인 편의)
from losses import compute_pos_weight_from_labels  # noqa: F401
