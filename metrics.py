# -*- coding: utf-8 -*-
"""
metrics.py — 분류/회귀 지표 및 임계값 탐색 (+ 확장 지표 유틸)
PUBLIC:
  - metrics_from_probs(y_true, y_prob, threshold=0.5) -> dict
  - find_best_threshold_for_f1(y_true, y_prob, step=0.001) -> (best_tau, best_f1)
ADDED (optional):
  - roc_auc_pr_auc(y_true, y_score) -> (auroc, auprc)
  - ks_statistic(y_true, y_score) -> ks
  - calibration_ece_mce(y_true, y_score, bins=15) -> (ece, mce, bin_stats)
  - classification_report_extended(y_true, y_score, threshold=0.5) -> dict
  - regression_metrics(y_true, y_pred, seasonal_period=None) -> dict
NOTE:
  * 모든 입력은 torch.Tensor/np.ndarray 모두 허용 (내부에서 torch→numpy 안전 변환)
  * 외부 의존성(sklearn 등) 없이 구현
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional

# ────────────────────────── helpers ──────────────────────────

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _safe_float(x, nd=None):
    try:
        v = float(x)
        if nd is None: return v
        return float(np.round(v, nd))
    except Exception:
        return float("nan")

# ────────────────────── core public (unchanged) ──────────────────────

@torch.no_grad()
def metrics_from_probs(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5, eps=1e-8) -> Dict[str, float]:
    """
    기존 인터페이스 유지. 확장 지표는 classification_report_extended에서 제공.
    """
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
    return {
        "threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": float(acc), "Precision": float(prec), "Recall": float(rec), "F1": float(f1),
        "MSE": float(mse), "MAE": float(mae)
    }

@torch.no_grad()
def find_best_threshold_for_f1(y_true: torch.Tensor, y_prob: torch.Tensor, step: float = 0.001) -> Tuple[float, float]:
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

# ────────────────────── added: classification extras ──────────────────────

@torch.no_grad()
def roc_auc_pr_auc(y_true, y_score) -> Tuple[float, float]:
    """
    AUROC, AUPRC (sklearn 없이 구현)
    """
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    # 정렬 (score 내림차순)
    order = np.argsort(-ys, kind="mergesort")
    yt = yt[order]; ys = ys[order]

    P = int((yt == 1).sum())
    N = int((yt == 0).sum())
    if P == 0 or N == 0:
        return float("nan"), float("nan")

    # ROC: 누적 TPR/FPR 점을 unique score에서만 적분 (계단형)
    tps = np.cumsum(yt == 1)
    fps = np.cumsum(yt == 0)
    # score이 변하는 지점(마지막 포함)
    idx = np.r_[np.where(np.diff(ys))[0], yt.size - 1]
    tpr = tps[idx] / P
    fpr = fps[idx] / N
    # 구간 사다리꼴 적분
    auroc = np.trapz(np.r_[0, tpr, 1], x=np.r_[0, fpr, 1])

    # PR: precision-recall 곡선
    tp = tps[idx]
    fp = fps[idx]
    fn = P - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    rec  = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    # PR AUC는 recall을 x축으로
    # 시작점 (rec=0, prec=1) 연결
    auprc = np.trapz(np.r_[1, prec], x=np.r_[0, rec])
    return float(auroc), float(auprc)

@torch.no_grad()
def ks_statistic(y_true, y_score) -> float:
    """
    Kolmogorov–Smirnov statistic for binary classification scores.
    """
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    order = np.argsort(ys)  # 오름차순
    ys = ys[order]; yt = yt[order]
    pos = max(1, int((yt == 1).sum()))
    neg = max(1, int((yt == 0).sum()))
    cum_pos = np.cumsum(yt == 1) / pos
    cum_neg = np.cumsum(yt == 0) / neg
    ks = np.max(np.abs(cum_pos - cum_neg)) if ys.size > 0 else float("nan")
    return float(ks)

@torch.no_grad()
def calibration_ece_mce(y_true, y_score, bins: int = 15):
    """
    Expected Calibration Error / Max Calibration Error (+ bin stats)
    반환: (ece, mce, {"conf":[], "acc":[], "count":[]})
    """
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    ys = np.clip(ys, 1e-9, 1 - 1e-9)
    bins = max(3, int(bins))
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(ys, edges) - 1
    confs, accs, counts = [], [], []
    for b in range(bins):
        m = (idx == b)
        if m.sum() == 0:
            confs.append((edges[b] + edges[b + 1]) / 2)
            accs.append(np.nan)
            counts.append(0)
        else:
            confs.append(ys[m].mean())
            accs.append((yt[m] == 1).mean())
            counts.append(int(m.sum()))
    confs = np.asarray(confs); accs = np.asarray(accs); counts = np.asarray(counts)
    valid = counts > 0
    if valid.sum() == 0:
        return float("nan"), float("nan"), {"conf": confs.tolist(), "acc": accs.tolist(), "count": counts.tolist()}
    ece = np.sum((counts[valid] / counts[valid].sum()) * np.abs(accs[valid] - confs[valid]))
    mce = np.max(np.abs(accs[valid] - confs[valid]))
    return float(ece), float(mce), {"conf": confs.tolist(), "acc": accs.tolist(), "count": counts.tolist()}

@torch.no_grad()
def classification_report_extended(y_true, y_score, threshold: float = 0.5, eps: float = 1e-8) -> Dict[str, Any]:
    """
    확장 분류 리포트: 기존 metrics_from_probs + AUROC/AUPRC/KS/MCC/BalAcc/Specificity/Brier/ECE/MCE
    """
    yt = torch.as_tensor(y_true).float().view(-1)
    ys = torch.as_tensor(y_score).float().view(-1)
    # base
    base = metrics_from_probs(yt, ys, threshold=threshold, eps=eps)
    # extras
    auroc, auprc = roc_auc_pr_auc(yt, ys)
    ks = ks_statistic(yt, ys)
    # specificity, bal acc, mcc, brier
    ypred = (ys >= threshold).float()
    tp = float((ypred * yt).sum().item())
    tn = float(((1 - ypred) * (1 - yt)).sum().item())
    fp = float((ypred * (1 - yt)).sum().item())
    fn = float(((1 - ypred) * yt).sum().item())
    tpr = tp / max(eps, tp + fn)
    tnr = tn / max(eps, tn + fp)  # specificity
    bal_acc = 0.5 * (tpr + tnr)
    denom = max(eps, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom
    brier = torch.mean((ys - yt) ** 2).item()
    ece, mce, bins = calibration_ece_mce(yt, ys, bins=15)
    base.update({
        "AUROC": _safe_float(auroc), "AUPRC": _safe_float(auprc), "KS": _safe_float(ks),
        "Specificity": _safe_float(tnr), "BalancedAccuracy": _safe_float(bal_acc),
        "MCC": _safe_float(mcc), "Brier": _safe_float(brier),
        "ECE": _safe_float(ece), "MCE": _safe_float(mce),
        "CalibrationBins": bins
    })
    return base

# ────────────────────────── regression extras ──────────────────────────

@torch.no_grad()
def regression_metrics(y_true, y_pred, seasonal_period: Optional[int] = None) -> Dict[str, Any]:
    """
    회귀 공통 지표: MAE/MSE/RMSE/R2/MAPE/sMAPE/MASE (seasonal_period 제공 시)
    """
    yt = torch.as_tensor(y_true).float().view(-1)
    yp = torch.as_tensor(y_pred).float().view(-1)
    n = int(yt.numel())
    mae = torch.mean(torch.abs(yp - yt)).item()
    mse = torch.mean((yp - yt) ** 2).item()
    rmse = float(np.sqrt(mse))
    # R^2
    ybar = float(torch.mean(yt).item())
    ss_tot = float(torch.sum((yt - ybar) ** 2).item())
    ss_res = float(torch.sum((yp - yt) ** 2).item())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # MAPE / sMAPE
    yt_np = yt.detach().cpu().numpy()
    yp_np = yp.detach().cpu().numpy()
    denom = np.maximum(np.abs(yt_np), 1e-8)
    mape = float(np.mean(np.abs((yp_np - yt_np) / denom)) * 100.0)
    smape = float(np.mean(2.0 * np.abs(yp_np - yt_np) / np.maximum(np.abs(yt_np) + np.abs(yp_np), 1e-8)) * 100.0)
    # MASE (seasonal naive 대비)
    mase = float("nan")
    if seasonal_period is not None and seasonal_period > 0 and n > seasonal_period:
        naive = yt_np[seasonal_period:]
        prev  = yt_np[:-seasonal_period]
        q = np.mean(np.abs(naive - prev))
        mase = float(np.mean(np.abs(yp_np - yt_np)) / max(q, 1e-8))
    return {
        "MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse),
        "R2": float(r2), "MAPE": float(mape), "sMAPE": float(smape), "MASE": float(mase)
    }
