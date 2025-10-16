# -*- coding: utf-8 -*-
"""
metrics_writer.py — y_true(0/1)와 y_score(확률)에서 핵심 지표를 계산해
out_dir/metrics.json(+ 선택: preds.csv)로 저장합니다.

의존성: numpy(필수), 표준 라이브러리만 사용
사용:
    from tools.metrics_writer import write_metrics_json
    write_metrics_json(y_true, y_score, out_dir, save_preds=True)
"""
import json, os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

def _auc(x: np.ndarray, y: np.ndarray) -> float:
    # x가 증가하도록 정렬 가정
    return float(np.trapz(y, x)) if x.size > 1 else float(0.0)

def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # y_score 내림차순 정렬
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    # 양성/음성 수
    P = float(np.sum(y_true == 1))
    N = float(np.sum(y_true == 0))
    if P == 0 or N == 0:
        # 한쪽 클래스만 있는 경우 안전 처리
        tpr = np.array([0.0, 1.0]) if P > 0 else np.array([0.0, 0.0])
        fpr = np.array([0.0, 1.0]) if N > 0 else np.array([0.0, 0.0])
        thr = np.array([1.0, 0.0])
        return fpr, tpr, thr

    # 임계값마다 누적 TP/FP
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    fns = P - tps
    tns = N - fps

    tpr = tps / P
    fpr = fps / N
    thr = y_score[threshold_idxs]
    # 시작점(0,0)과 끝점(1,1) 보강
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    thr = np.r_[thr[0] + 1e-12, thr, thr[-1] - 1e-12]
    return fpr, tpr, thr

def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    tp = np.cumsum(y_true == 1).astype(float)
    fp = np.cumsum(y_true == 0).astype(float)
    fn = float(np.sum(y_true == 1)) - tp

    precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp), where=(tp+fp) > 0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp+fn) > 0)

    # (0,1) 보강: recall=0일 때 precision=1로 정의
    precision = np.r_[1.0, precision]
    recall    = np.r_[0.0, recall]
    thresholds= y_score
    return precision, recall, thresholds

def _bin_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = float(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    acc  = (tp + tn) / max(1.0, (tp + tn + fp + fn))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, accuracy=acc)

def _best_threshold(y_true: np.ndarray, y_score: np.ndarray, strategy: str = "f1") -> Tuple[float, Dict[str, float]]:
    # 여러 기준 중 최댓값을 주는 threshold 선택
    thresholds = np.linspace(0.0, 1.0, 1001)  # 0.001 step
    best_thr, best_val, best_stat = 0.5, -1.0, {}
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        m = _bin_metrics(y_true, y_pred)
        val = (
            m["f1"] if strategy == "f1"
            else (m["recall"] + (1 - (m["fp"]/max(1.0, m["fp"]+m["tn"]))))/2  # 예: 재현율-특이도 절충
        )
        if val > best_val:
            best_val, best_thr, best_stat = val, float(thr), m
    return best_thr, best_stat

def write_metrics_json(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir: str,
    default_threshold: float = 0.5,
    strategy_for_best: str = "f1",
    save_preds: bool = False,
    preds_name: str = "preds.csv",
    extra_meta: Optional[dict] = None,
) -> Dict:
    """
    y_true: (N,) 0/1
    y_score: (N,) 0~1 확률
    out_dir/metrics.json 저장. (선택) out_dir/preds.csv 저장.
    """
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int).flatten()
    y_score= np.asarray(y_score).astype(float).flatten()
    assert y_true.shape == y_score.shape, "y_true, y_score 길이가 다릅니다."

    # 곡선/면적
    fpr, tpr, thr_roc = _roc_curve(y_true, y_score)
    auroc = _auc(fpr, tpr)
    precision, recall, thr_pr = _precision_recall_curve(y_true, y_score)
    # PR-AUC은 recall을 x축으로 정렬 필요
    order = np.argsort(recall)
    auprc = _auc(recall[order], precision[order])

    # threshold 기반 통계
    thr0 = float(default_threshold)
    m0 = _bin_metrics(y_true, (y_score >= thr0).astype(int))
    thr_best, m_best = _best_threshold(y_true, y_score, strategy=strategy_for_best)

    preval = float(np.mean(y_true == 1))

    metrics = {
        "counts": {
            "n": int(y_true.size),
            "pos": int(np.sum(y_true==1)),
            "neg": int(np.sum(y_true==0)),
        },
        "prevalence": preval,
        "roc_auc": auroc,
        "pr_auc": auprc,
        "threshold_default": thr0,
        "metrics_at_default": m0,
        "threshold_best": thr_best,
        "metrics_at_best": m_best,
        "notes": {
            "best_strategy": strategy_for_best,
            "comment": "threshold_best는 지정된 전략의 점수가 최대가 되는 임계값입니다.",
        },
    }
    if extra_meta:
        metrics["meta"] = extra_meta

    # 저장
    (outp / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 선택: 예측 원본도 저장
    if save_preds:
        import csv
        with (outp / preds_name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y_true", "y_score"])
            for yt, ys in zip(y_true, y_score):
                w.writerow([int(yt), float(ys)])

    return metrics
