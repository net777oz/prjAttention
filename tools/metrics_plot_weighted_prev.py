# -*- coding: utf-8 -*-
"""
각 ctx의 artifacts/{run}/preds.csv 를 읽어, ctx=1의 prevalence를 기준(p_ref)으로
샘플링 없이 음성(y_true=0)만 가중치 w0를 곱해 가중 지표를 계산/플롯.

- w0 = ((1-p_ref)/p_ref) * (p_k/(1-p_k))
- 가중치는 라벨 기준(예측과 무관): y_true==1 → 1, y_true==0 → w0
- τ는 기본 0.5 또는 metrics.json의 best τ 사용 가능(--use-best-tau)
- 스무딩 옵션: --smooth ma --smooth-window 5 (이동평균)
"""

import os, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = "./artifacts"

def find_run_dir_for_ctx(ctx: int):
    pats = [
        os.path.join(ART, f"train_classify_ctx{ctx}_*"),
        os.path.join(ART, f"train_classify_ctx{ctx:02d}_*"),
    ]
    cands = []
    for pat in pats:
        for d in glob.glob(pat):
            p = os.path.join(d, "preds.csv")
            if os.path.isfile(p):
                cands.append((os.path.getmtime(p), d))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def load_preds_metrics(run_dir: str):
    p_pred = os.path.join(run_dir, "preds.csv")
    p_met  = os.path.join(run_dir, "metrics.json")
    df = pd.read_csv(p_pred)
    y_true = df["y_true"].astype(int).values
    y_score = df["y_score"].astype(float).values
    tau_best = None
    if os.path.isfile(p_met):
        try:
            with open(p_met, "r", encoding="utf-8") as f:
                m = json.load(f)
            tau_best = float(m.get("threshold_best", None))
        except Exception:
            pass
    return y_true, y_score, tau_best

def binarize(y_score, tau):
    return (np.asarray(y_score) >= float(tau)).astype(int)

def weighted_metrics(y_true, y_pred, w0: float):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    w = np.where(y_true == 1, 1.0, float(w0))  # 라벨 기준 가중치

    tp = float(np.sum(((y_true == 1) & (y_pred == 1)) * w))
    tn = float(np.sum(((y_true == 0) & (y_pred == 0)) * w))
    fp = float(np.sum(((y_true == 0) & (y_pred == 1)) * w))
    fn = float(np.sum(((y_true == 1) & (y_pred == 0)) * w))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    acc  = (tp + tn) / max(1.0, (tp + tn + fp + fn))
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc,
                tp=tp, tn=tn, fp=fp, fn=fn)

def safe_w0(p_ref, p_k):
    eps = 1e-12
    p_ref = np.clip(p_ref, eps, 1 - eps)
    p_k   = np.clip(p_k,   eps, 1 - eps)
    return ((1 - p_ref) / p_ref) * (p_k / (1 - p_k))

def smooth_series(arr, method="none", window=5):
    x = np.asarray(arr, dtype=float)
    n = len(x)
    if method not in ("ma", "none") or n == 0:
        return x
    if method == "none" or window <= 1:
        return x
    # 이동평균: 가장자리 edge-pad 후 same 길이 유지
    w = int(window)
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    xpad = np.pad(x, (pad_l, pad_r), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(xpad, kernel, mode="valid")
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=31)
    ap.add_argument("--tau", type=float, default=None, help="고정 임계값(없으면 --use-best-tau 또는 0.5)")
    ap.add_argument("--use-best-tau", action="store_true", help="metrics.json의 best τ 사용")
    ap.add_argument("--out-csv", type=str, default="metrics_weighted_prev.csv")
    ap.add_argument("--out-png", type=str, default="metrics_weighted_prev.png")
    # smoothing
    ap.add_argument("--smooth", type=str, default="none", choices=["none","ma"], help="곡선 스무딩 방식")
    ap.add_argument("--smooth-window", type=int, default=5, help="이동평균 창 크기(홀수 권장)")
    ap.add_argument("--hide-raw", action="store_true", help="raw 곡선 숨기기")
    args = ap.parse_args()

    rows = []
    missing = []

    # 1) 기준(ctx=1) prevalence
    rd1 = find_run_dir_for_ctx(1)
    if not rd1:
        raise SystemExit("[ERR] ctx=1 preds.csv를 찾을 수 없습니다.")
    y_true_1, y_score_1, tau_best_1 = load_preds_metrics(rd1)
    if y_true_1.size == 0:
        raise SystemExit("[ERR] ctx=1 preds.csv가 비어 있습니다.")
    p_ref = float((y_true_1 == 1).mean())
    print(f"[INFO] p_ref(ctx=1) = {p_ref:.6f}")

    # 2) ctx별 가중 지표
    for k in range(args.start, args.end+1 - 10):
        rd = find_run_dir_for_ctx(k)
        if not rd:
            missing.append(k); continue
        y_true, y_score, tau_best = load_preds_metrics(rd)
        if y_true.size == 0:
            missing.append(k); continue

        p_k = float((y_true == 1).mean())
        w0 = safe_w0(p_ref, p_k)

        # τ
        if args.tau is not None:
            tau_used = float(args.tau)
        elif args.use_best_tau and (tau_best is not None):
            tau_used = float(tau_best)
        else:
            tau_used = 0.5

        y_pred = binarize(y_score, tau_used)
        met = weighted_metrics(y_true, y_pred, w0)
        rows.append({
            "ctx": k,
            "run": os.path.basename(rd),
            "p_ref": p_ref,
            "p_ctx": p_k,
            "w0": w0,
            "tau_used": tau_used,
            **met
        })

    df = pd.DataFrame(rows).sort_values("ctx")
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] 저장: {args.out_csv}")

    # 3) 플롯
    if not df.empty:
        x = df["ctx"].values.astype(float)
        series = {
            "Precision (weighted)": df["precision"].values,
            "Recall (weighted)":    df["recall"].values,
            "F1 (weighted)":        df["f1"].values,
            "Accuracy (weighted)":  df["accuracy"].values,
        }
        plt.figure(figsize=(12,6))
        for name, y in series.items():
            y_s = smooth_series(y, method=args.smooth, window=args.smooth_window)
            if not args.hide_raw:
                plt.plot(x, y, linestyle="--", linewidth=1.0, alpha=0.5, label=f"{name} (raw)")
            plt.plot(x, y_s, marker="o", linewidth=2.0, label=name)
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xlabel("context length (ctx)")
        plt.ylabel("metric")
        plt.title(f"Weighted-by-Label(neg) Metrics, anchored to ctx=1 (p_ref={p_ref:.3f})")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=150)
        print(f"[OK] 저장: {args.out_png}")

    if missing:
        print("[INFO] preds.csv 없는 ctx:", ", ".join(map(str, missing)))

if __name__ == "__main__":
    main()
