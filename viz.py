# -*- coding: utf-8 -*-
"""
viz.py — 모든 플롯은 여기서만 생성 (중복 생성 없음, 레거시 호출 호환)
- 저장 위치: AP_OUT_DIR/plots/after_<split>_<name>.png
  (AP_OUT_DIR 없으면 ./artifacts/default_run/plots/...)
- 제공 함수:
  - plot_training_curve(train_losses, val_losses=None, *args, **kwargs)
  - plot_cls_curves(y_true, y_score, *args, **kwargs) -> dict(paths)
  - plot_roc(y_true, y_score, split)
  - plot_pr(y_true, y_score, split)
  - plot_score_hists(y_true, y_score, split)
  - plot_confusion(y_true_bin, y_pred_bin, split)
  - plot_calibration(y_true, y_score, split, bins=15)
  - plot_ks(y_true, y_score, split)
  - plot_cost_curve(y_true, y_score, split, fp_cost=1.0, fn_cost=1.0, steps=200)
  - plot_lift_gain(y_true, y_score, split, steps=20)
  - plot_residuals(y_true, y_pred, split)
  - plot_residual_qq(y_true, y_pred, split)
  - plot_residual_time(y_true, y_pred, split)
  - plot_pred_vs_true(y_true, y_pred, split)
  - plot_latency_bar(latency_dict, split)
  - plot_samples(X, y_true=None, y_pred=None, split='val', max_samples=3, ch=0, title='Samples')
"""
import os
from pathlib import Path
import numpy as np

# ─────────────────────────── path helpers ───────────────────────────

def _run_root() -> Path:
    base = os.environ.get("AP_OUT_DIR")
    if base:
        return Path(base).expanduser().resolve()
    return Path("./artifacts").resolve() / "default_run"

def _plots_dir() -> Path:
    p = _run_root() / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _savefig(fig, split: str, name: str):
    out = _plots_dir() / f"after_{split}_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    return out

def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x) if x is not None else None

def _looks_numeric_series(x):
    if x is None: return False
    try:
        arr = _to_np(x)
        if arr is None: return False
        arr = np.asarray(arr)
        return arr.dtype.kind in ("i", "u", "f")
    except Exception:
        return False

def _compat_resolve_split_tau(args, kwargs, default_split="val", default_tau=0.5):
    """
    지원 패턴:
      - (split='val', tau=0.5)
      - ("val",)
      - (0.5, ...)
      - (tau, out_dir, prefix='before_val')
    반환: (split, tau)
    """
    split = kwargs.pop("split", None)
    tau = kwargs.pop("tau", None)
    prefix = kwargs.get("prefix")  # 저장명엔 사용 안 함

    if len(args) == 1:
        a0 = args[0]
        if isinstance(a0, str):
            split = a0
        elif isinstance(a0, (int, float)):
            tau = float(a0)
    elif len(args) >= 2:
        a0, a1 = args[0], args[1]
        if isinstance(a0, (int, float)):
            tau = float(a0)
        elif isinstance(a0, str):
            split = a0
            if isinstance(a1, (int, float)):
                tau = float(a1)

    if (not split) and isinstance(prefix, str) and "_" in prefix:
        split = prefix.split("_")[-1]

    if not split:
        split = default_split
    if tau is None:
        tau = default_tau
    return split, float(tau)

# ─────────────────────────── training curve ───────────────────────────

def plot_training_curve(train_losses, val_losses=None, *args, **kwargs):
    """
    저장: plots/after_<split>_loss.png
    """
    if not _looks_numeric_series(val_losses):
        val_losses = None
    args_list = list(args)
    if val_losses is None and len(args_list) > 0 and _looks_numeric_series(args_list[0]):
        val_losses = args_list[0]
        args_list = args_list[1:]
    split, _ = _compat_resolve_split_tau(tuple(args_list), kwargs)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tr = _to_np(train_losses).astype(float).ravel() if train_losses is not None else None
    va = _to_np(val_losses).astype(float).ravel() if val_losses is not None else None

    fig = plt.figure()
    if tr is not None and tr.size > 0:
        plt.plot(np.arange(1, tr.size+1), tr, label="train")
    if va is not None and va.size > 0:
        plt.plot(np.arange(1, va.size+1), va, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title(kwargs.get("title", "Training Curve"))
    plt.grid(True, alpha=0.3)
    if (tr is not None and tr.size>0) and (va is not None and va.size>0):
        plt.legend(loc="best")
    return _savefig(fig, split, "loss")

# ─────────────────────────── classification helpers ───────────────────────────

def plot_cls_curves(y_true, y_score, *args, **kwargs):
    """
    저장물:
      after_<split>_roc.png
      after_<split>_pr.png
      after_<split>_hist_all.png (+ _pos/_neg)
      after_<split>_confusion.png   (τ 기준)
    """
    split, tau = _compat_resolve_split_tau(args, kwargs)
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    out = {}
    out["roc"] = str(plot_roc(yt, ys, split))
    out["pr"]  = str(plot_pr(yt, ys, split))
    out["hist_all"] = str(plot_score_hists(yt, ys, split))
    yp = (ys >= float(tau)).astype(int)
    out["confusion"] = str(plot_confusion(yt, yp, split))
    return out

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from itertools import accumulate
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    order = np.argsort(-ys); yt=yt[order]; ys=ys[order]
    P = int((yt==1).sum()); N = int((yt==0).sum())
    fig = plt.figure()
    if P==0 or N==0:
        plt.plot([0,1],[0,1],"--"); plt.title("ROC (degenerate)")
    else:
        tps = np.array(list(accumulate((yt==1).astype(int))))
        fps = np.array(list(accumulate((yt==0).astype(int))))
        idx = np.r_[np.where(np.diff(ys))[0], yt.size-1]
        tpr = tps[idx]/max(P,1); fpr = fps[idx]/max(N,1)
        plt.plot([0,1],[0,1],"--"); plt.plot(np.r_[0,fpr,1], np.r_[0,tpr,1])
        plt.title("ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(True, alpha=0.3)
    return _savefig(fig, split, "roc")

def plot_pr(y_true: np.ndarray, y_score: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    order = np.argsort(-ys); yt=yt[order]
    P = max(int((yt==1).sum()), 1)
    tp = np.cumsum(yt==1); fp = np.cumsum(yt==0); fn = P - tp
    prec = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)>0)
    rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)>0)
    import matplotlib.pyplot as plt
    fig = plt.figure(); plt.plot(np.r_[0,rec], np.r_[1,prec])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR"); plt.grid(True, alpha=0.3)
    return _savefig(fig, split, "pr")

def plot_score_hists(y_true: np.ndarray, y_score: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    fig = plt.figure(); plt.hist(ys, bins=50, alpha=0.85); plt.title("Score Histogram (All)")
    p_all = _savefig(fig, split, "hist_all")
    pos = ys[yt==1]; neg = ys[yt==0]
    if pos.size>0:
        fig = plt.figure(); plt.hist(pos, bins=50, alpha=0.9); plt.title("Score Histogram (Positives)")
        _savefig(fig, split, "hist_pos")
    if neg.size>0:
        fig = plt.figure(); plt.hist(neg, bins=50, alpha=0.9); plt.title("Score Histogram (Negatives)")
        _savefig(fig, split, "hist_neg")
    return p_all

def plot_confusion(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true_bin).astype(int).ravel()
    yp = _to_np(y_pred_bin).astype(int).ravel()
    tn = int(((yt==0)&(yp==0)).sum())
    fp = int(((yt==0)&(yp==1)).sum())
    fn = int(((yt==1)&(yp==0)).sum())
    tp = int(((yt==1)&(yp==1)).sum())
    mat = np.array([[tn, fp],[fn, tp]])
    fig = plt.figure()
    plt.imshow(mat, cmap="Blues")
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    for (i,j),v in np.ndenumerate(mat): plt.text(j, i, str(v), ha="center", va="center")
    plt.title("Confusion"); plt.grid(False)
    return _savefig(fig, split, "confusion")

# ───── 추가 분류 플롯: Calibration / KS / Cost Curve / Lift-Gain ─────

def plot_calibration(y_true, y_score, split: str, bins: int = 15):
    """
    Reliability diagram + ECE/MCE 표시
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    ys = np.clip(ys, 1e-9, 1-1e-9)
    bins = max(3, int(bins))
    edges = np.linspace(0, 1, bins+1)
    idx = np.digitize(ys, edges) - 1
    accs, confs, counts = [], [], []
    for b in range(bins):
        m = (idx == b)
        if m.sum() == 0:
            accs.append(np.nan); confs.append((edges[b]+edges[b+1])/2); counts.append(0)
        else:
            accs.append( (yt[m]==1).mean() )
            confs.append( ys[m].mean() )
            counts.append( m.sum() )
    accs = np.array(accs); confs = np.array(confs); counts = np.array(counts)
    valid = counts > 0
    ece = np.nansum( (counts[valid]/counts[valid].sum()) * np.abs(accs[valid]-confs[valid]) )
    mce = np.nanmax( np.abs(accs[valid]-confs[valid]) ) if valid.any() else np.nan

    fig = plt.figure()
    plt.plot([0,1],[0,1],'--',alpha=0.6,label='Perfect')
    plt.scatter(confs[valid], accs[valid], s=20, alpha=0.9, label='Bins')
    plt.xlabel("Confidence"); plt.ylabel("Empirical accuracy")
    plt.title(f"Calibration (ECE={ece:.3f}, MCE={mce:.3f})")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")
    return _savefig(fig, split, "calibration")

def plot_ks(y_true, y_score, split: str):
    """
    KS 곡선 및 KS 통계(최대 거리) 표시
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    order = np.argsort(ys)  # ascending
    ys = ys[order]; yt = yt[order]
    pos = (yt==1).sum(); neg = (yt==0).sum()
    cum_pos = np.cumsum(yt==1)/max(1,pos)
    cum_neg = np.cumsum(yt==0)/max(1,neg)
    ks_vals = np.abs(cum_pos - cum_neg)
    ks = float(np.max(ks_vals))
    i = int(np.argmax(ks_vals))
    thr = ys[i]

    fig = plt.figure()
    x = np.linspace(0,1,len(ys))
    plt.plot(x, cum_pos, label='TPR cum')
    plt.plot(x, cum_neg, label='FPR cum')
    plt.vlines(x[i], cum_neg[i], cum_pos[i], colors='r', linestyles='--', label=f'KS={ks:.3f} @thr~{thr:.3f}')
    plt.xlabel("Score quantile"); plt.ylabel("Cumulative rate")
    plt.title("KS Curve"); plt.grid(True, alpha=0.3); plt.legend(loc="best")
    return _savefig(fig, split, "ks")

def plot_cost_curve(y_true, y_score, split: str, fp_cost: float = 1.0, fn_cost: float = 1.0, steps: int = 200):
    """
    임계값 sweep에 따른 비용( FP*cfp + FN*cfn ) 곡선
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    taus = np.linspace(0,1,max(10,int(steps)))
    costs = []
    for t in taus:
        yp = (ys >= t).astype(int)
        fp = int(((yt==0)&(yp==1)).sum())
        fn = int(((yt==1)&(yp==0)).sum())
        costs.append(fp*fp_cost + fn*fn_cost)
    costs = np.asarray(costs)
    i = int(np.argmin(costs))
    fig = plt.figure()
    plt.plot(taus, costs)
    plt.scatter([taus[i]], [costs[i]], color='r', s=24, label=f"min@τ={taus[i]:.3f}")
    plt.xlabel("threshold τ"); plt.ylabel("cost")
    plt.title(f"Cost Curve (cfp={fp_cost}, cfn={fn_cost})")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")
    return _savefig(fig, split, "cost_curve")

def plot_lift_gain(y_true, y_score, split: str, steps: int = 20):
    """
    Lift & Cumulative Gain chart
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(int).ravel()
    ys = _to_np(y_score).astype(float).ravel()
    order = np.argsort(-ys)
    yt = yt[order]
    n = yt.size
    steps = max(5,int(steps))
    ks = np.linspace(1/steps, 1, steps)
    gains = []
    lifts = []
    base = yt.mean() if n>0 else 0.0
    for k in ks:
        m = int(np.ceil(n*k))
        captured = (yt[:m]==1).sum() / max(1,(yt==1).sum())
        gains.append(captured)
        lifts.append( (captured / k) if k>0 else np.nan )

    fig = plt.figure(figsize=(7,3.5))
    import matplotlib.pyplot as plt
    ax1 = plt.gca()
    ax1.plot(ks*100, gains, label="Cumulative Gain")
    ax1.plot([0,100],[0,1],'--',alpha=0.5,label="Baseline")
    ax1.set_xlabel("Top k%"); ax1.set_ylabel("Captured positive ratio"); ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    plt.twinx()
    plt.plot(ks*100, lifts, color="tab:orange")
    plt.ylabel("Lift")
    plt.title("Lift & Cumulative Gain")
    return _savefig(fig, split, "lift_gain")

# ─────────────────────────── regression ───────────────────────────

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(float).ravel()
    yp = _to_np(y_pred).astype(float).ravel()
    resid = (yp - yt).astype(float)
    fig = plt.figure(); plt.hist(resid, bins=50, alpha=0.85); plt.title("Residual Histogram")
    plt.xlabel("pred - true"); plt.ylabel("count"); plt.grid(True, alpha=0.3)
    return _savefig(fig, split, "resid")

def plot_residual_qq(y_true: np.ndarray, y_pred: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import probplot as _pp  # SciPy 없으면 간단히 대체 가능하지만, 여기선 존재 가정X → 자체구현
    try:
        fig = plt.figure()
        yt = _to_np(y_true).astype(float).ravel()
        yp = _to_np(y_pred).astype(float).ravel()
        resid = (yp - yt).astype(float)
        _pp(resid, dist='norm', plot=plt)
        plt.title("Residual QQ-plot")
        return _savefig(fig, split, "resid_qq")
    except Exception:
        # SciPy 미존재 시 근사 구현
        yt = _to_np(y_true).astype(float).ravel()
        yp = _to_np(y_pred).astype(float).ravel()
        resid = (yp - yt).astype(float)
        q = np.linspace(0.001,0.999,199)
        emp = np.quantile(resid, q)
        # 표준정규 분위수 근사 (역오차함수)
        from math import sqrt, log
        def approx_norm_ppf(p):
            # Abramowitz-Stegun 근사
            a1=-39.696830; a2=220.946098; a3=-275.928510
            b1=-54.476098; b2=161.585836; b3=-155.698979
            c1=0.007784695; c2=0.322467129; c3=2.515517; c4=0.802853; c5=0.010328
            pl = 0.02425; ph = 1 - pl
            if p < pl:
                q = sqrt(-2*log(p))
                return (((((c1*q + c2)*q + c3)*q + c4)*q + c5) / ((((c1*q + c2)*q + 1)*q) + 1))
            if p > ph:
                q = sqrt(-2*log(1-p))
                return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5) / ((((c1*q + c2)*q + 1)*q) + 1))
            # 중앙영역 간단 근사
            return np.sqrt(2)*np.erfinv(2*p-1)
        theo = np.array([approx_norm_ppf(p) for p in q])
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig = plt.figure(); plt.scatter(theo, emp, s=8, alpha=0.7)
        plt.plot([theo.min(),theo.max()],[theo.min(),theo.max()],'--',alpha=0.6)
        plt.xlabel("Theoretical quantiles (N(0,1))"); plt.ylabel("Empirical residual quantiles")
        plt.title("Residual QQ-plot (approx)")
        return _savefig(fig, split, "resid_qq")

def plot_residual_time(y_true: np.ndarray, y_pred: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(float).ravel()
    yp = _to_np(y_pred).astype(float).ravel()
    resid = (yp - yt).astype(float)
    fig = plt.figure(); plt.plot(resid, lw=0.8)
    plt.xlabel("time (ordered index)"); plt.ylabel("residual"); plt.title("Residual over time")
    plt.grid(True, alpha=0.3)
    return _savefig(fig, split, "resid_time")

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(float).ravel()
    yp = _to_np(y_pred).astype(float).ravel()
    n = yt.shape[0]; idx = np.arange(n)
    if n>100_000:
        idx = np.random.RandomState(0).choice(n, size=100_000, replace=False)
    fig = plt.figure(); plt.scatter(yt[idx], yp[idx], s=2, alpha=0.5)
    plt.xlabel("true"); plt.ylabel("pred"); plt.title("Pred vs True"); plt.grid(True, alpha=0.3)
    return _savefig(fig, split, "pred_vs_true")

# ─────────────────────────── performance ───────────────────────────

def plot_latency_bar(latency_dict: dict, split: str):
    """
    latency_dict 예: {"b1":4.2, "b64":1.8, "b1024":0.7}
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not isinstance(latency_dict, dict) or len(latency_dict)==0:
        # 빈 입력이면 빈 그림이라도 남김
        fig = plt.figure(); plt.title("Latency (no data)"); return _savefig(fig, split, "latency_bar")
    keys = list(latency_dict.keys())
    vals = [latency_dict[k] for k in keys]
    fig = plt.figure()
    x = np.arange(len(keys)); plt.bar(x, vals)
    plt.xticks(x, keys); plt.ylabel("ms"); plt.title("Latency by batch")
    for i, v in enumerate(vals): plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    return _savefig(fig, split, "latency_bar")

# ─────────────────────────── samples ───────────────────────────

def plot_samples(X, y_true=None, y_pred=None, split: str = "val", max_samples: int = 3, ch: int = 0, title: str = "Samples"):
    """
    X: [N,C,T] 또는 [N,T] (torch.Tensor 또는 np.ndarray)
    y_true: (선택) [N] 또는 [N,T] — 다음 스텝 값 또는 시계열
    y_pred: (선택) [N] 또는 [N,T]
    저장: plots/after_<split>_samples.png
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Xn = _to_np(X)
    yt = _to_np(y_true) if y_true is not None else None
    yp = _to_np(y_pred) if y_pred is not None else None

    if Xn.ndim == 2:  # [N,T] → [N,1,T]
        Xn = Xn[:, None, :]

    N, C, T = Xn.shape
    sel = np.arange(min(max_samples, N))

    fig, axes = plt.subplots(len(sel), 1, figsize=(8, 2.5*len(sel)), squeeze=False)
    axes = axes.ravel()

    for ax, i in zip(axes, sel):
        series = Xn[i, ch if ch < C else 0, :]
        ax.plot(np.arange(T), series, lw=1.5)
        ax.set_xlim(0, T-1)
        ax.grid(True, alpha=0.3)

        # 다음 스텝 포인트(스칼라) 표기
        if yt is not None:
            if yt.ndim == 1:
                ax.scatter([T], [float(yt[i])], marker="o", s=24, label="y_true@T")
            elif yt.ndim == 2:
                ax.plot(np.arange(yt.shape[1]), yt[i], lw=1.0, alpha=0.8, label="y_true")
        if yp is not None:
            if yp.ndim == 1:
                ax.scatter([T], [float(yp[i])], marker="x", s=36, label="y_pred@T")
            elif yp.ndim == 2:
                ax.plot(np.arange(yp.shape[1]), yp[i], lw=1.0, alpha=0.8, linestyle="--", label="y_pred")

        ax.set_title(f"{title} — sample #{int(i)} (ch={ch})")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return _savefig(fig, split, "samples")
