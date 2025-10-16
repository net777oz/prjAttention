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
  - plot_residuals(y_true, y_pred, split)
  - plot_pred_vs_true(y_true, y_pred, split)
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
    return Path("./artifacts").resolve() / "default_run"  # 러너에서 AP_OUT_DIR 세팅 권장

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
        if arr.dtype.kind in ("i", "u", "f"):  # int/uint/float
            return True
        # 문자열 배열일 수 있음
        return False
    except Exception:
        return False

def _compat_resolve_split_tau(args, kwargs, default_split="val", default_tau=0.5):
    """
    지원 패턴:
      - (split='val', tau=0.5)  # 키워드
      - ("val",)                # 첫 위치인자 = split
      - (0.5, ...)              # 첫 위치인자 = tau (레거시)
      - (tau, out_dir, prefix='before_val')  # 레거시: prefix에서 split 추출
    반환: (split, tau)
    """
    split = kwargs.pop("split", None)
    tau = kwargs.pop("tau", None)
    prefix = kwargs.get("prefix")  # 그대로 두되 저장엔 사용 안 함

    # 위치 인자 해석
    if len(args) == 1:
        a0 = args[0]
        if isinstance(a0, str):
            split = a0
        elif isinstance(a0, (int, float)):
            tau = float(a0)
    elif len(args) >= 2:
        a0, a1 = args[0], args[1]
        if isinstance(a0, (int, float)):
            tau = float(a0)        # (tau, out_dir, prefix=...)
        elif isinstance(a0, str):
            split = a0             # ("val", tau, ...)
            if isinstance(a1, (int, float)):
                tau = float(a1)

    # prefix로부터 split 추출(예: "before_val" → "val")
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
    호환되는 호출 예:
      - plot_training_curve(tr, va, split='val')
      - plot_training_curve(tr, va, out_dir, prefix='before_val')
      - plot_training_curve(tr, out_dir, title='Training Loss')  # 레거시: 두번째가 경로면 무시
    저장: plots/after_<split>_loss.png
    """
    # 1) 두번째 위치 인자가 숫자 시퀀스가 아니면(경로/문자열) → val_losses=None로 간주
    if not _looks_numeric_series(val_losses):
        # val_losses가 경로/문자열/None인 경우, 레거시 호출로 간주하고 무시
        val_losses = None
        # args는 그대로 split/tau 해석용으로 넘김

    # 2) args에 '숫자 시퀀스'가 첫 슬롯으로 온 경우(레거시: tr, va, out_dir, ...) → 그걸 val_losses로 해석
    args_list = list(args)
    if val_losses is None and len(args_list) > 0 and _looks_numeric_series(args_list[0]):
        val_losses = args_list[0]
        args_list = args_list[1:]  # 나머지는 split/tau/out_dir/prefix 등

    # 3) split/tau 해석 (out_dir는 무시, prefix에서 split 추출)
    split, _ = _compat_resolve_split_tau(tuple(args_list), kwargs, default_split="val", default_tau=0.5)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tr = _to_np(train_losses).astype(float).ravel() if train_losses is not None else None
    va = _to_np(val_losses).astype(float).ravel() if val_losses is not None else None

    fig = plt.figure()
    if tr is not None and tr.size > 0:
        plt.plot(np.arange(1, tr.size+1), tr, label="train")
    if va is not None and va.size > 0:
        plt.plot(np.arange(1, va.size+1), va, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(kwargs.get("title", "Training Curve")); plt.grid(True, alpha=0.3)
    if (tr is not None and tr.size>0) and (va is not None and va.size>0):
        plt.legend(loc="best")
    return _savefig(fig, split, "loss")

# ─────────────────────────── classification helpers ───────────────────────────

def plot_cls_curves(y_true, y_score, *args, **kwargs):
    """
    분류 전용 종합 플로팅(레거시/신규 모두 지원):
      - 신규: plot_cls_curves(y_true, y_score, split='val', tau=0.5)
      - 레거시: plot_cls_curves(y_true, y_score, tau, out_dir, prefix='before_val')
    저장물:
      plots/after_<split>_roc.png
      plots/after_<split>_pr.png
      plots/after_<split>_hist_all.png (+ _pos/_neg)
      plots/after_<split>_confusion.png   (τ 기준)
    반환: {"roc":Path, "pr":Path, "hist_all":Path, "confusion":Path, ...}
    """
    split, tau = _compat_resolve_split_tau(args, kwargs, default_split="val", default_tau=0.5)
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

# ─────────────────────────── regression ───────────────────────────

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, split: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yt = _to_np(y_true).astype(float).ravel()
    yp = _to_np(y_pred).astype(float).ravel()
    resid = (yp - yt).astype(float)
    fig = plt.figure(); plt.hist(resid, bins=50, alpha=0.85); plt.title("Residual Histogram")
    plt.xlabel("pred - true"); plt.ylabel("count"); plt.grid(True, alpha=0.3)
    return _savefig(fig, split, "resid")

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

# ─────────────────────────── samples ───────────────────────────

def plot_samples(X, y_true=None, y_pred=None, split: str = "val", max_samples: int = 3, ch: int = 0, title: str = "Samples"):
    """
    X: [N,C,T] 또는 [N,T] (torch.Tensor 또는 np.ndarray)
    y_true: (선택) [N] 또는 [N,T] — 다음 스텝 값 또는 시계열
    y_pred: (선택) [N] 또는 [N,T]
    저장: plots/after_<split>_samples.png (단일 그림, 최대 max_samples 서브플롯)
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

        # 다음 스텝 포인트(스칼라) 표기 — y_true/ y_pred가 1D이면 T 위치에 점으로 표시
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
