# -*- coding: utf-8 -*-
"""
LLM-스타일 시계열 Transformer (llm_ts) 통합 스크립트 — 회귀/이진분류 겸용
(요약)
- 분류 손실: α*BCEWithLogits + (1-α)*(1-SoftF1), pos_weight 전역/배치/없음
- τ: 검증 윈도우서 F1 최대화로 선택(기본 0.5 폴백)
- 리포트/플롯: F1-τ, PR, ROC, 확률 히스토그램, Confusion, (NEW) Precision/Recall vs Threshold, 학습곡선
- OOM 회피: 검증 배치 평가
- 결과: artifacts/<out>/...
"""

import argparse, os, time, math
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from ttm_flow.model import load_model_and_cfg

USE_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    USE_RICH = True
    console = Console()
except Exception:
    console = None

# ----------------------------- utils
def set_seed(seed: int = 777):
    import torch.backends.cudnn as cudnn
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False

def ensure_outdir(p: str): os.makedirs(p, exist_ok=True)

def read_csv_no_header(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8-sig") as f:
        x = np.loadtxt(f, delimiter=",", dtype=np.float32)
    return x[None, :] if x.ndim == 1 else x

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(dtype=torch.float32).contiguous()

def stack_multivar(csv_paths) -> torch.Tensor:
    mats = [to_tensor(read_csv_no_header(p)) for p in csv_paths]
    shapes = [tuple(m.shape) for m in mats]
    if len(set(shapes)) != 1:
        raise ValueError(f"All CSV shapes must match. got={shapes}")
    return torch.stack(mats, dim=1)  # [N,C,T]

def parse_csvs(args) -> torch.Tensor:
    if args.csv_list:
        paths = [p.strip() for p in args.csv_list.split(",") if p.strip()]
        if len(paths) < 1: raise SystemExit("--csv-list must have at least 1 path")
        return stack_multivar(paths)
    if args.csv1 and args.csv2 and args.csv3:
        return stack_multivar([args.csv1, args.csv2, args.csv3])
    if args.csv:
        M = to_tensor(read_csv_no_header(args.csv))  # [N,T]
        N, T = M.shape
        return torch.stack([M[0::3], M[1::3], M[2::3]], dim=1) if (N>=3 and N%3==0) else M.unsqueeze(1)
    raise SystemExit("CSV 입력 필요: --csv-list 또는 --csv1/--csv2/--csv3 또는 --csv")

def make_run_dir(args) -> str:
    base = "artifacts"
    slug = f"{args.mode}_{args.task}_ctx{args.context_len}_alpha{args.alpha:.2f}_pw{args.pos_weight}_{args.backbone}_seed{args.seed}"
    run_name = args.out if args.out else slug
    out_dir = os.path.join(base, run_name)
    ensure_outdir(out_dir); ensure_outdir(os.path.join(out_dir, "plots"))
    return out_dir

# ----------------------------- windows
def build_windows_dataset(X: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    N, C, T = X.shape
    L = max(1, min(L, T-1))
    W = T - L
    Xw_full = X.unfold(dimension=2, size=L, step=1)   # [N,C,T-L+1,L]
    Xw = Xw_full[:, :, :W, :]                         # [N,C,W,L]
    Yw = X[:, :, L:]                                  # [N,C,W]
    Xw = Xw.permute(0,2,1,3).contiguous().view(N*W, C, L)  # [N*W,C,L]
    Yw = Yw.permute(0,2,1).contiguous().view(N*W, C)       # [N*W,C]
    return Xw, Yw

# ----------------------------- UI
def fmt_time(secs: Optional[float]) -> str:
    if secs is None or np.isnan(secs): return "-"
    m, s = divmod(int(secs), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def build_train_table(ep, epochs, bi, num_batches, done, total,
                      loss_cur, loss_avg, lr, step_time, eta_sec, mem_mb, bs, amp, compiled):
    t = Table(expand=True, show_header=False, pad_edge=False, box=None)
    left = Table(show_header=False, pad_edge=False, box=None)
    left.add_row("Epoch", f"{ep}/{epochs}")
    left.add_row("Batch", f"{bi}/{num_batches}")
    left.add_row("Progress", f"{done}/{total} ({100.0*done/total:5.1f}%)")
    left.add_row("ETA", fmt_time(eta_sec))
    mid = Table(show_header=False, pad_edge=False, box=None)
    mid.add_row("Loss(cur)", f"{loss_cur:.6f}" if np.isfinite(loss_cur) else "nan")
    mid.add_row("Loss(avg)", f"{loss_avg:.6f}" if np.isfinite(loss_avg) else "nan")
    mid.add_row("LR", f"{lr:.2e}")
    mid.add_row("Step", f"{step_time:.2f}s")
    right = Table(show_header=False, pad_edge=False, box=None)
    right.add_row("BatchSize", str(bs))
    right.add_row("AMP", "ON" if amp else "OFF")
    right.add_row("Compile", compiled or "OFF")
    right.add_row("GPU Mem", f"{int(mem_mb)}MB")
    t.add_row(left, mid, right)
    return Panel(t, title="train", border_style="cyan")

def build_eval_table(phase: str, i:int, N:int, metric_name:str, metric_val:float, eta_sec:float):
    t = Table(expand=True, show_header=False, pad_edge=False, box=None)
    t.add_row("Phase", phase)
    t.add_row("Row", f"{i}/{N}")
    t.add_row(metric_name, f"{metric_val:.6f}" if np.isfinite(metric_val) else "nan")
    t.add_row("ETA", fmt_time(eta_sec))
    return Panel(t, title="eval", border_style="magenta")

# ----------------------------- classify helpers
class SoftF1Loss(torch.nn.Module):
    def __init__(self, eps=1e-8): super().__init__(); self.eps = eps
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        p = torch.sigmoid(logits); y = targets
        tp = (p * y).sum(); fp = (p * (1 - y)).sum(); fn = ((1 - p) * y).sum()
        soft_f1 = (2 * tp) / (2 * tp + fp + fn + self.eps)
        return 1.0 - soft_f1

@torch.no_grad()
def metrics_from_probs(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5, eps=1e-8):
    y_true = y_true.float().view(-1); y_prob = y_prob.float().view(-1)
    y_pred = (y_prob >= threshold).float()
    tp = (y_pred * y_true).sum().item()
    tn = ((1 - y_pred) * (1 - y_true)).sum().item()
    fp = (y_pred * (1 - y_true)).sum().item()
    fn = ((1 - y_pred) * y_true).sum().item()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(eps, (tp + fp)); rec = tp / max(eps, (tp + fn))
    f1 = 2 * prec * rec / max(eps, (prec + rec))
    mse = torch.mean((y_prob - y_true) ** 2).item()
    mae = torch.mean(torch.abs(y_prob - y_true)).item()
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
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1: best_f1, best_tau = f1, float(tau)
    return best_tau, best_f1

def compute_pos_weight_from_labels(y_bin: torch.Tensor) -> float:
    pos = float((y_bin > 0.5).sum()); neg = float((y_bin <= 0.5).sum())
    return (neg / pos) if pos > 0 else 1.0

# ----------------------------- eval (regress/classify)
@torch.no_grad()
def eval_model(model, X: torch.Tensor, L: int, desc="eval", heartbeat_sec=5,
               task: str = "regress", bin_rule: str = "nonzero", bin_thr: float = 0.0,
               tau_for_cls: float = 0.5):
    model.eval()
    dev = next(model.parameters()).device
    N, C, T = X.shape
    last = time.time(); start = time.time()
    mse_list, mae_list = [], []
    prob_buf, true_buf = [], []
    metric_name = "avg MSE" if task == "regress" else f"F1@τ={tau_for_cls:.3f}"
    metric_val = float("nan")

    def binarize(y_next: torch.Tensor) -> torch.Tensor:
        if bin_rule == "nonzero": return (y_next != 0).float()
        if bin_rule == "gt": return (y_next > bin_thr).float()
        if bin_rule == "ge": return (y_next >= bin_thr).float()
        return (y_next != 0).float()

    if USE_RICH:
        with Live(build_eval_table(desc, 0, N, metric_name, float("nan"), None),
                  refresh_per_second=8, console=console) as live:
            for i in range(N):
                row = X[i].unsqueeze(0)
                Xw, Yw_reg = build_windows_dataset(row, L)
                if Xw.numel() == 0: continue
                Xw = Xw.to(dev, non_blocking=True)

                if task == "regress":
                    Yw = Yw_reg.to(dev, non_blocking=True)
                    pred, _ = model(Xw)
                    mse_list.append(F.mse_loss(pred, Yw).item())
                    mae_list.append(F.l1_loss(pred, Yw).item())
                    metric_val = float(np.mean(mse_list)) if mse_list else float("nan")
                else:
                    Yw_bin = binarize(Yw_reg).to(dev, non_blocking=True)
                    logits, _ = model(Xw)
                    probs = torch.sigmoid(logits)
                    prob_buf.append(probs.detach().cpu()); true_buf.append(Yw_bin.detach().cpu())
                    yprob = torch.cat(prob_buf, dim=0); ytrue = torch.cat(true_buf, dim=0)
                    metric_val = metrics_from_probs(ytrue, yprob, threshold=tau_for_cls)["F1"]

                now = time.time()
                if now - last >= heartbeat_sec:
                    done = i + 1
                    rate = (done / max(1, now - start))
                    remaining = (N - done) / rate if rate > 0 else None
                    live.update(build_eval_table(desc, done, N, metric_name, metric_val, remaining))
                    last = now
            live.update(build_eval_table(desc, N, N, metric_name, metric_val, 0.0))
    else:
        for i in range(N):
            row = X[i].unsqueeze(0)
            Xw, Yw_reg = build_windows_dataset(row, L)
            if Xw.numel() == 0: continue
            Xw = Xw.to(dev, non_blocking=True)
            if task == "regress":
                Yw = Yw_reg.to(dev, non_blocking=True)
                pred, _ = model(Xw)
                mse_list.append(F.mse_loss(pred, Yw).item())
                mae_list.append(F.l1_loss(pred, Yw).item())
            else:
                Yw_bin = binarize(Yw_reg).to(dev, non_blocking=True)
                logits, _ = model(Xw)
                probs = torch.sigmoid(logits)
                prob_buf.append(probs.detach().cpu()); true_buf.append(Yw_bin.detach().cpu())
        if task == "regress":
            print(f"[{desc}] avg MSE={float(np.mean(mse_list)) if mse_list else float('nan'):.6f}")
        else:
            if prob_buf:
                yprob = torch.cat(prob_buf, dim=0); ytrue = torch.cat(true_buf, dim=0)
                rep = metrics_from_probs(ytrue, yprob, threshold=tau_for_cls)
                print(f"[{desc}] F1@τ={rep['threshold']:.3f}={rep['F1']:.6f} "
                      f"(Acc={rep['Accuracy']:.6f}, P={rep['Precision']:.6f}, R={rep['Recall']:.6f})")

    if task == "regress":
        return (float(np.mean(mse_list)) if mse_list else float("nan"),
                float(np.mean(mae_list)) if mae_list else float("nan"), None, None)
    else:
        yprob = torch.cat(prob_buf, dim=0) if prob_buf else torch.empty(0)
        ytrue = torch.cat(true_buf, dim=0) if true_buf else torch.empty(0)
        return None, None, ytrue, yprob

# ----------------------------- plots
def plot_samples(model, X: torch.Tensor, L: int, outdir: str, k: int = 3, prefix: str = "after"):
    ensure_outdir(outdir); plots_dir = os.path.join(outdir, "plots"); ensure_outdir(plots_dir)
    model.eval(); dev = next(model.parameters()).device
    N, C, T = X.shape
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    for i in range(min(k, N)):
        row = X[i]; preds = torch.full((C, T), float("nan"))
        Lc = max(1, min(L, T-1))
        Xw, _ = build_windows_dataset(row.unsqueeze(0), Lc)
        with torch.no_grad():
            pred, _ = model(Xw.to(dev, non_blocking=True)); pred = pred.cpu()
        preds[:, Lc:] = pred.T
        x_np = row.numpy(); p_np = preds.numpy()
        plt.figure(figsize=(12,4))
        for c in range(C):
            col = colors[c % len(colors)]
            plt.plot(range(T), x_np[c], color=col, linewidth=1.2, label=f"ch{c+1} true")
            m = ~np.isnan(p_np[c])
            if m.any():
                plt.plot(np.arange(T)[m], p_np[c][m], color=col, ls="--", linewidth=1.2, label=f"ch{c+1} pred")
        plt.legend(loc="upper right", fontsize=9, ncol=3)
        plt.title(f"row {i} (context={Lc})")
        plt.xlabel("time (col index)"); plt.ylabel("value")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_row{i:04d}.png")); plt.close()

def plot_training_curve(loss_hist: List[float], outdir: str, title="Training Loss"):
    if not loss_hist: return
    plots_dir = os.path.join(outdir, "plots"); ensure_outdir(plots_dir)
    plt.figure(figsize=(8,5)); plt.plot(range(1, len(loss_hist)+1), loss_hist, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_loss.png")); plt.close()

def plot_cls_curves(y_true: torch.Tensor, y_prob: torch.Tensor, tau: float, outdir: str, prefix="after"):
    plots_dir = os.path.join(outdir, "plots"); ensure_outdir(plots_dir)
    yt = y_true.float().view(-1).cpu().numpy()
    yp = y_prob.float().view(-1).cpu().numpy()

    # 1) F1 vs threshold
    taus = np.linspace(0.0, 1.0, 501); f1s = []
    for t in taus:
        yhat = (yp >= t).astype(np.float32)
        tp = np.sum((yhat==1) & (yt==1)); fp = np.sum((yhat==1) & (yt==0)); fn = np.sum((yhat==0) & (yt==1))
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1s.append(2*prec*rec / (prec + rec + 1e-8))
    plt.figure(figsize=(9,5))
    plt.plot(taus, f1s); plt.axvline(tau, ls="--")
    plt.xlabel("Threshold τ"); plt.ylabel("F1"); plt.title("F1 vs Threshold")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_f1_vs_threshold.png")); plt.close()

    # 2) PR curve (threshold sweep, high→low)
    order = np.argsort(-yp)
    tp=fp=0; fn=int(np.sum(yt==1)); precs=[]; recs=[]
    for idx in order:
        if yt[idx]==1: tp+=1; fn-=1
        else: fp+=1
        precs.append(tp / (tp+fp + 1e-8)); recs.append(tp / (tp+fn + 1e-8))
    plt.figure(figsize=(6,6))
    plt.plot(recs, precs)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_pr.png")); plt.close()

    # (NEW) 2-1) Precision/Recall vs Threshold
    taus_dense = np.linspace(0, 1, 501)
    prec_l, rec_l = [], []
    P = max(1, int(np.sum(yt==1))); Nn = max(1, int(np.sum(yt==0)))
    for t in taus_dense:
        yhat = (yp >= t).astype(np.float32)
        tp = np.sum((yhat==1) & (yt==1))
        fp = np.sum((yhat==1) & (yt==0))
        fn = np.sum((yhat==0) & (yt==1))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        prec_l.append(prec); rec_l.append(rec)
    plt.figure(figsize=(9,5))
    plt.plot(taus_dense, prec_l, label="Precision")
    plt.plot(taus_dense, rec_l, label="Recall")
    plt.axvline(tau, ls="--")
    plt.xlabel("Threshold τ"); plt.ylabel("Score"); plt.legend()
    plt.title("Precision / Recall vs Threshold")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_precision_recall_vs_threshold.png")); plt.close()

    # 3) ROC curve
    P = int(np.sum(yt==1)); Nn = int(np.sum(yt==0))
    tprs=[]; fprs=[]
    for t in np.linspace(0,1,501):
        yhat = (yp >= t).astype(np.float32)
        tp = np.sum((yhat==1) & (yt==1)); fn = np.sum((yhat==0) & (yt==1))
        fp = np.sum((yhat==1) & (yt==0)); tn = np.sum((yhat==0) & (yt==0))
        tprs.append(tp / (P + 1e-8)); fprs.append(fp / (Nn + 1e-8))
    plt.figure(figsize=(6,6))
    plt.plot(fprs, tprs); plt.plot([0,1],[0,1], ls="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_roc.png")); plt.close()

    # 4) Probability histogram
    plt.figure(figsize=(10,4))
    plt.hist(yp[yt==0], bins=50, alpha=0.7, label="y=0")
    plt.hist(yp[yt==1], bins=50, alpha=0.7, label="y=1")
    plt.axvline(tau, ls="--"); plt.legend(); plt.title("Probability Histogram")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_prob_hist.png")); plt.close()

    # 5) Confusion @ τ
    yhat = (yp >= tau).astype(np.float32)
    tp = int(np.sum((yhat==1) & (yt==1)))
    tn = int(np.sum((yhat==0) & (yt==0)))
    fp = int(np.sum((yhat==1) & (yt==0)))
    fn = int(np.sum((yhat==0) & (yt==1)))
    cm = np.array([[tn, fp],[fn, tp]])
    plt.figure(figsize=(4.2,4.2))
    plt.imshow(cm, cmap="Blues")
    for (i,j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.xticks([0,1], ["0","1"]); plt.yticks([0,1], ["0","1"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(f"Confusion @ τ={tau:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_confusion.png")); plt.close()

# ----------------------------- train loop
def train_all_epochs(model, dl: DataLoader, opt, scaler: GradScaler,
                     epochs:int, amp_enabled:bool=True,
                     log_every:int=50, heartbeat_sec:int=5, compiled:str|None=None,
                     task:str="regress", alpha:float=0.5,
                     pos_weight_mode:str="global", global_pos_weight:Optional[float]=None):
    dev = next(model.parameters()).device
    total_steps = epochs * len(dl); step_counter = 0
    last = time.time(); avg_step_time = None

    softf1 = SoftF1Loss() if task == "classify" else None
    bce_loss_fn = None
    if task == "classify":
        if pos_weight_mode == "none":
            bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        else:
            pw = torch.tensor([global_pos_weight if global_pos_weight is not None else 1.0],
                              dtype=torch.float32, device=dev)
            bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="mean")

    epoch_loss_hist: List[float] = []

    def update_table(ep, bi, loss_cur, loss_avg, bs):
        nonlocal avg_step_time, last, step_counter
        now = time.time(); dt = now - last; last = now
        avg_step_time = dt if avg_step_time is None else 0.9*avg_step_time + 0.1*dt
        done = step_counter; remaining = (total_steps - done) * (avg_step_time if avg_step_time else 0.0)
        lr = opt.param_groups[0].get("lr", 0.0)
        mem = (torch.cuda.memory_allocated()/ (1024**2)) if torch.cuda.is_available() else 0.0
        if USE_RICH:
            table = build_train_table(ep, epochs, bi, len(dl),
                                      done, total_steps, loss_cur, loss_avg, lr,
                                      avg_step_time if avg_step_time else 0.0,
                                      remaining, mem, bs, amp_enabled, compiled)
            live.update(table)
        else:
            print(f"\r[train] ep {ep}/{epochs} batch {bi}/{len(dl)} "
                  f"done {done}/{total_steps} loss={loss_cur:.6f} avg={loss_avg:.6f} "
                  f"lr={lr:.2e} step={avg_step_time:.2f}s ETA={fmt_time(remaining)} mem={int(mem)}MB",
                  end="", flush=True)

    if USE_RICH:
        live = Live(Panel("initializing...", title="train", border_style="cyan"),
                    refresh_per_second=10, console=console); live.start()
    else:
        live = None

    try:
        for ep in range(1, epochs+1):
            model.train(); total, steps = 0.0, 0
            for bi, (xb, yb) in enumerate(dl):
                xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                if amp_enabled:
                    with autocast():
                        logits, _ = model(xb)
                        if task == "regress":
                            loss = F.mse_loss(logits, yb)
                        else:
                            if pos_weight_mode == "batch":
                                pw = compute_pos_weight_from_labels(yb)
                                bce_loss_fn.pos_weight = torch.tensor([pw], dtype=torch.float32, device=dev)
                            bce = bce_loss_fn(logits, yb); s1f = softf1(logits, yb)
                            loss = alpha * bce + (1 - alpha) * s1f
                    if not torch.isfinite(loss):
                        step_counter += 1; update_table(ep, bi, float("nan"), float("nan"), xb.size(0)); continue
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    logits, _ = model(xb)
                    if task == "regress":
                        loss = F.mse_loss(logits, yb)
                    else:
                        if pos_weight_mode == "batch":
                            pw = compute_pos_weight_from_labels(yb)
                            bce_loss_fn.pos_weight = torch.tensor([pw], dtype=torch.float32, device=dev)
                        bce = bce_loss_fn(logits, yb); s1f = softf1(logits, yb)
                        loss = alpha * bce + (1 - alpha) * s1f
                    if not torch.isfinite(loss):
                        step_counter += 1; update_table(ep, bi, float("nan"), float("nan"), xb.size(0)); continue
                    loss.backward(); opt.step()

                total += loss.item(); steps += 1; step_counter += 1
                avg = total / steps
                if ((bi % max(1, log_every)) == 0):
                    update_table(ep, bi, loss.item(), avg, xb.size(0))
            ep_avg = total/max(1,steps); epoch_loss_hist.append(ep_avg)
            print(f"\n[EPOCH {ep}/{epochs}] avg_loss={ep_avg:.6f}", flush=True)
    finally:
        if USE_RICH: live.stop()
        else: print()
    return epoch_loss_hist

# ----------------------------- main
def main():
    ap = argparse.ArgumentParser()
    # 입력
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--csv1", type=str, default=None)
    ap.add_argument("--csv2", type=str, default=None)
    ap.add_argument("--csv3", type=str, default=None)
    ap.add_argument("--csv-list", type=str, default=None, help='Comma list: "a.csv,b.csv,..."')

    # 모드/작업/모델/학습/출력
    ap.add_argument("--mode", type=str, required=True, choices=["train","finetune","infer"])
    ap.add_argument("--task", type=str, default="regress", choices=["regress","classify"])
    ap.add_argument("--backbone", type=str, default="llm_ts")
    ap.add_argument("--context-len", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", type=str, default=None, help="artifacts/<out>/ 에 저장. 없으면 옵션 슬러그 사용")
    ap.add_argument("--plot-samples", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=50)

    # 분류 전용
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=str, default="global", choices=["global","batch","none"])
    ap.add_argument("--thresh-default", type=float, default=0.5)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--bin-rule", type=str, default="nonzero", choices=["nonzero","gt","ge"])
    ap.add_argument("--bin-thr", type=float, default=0.0)

    # 리소스
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", type=str, default="", help='""|"reduce-overhead"|"max-autotune"')
    ap.add_argument("--eval-batch-size", type=int, default=4096)

    # 체크포인트
    ap.add_argument("--ckpt", type=str, default=None)

    args = ap.parse_args()
    set_seed(args.seed)
    out_dir = make_run_dir(args)

    # 데이터
    X = parse_csvs(args); N, C, T = X.shape

    # context_len 보정
    if args.context_len >= T:
        print(f"[WARN] context_len({args.context_len}) >= T({T}) → auto-set to T-1", flush=True)
        args.context_len = T - 1
    if args.context_len < 1:
        print(f"[WARN] context_len({args.context_len}) < 1 → auto-set to 1", flush=True)
        args.context_len = 1

    # 모델
    model, _ = load_model_and_cfg(backbone=args.backbone, in_channels=C)
    model = model.to(device=args.device, dtype=torch.float32)

    compiled_mode = None
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile)
            compiled_mode = args.compile
            print(f"[INFO] torch.compile enabled: mode={args.compile}", flush=True)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}", flush=True)

    print(f"[INFO] start | mode={args.mode}, task={args.task}, N={N},C={C},T={T},context={args.context_len},params={sum(p.numel() for p in model.parameters())}", flush=True)

    # ---- infer
    if args.mode == "infer":
        if not args.ckpt: raise SystemExit("--mode infer 는 --ckpt 가 필요합니다.")
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

        if args.task == "regress":
            bmse, bmae, _, _ = eval_model(model, X, args.context_len, desc="infer", heartbeat_sec=5, task="regress")
            print(f"[INFER] MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)
            plot_samples(model, X, args.context_len, out_dir, k=args.plot_samples, prefix="infer")
            with open(os.path.join(out_dir, "infer_report.txt"), "w", encoding="utf-8") as f:
                f.write(f"infer MSE {bmse:.6f} MAE {bmae:.6f}\n")
        else:
            _, _, ytrue, yprob = eval_model(model, X, args.context_len, desc="infer",
                                            heartbeat_sec=5, task="classify",
                                            bin_rule=args.bin_rule, bin_thr=args.bin_thr,
                                            tau_for_cls=args.thresh_default)
            if yprob is not None and yprob.numel():
                rep = metrics_from_probs(ytrue, yprob, threshold=args.thresh_default)
                print(f"[INFER-CLS] τ={rep['threshold']:.3f} | "
                      f"F1={rep['F1']:.6f} Acc={rep['Accuracy']:.6f} "
                      f"P={rep['Precision']:.6f} R={rep['Recall']:.6f} "
                      f"(MSE={rep['MSE']:.6f}, MAE={rep['MAE']:.6f})", flush=True)
                with open(os.path.join(out_dir, "infer_report.txt"), "w", encoding="utf-8") as f:
                    f.write(f"τ {rep['threshold']:.3f} F1 {rep['F1']:.6f} Acc {rep['Accuracy']:.6f} "
                            f"P {rep['Precision']:.6f} R {rep['Recall']:.6f} "
                            f"MSE {rep['MSE']:.6f} MAE {rep['MAE']:.6f}\n")
                plot_cls_curves(ytrue, yprob, args.thresh_default, out_dir, prefix="infer")
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        print(f"[DONE] infer saved to {out_dir}", flush=True)
        return

    # ---- train/finetune
    if args.mode == "finetune":
        if not args.ckpt: raise SystemExit("--mode finetune 는 --ckpt 가 필요합니다.")
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    print("[INFO] starting BEFORE eval...", flush=True)
    if args.task == "regress":
        bmse, bmae, _, _ = eval_model(model, X, args.context_len, desc="before", heartbeat_sec=5, task="regress")
        print(f"[BEFORE] MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)
    else:
        _, _, ytrue_b, yprob_b = eval_model(model, X, args.context_len, desc="before",
                                            heartbeat_sec=5, task="classify",
                                            bin_rule=args.bin_rule, bin_thr=args.bin_thr,
                                            tau_for_cls=args.thresh_default)
        if yprob_b is not None and yprob_b.numel():
            rep_b = metrics_from_probs(ytrue_b, yprob_b, threshold=args.thresh_default)
            print(f"[BEFORE-CLS] τ={rep_b['threshold']:.3f} | F1={rep_b['F1']:.6f} "
                  f"Acc={rep_b['Accuracy']:.6f} P={rep_b['Precision']:.6f} R={rep_b['Recall']:.6f} "
                  f"(MSE={rep_b['MSE']:.6f}, MAE={rep_b['MAE']:.6f})", flush=True)
            plot_cls_curves(ytrue_b, yprob_b, args.thresh_default, out_dir, prefix="before")

    Xw_all, Yw_next = build_windows_dataset(X, args.context_len)
    if args.task == "regress":
        Yw_all = Yw_next
    else:
        if args.bin_rule == "nonzero": Yw_all = (Yw_next != 0).float()
        elif args.bin_rule == "gt":   Yw_all = (Yw_next > args.bin_thr).float()
        else:                          Yw_all = (Yw_next >= args.bin_thr).float()

    total = Xw_all.shape[0]
    idx = torch.arange(total); g = torch.Generator().manual_seed(args.seed)
    perm = idx[torch.randperm(total, generator=g)]
    val_size = int(total * args.val_ratio) if args.task == "classify" else 0
    val_idx = perm[:val_size] if val_size > 0 else torch.empty(0, dtype=torch.long)
    trn_idx = perm[val_size:] if val_size > 0 else perm

    Xw_trn, Yw_trn = Xw_all[trn_idx], Yw_all[trn_idx]
    Xw_val, Yw_val = (Xw_all[val_idx], Yw_all[val_idx]) if val_size > 0 else (None, None)

    ds = TensorDataset(Xw_trn, Yw_trn)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=False, persistent_workers=True if args.num_workers>0 else False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    global_pw = None
    if args.task == "classify" and args.pos_weight != "none":
        global_pw = compute_pos_weight_from_labels(Yw_trn)
        print(f"[INFO] pos_weight(global) = {global_pw:.6f} (mode={args.pos_weight})", flush=True)

    loss_hist = train_all_epochs(model, dl, opt, scaler,
                                 epochs=args.epochs, amp_enabled=args.amp,
                                 log_every=args.log_every, heartbeat_sec=5, compiled=compiled_mode,
                                 task=args.task, alpha=args.alpha,
                                 pos_weight_mode=args.pos_weight, global_pos_weight=global_pw)
    plot_training_curve(loss_hist, out_dir, title="Training Loss")

    print("[INFO] starting AFTER eval...", flush=True)
    if args.task == "regress":
        amse, amae, _, _ = eval_model(model, X, args.context_len, desc="after", heartbeat_sec=5, task="regress")
        print(f"[AFTER ] MSE={amse:.6f} MAE={amae:.6f}", flush=True)
        with open(os.path.join(out_dir, "train_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"before MSE {bmse:.6f} MAE {bmae:.6f}\n")
            f.write(f"after  MSE {amse:.6f} MAE {amae:.6f}\n")
        plot_samples(model, X, args.context_len, out_dir, k=args.plot_samples, prefix="after")
    else:
        if (Xw_val is not None) and (Yw_val is not None) and (Xw_val.numel() > 0):
            dev = args.device; eval_bs = max(1, args.eval_batch_size)
            probs_chunks = []
            with torch.no_grad():
                for i in range(0, Xw_val.size(0), eval_bs):
                    xb = Xw_val[i:i+eval_bs].to(dev, non_blocking=True)
                    if args.amp:
                        with autocast(): logits_b, _ = model(xb)
                    else:
                        logits_b, _ = model(xb)
                    probs_chunks.append(torch.sigmoid(logits_b).cpu())
            probs_val = torch.cat(probs_chunks, dim=0)
            tau, f1_at_tau = find_best_threshold_for_f1(Yw_val.cpu(), probs_val, step=0.001)
        else:
            tau, f1_at_tau = args.thresh_default, float("nan")
            print("[WARN] 검증 분할이 없어 τ 기본값을 사용합니다.", flush=True)

        _, _, ytrue_all, yprob_all = eval_model(model, X, args.context_len, desc="after",
                                                heartbeat_sec=5, task="classify",
                                                bin_rule=args.bin_rule, bin_thr=args.bin_thr,
                                                tau_for_cls=tau)
        rep = metrics_from_probs(ytrue_all, yprob_all, threshold=tau)
        print(f"[AFTER-CLS] Selected τ={rep['threshold']:.3f} | "
              f"F1={rep['F1']:.6f} Acc={rep['Accuracy']:.6f} "
              f"P={rep['Precision']:.6f} R={rep['Recall']:.6f} "
              f"(MSE={rep['MSE']:.6f}, MAE={rep['MAE']:.6f})", flush=True)
        with open(os.path.join(out_dir, "train_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"tau {rep['threshold']:.3f}\n")
            f.write(f"F1 {rep['F1']:.6f} Acc {rep['Accuracy']:.6f} P {rep['Precision']:.6f} R {rep['Recall']:.6f}\n")
            f.write(f"(ref) MSE {rep['MSE']:.6f} MAE {rep['MAE']:.6f}\n")

        # 그래프 저장 (PR/ROC/히스토/Confusion + NEW: PR vs τ)
        plot_cls_curves(ytrue_all, yprob_all, rep["threshold"], out_dir, prefix="after")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"[DONE] model/report/plots saved to {out_dir}", flush=True)

if __name__ == "__main__":
    main()
