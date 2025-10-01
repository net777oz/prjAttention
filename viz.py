# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: viz.py                                                              ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  플롯: 시계열 샘플, F1-τ 커브, PR/ROC, Prob-Hist, Confusion, Loss  ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   plot_training_curve(loss_hist, outdir, title) -> None                   ║
║   plot_samples(model, X:[N,1,T], L:int, outdir, k:int, prefix:str)        ║
║   plot_cls_curves(y_true, y_prob, tau, outdir, prefix) -> None            ║
╠───────────────────────────────────────────────────────────────────────────╣
║ SIDE EFFECTS  plots/*.png 파일 생성                                        ║
║ DEPENDENCY   pipeline/evaler → viz; (windows, data.ensure_outdir)          ║
╚════════════════════════════════════════════════════════════════════════════╝

viz.py — 플롯 유틸(시계열 샘플, F1-τ, PR/ROC, Hist, Confusion, Loss curve)
"""
import os, numpy as np, torch, matplotlib.pyplot as plt
from windows import build_windows_dataset
from data import ensure_outdir

def plot_training_curve(loss_hist, outdir:str, title="Training Loss"):
    if not loss_hist: return
    plots_dir = os.path.join(outdir, "plots"); ensure_outdir(plots_dir)
    plt.figure(figsize=(8,5)); plt.plot(range(1, len(loss_hist)+1), loss_hist, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_loss.png")); plt.close()

def plot_samples(model, X: torch.Tensor, L:int, outdir:str, k:int=3, prefix:str="after"):
    ensure_outdir(outdir); plots_dir = os.path.join(outdir, "plots"); ensure_outdir(plots_dir)
    model.eval(); dev = next(model.parameters()).device
    N, C, T = X.shape
    for i in range(min(k, N)):
        row = X[i]
        preds = torch.full((C, T), float("nan"))
        Lc = max(1, min(L, T-1))
        Xw, _, _, _ = build_windows_dataset(row.unsqueeze(0), Lc)
        with torch.no_grad():
            pred, _ = model(Xw.to(dev, non_blocking=True)); pred = pred.cpu()
        preds[:, Lc:] = pred.T
        x_np = row.numpy(); p_np = preds.numpy()
        plt.figure(figsize=(12,4))
        plt.plot(range(T), x_np[0], linewidth=1.2, label=f"true")
        m = ~np.isnan(p_np[0])
        if m.any():
            plt.plot(np.arange(T)[m], p_np[0][m], ls="--", linewidth=1.2, label=f"pred")
        plt.legend(loc="upper right", fontsize=9)
        plt.title(f"row {i} (context={Lc})")
        plt.xlabel("time (col index)"); plt.ylabel("value")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_row{i:04d}.png")); plt.close()

def plot_cls_curves(y_true: torch.Tensor, y_prob: torch.Tensor, tau: float, outdir: str, prefix="after"):
    plots_dir = os.path.join(outdir, "plots"); ensure_outdir(plots_dir)
    yt = y_true.float().view(-1).cpu().numpy()
    yp = y_prob.float().view(-1).cpu().numpy()

    # F1 vs threshold
    taus = np.linspace(0.0, 1.0, 501); f1s = []
    for t in taus:
        yhat = (yp >= t).astype(np.float32)
        tp = np.sum((yhat==1) & (yt==1)); fp = np.sum((yhat==1) & (yt==0)); fn = np.sum((yhat==0) & (yt==1))
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1s.append(2*prec*rec / (prec + rec + 1e-8))
    plt.figure(figsize=(9,5)); plt.plot(taus, f1s); plt.axvline(tau, ls="--")
    plt.xlabel("Threshold τ"); plt.ylabel("F1"); plt.title("F1 vs Threshold")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_f1_vs_threshold.png")); plt.close()

    # PR curve
    order = np.argsort(-yp)
    tp=fp=0; fn=int(np.sum(yt==1)); precs=[]; recs=[]
    for idx in order:
        if yt[idx]==1: tp+=1; fn-=1
        else: fp+=1
        precs.append(tp / (tp+fp + 1e-8)); recs.append(tp / (tp+fn + 1e-8))
    plt.figure(figsize=(6,6)); plt.plot(recs, precs)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_pr.png")); plt.close()

    # Precision/Recall vs Threshold
    taus_dense = np.linspace(0, 1, 501); prec_l, rec_l = [], []
    for t in taus_dense:
        yhat = (yp >= t).astype(np.float32)
        tp = np.sum((yhat==1) & (yt==1))
        fp = np.sum((yhat==1) & (yt==0))
        fn = np.sum((yhat==0) & (yt==1))
        prec = tp / (tp + fp + 1e-8); rec  = tp / (tp + fn + 1e-8)
        prec_l.append(prec); rec_l.append(rec)
    plt.figure(figsize=(9,5))
    plt.plot(taus_dense, prec_l, label="Precision"); plt.plot(taus_dense, rec_l, label="Recall")
    plt.axvline(tau, ls="--"); plt.xlabel("Threshold τ"); plt.ylabel("Score"); plt.legend()
    plt.title("Precision / Recall vs Threshold")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_precision_recall_vs_threshold.png")); plt.close()

    # ROC
    P = int(np.sum(yt==1)); Nn = int(np.sum(yt==0))
    tprs=[]; fprs=[]
    for t in np.linspace(0,1,501):
        yhat = (yp >= t).astype(np.float32)
        tp = np.sum((yhat==1) & (yt==1)); fn = np.sum((yhat==0) & (yt==1))
        fp = np.sum((yhat==1) & (yt==0)); tn = np.sum((yhat==0) & (yt==0))
        tprs.append(tp / (P + 1e-8)); fprs.append(fp / (Nn + 1e-8))
    plt.figure(figsize=(6,6)); plt.plot(fprs, tprs); plt.plot([0,1],[0,1], ls="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_roc.png")); plt.close()

    # Probability histogram (공유 bins)
    plt.figure(figsize=(10,4))
    bins = np.linspace(0.0, 1.0, 51)
    plt.hist(yp[yt==0], bins=bins, alpha=0.6, label="y=0")
    plt.hist(yp[yt==1], bins=bins, alpha=0.6, label="y=1")
    plt.axvline(tau, ls="--"); plt.xlim(0, 1); plt.legend()
    plt.title("Probability Histogram (shared bins)")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_prob_hist.png")); plt.close()

    # Confusion @ τ
    yhat = (yp >= tau).astype(np.float32)
    tp = int(np.sum((yhat==1) & (yt==1))); tn = int(np.sum((yhat==0) & (yt==0)))
    fp = int(np.sum((yhat==1) & (yt==0))); fn = int(np.sum((yhat==0) & (yt==1)))
    cm = np.array([[tn, fp],[fn, tp]])
    plt.figure(figsize=(4.2,4.2)); plt.imshow(cm, cmap="Blues")
    for (i,j), val in np.ndenumerate(cm): plt.text(j, i, str(val), ha="center", va="center")
    plt.xticks([0,1], ["0","1"]); plt.yticks([0,1], ["0","1"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(f"Confusion @ τ={tau:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{prefix}_confusion.png")); plt.close()
