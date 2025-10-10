# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: evaler.py                                                           ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  회귀/분류 공용 OOM-안전 평가(롤링윈도우 유지)                     ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   eval_model(model, X:[N,1,T], L:int, desc:str, task:str,                 ║
║              bin_rule:str, bin_thr:float, tau_for_cls:float,              ║
║              heartbeat_sec:int, indices:Optional[List[int]],              ║
║              split_name:Optional[str])                                    ║
║     -> (mse, mae, y_true, y_prob)                                         ║
║        * 회귀: (avg_mse, avg_mae, None, None)                             ║
║        * 분류: (None, None, y_true_all, y_prob_all)                       ║
╚════════════════════════════════════════════════════════════════════════════╝

evaler.py — OOM-안전 평가 루틴(회귀/분류 공용, 롤링 윈도우 유지)
"""
import time, numpy as np, torch, torch.nn.functional as F
from typing import Optional, Tuple, List
from windows import build_windows_dataset
from utils import USE_RICH, console, build_eval_table

@torch.no_grad()
def eval_model(model,
               X: torch.Tensor,
               L: int,
               desc: str = "eval",
               task: str = "regress",
               bin_rule: str = "nonzero",
               bin_thr: float = 0.0,
               tau_for_cls: float = 0.5,
               heartbeat_sec: int = 5,
               indices: Optional[List[int]] = None,
               split_name: Optional[str] = None):
    """
    반환:
      회귀: (avg_mse, avg_mae, None, None)
      분류: (None, None, y_true_all, y_prob_all)

    indices: 평가할 row 인덱스 서브셋(미지정 시 X 전체 사용)
    split_name: 로그 표시에 사용할 스플릿 이름(예: "val", "train")
    """
    model.eval()
    dev = next(model.parameters()).device
    N, C, T = X.shape

    rows = indices if (indices is not None and len(indices) > 0) else list(range(N))
    total_n = len(rows)

    last = time.time(); start = time.time()
    mse_list, mae_list = [], []
    prob_buf, true_buf = [], []
    tag = split_name if split_name else desc
    metric_name = "avg MSE" if task == "regress" else f"F1@τ={tau_for_cls:.3f}"
    metric_val = float("nan")

    def binarize(y_next: torch.Tensor) -> torch.Tensor:
        if bin_rule == "nonzero": return (y_next != 0).float()
        if bin_rule == "gt":      return (y_next > bin_thr).float()
        if bin_rule == "ge":      return (y_next >= bin_thr).float()
        return (y_next != 0).float()

    if USE_RICH:
        from rich.live import Live
        with Live(build_eval_table(f"{desc} ({tag})", 0, total_n, metric_name, float("nan"), None),
                  refresh_per_second=8, console=console) as live:
            for i, ridx in enumerate(rows):
                row = X[ridx].unsqueeze(0)
                Xw, Yw_reg, _, _ = build_windows_dataset(row, L)
                if Xw.numel() == 0:
                    continue
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
                    prob_buf.append(probs.detach().cpu())
                    true_buf.append(Yw_bin.detach().cpu())
                    yprob = torch.cat(prob_buf, dim=0)
                    ytrue = torch.cat(true_buf, dim=0)
                    # 실시간 표시는 F1만
                    tp = ((yprob>=tau_for_cls).float() * ytrue).sum().item()
                    fp = ((yprob>=tau_for_cls).float() * (1-ytrue)).sum().item()
                    fn = (((yprob<tau_for_cls).float()) * ytrue).sum().item()
                    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
                    metric_val = 2*prec*rec / (prec+rec+1e-8)

                now = time.time()
                if now - last >= heartbeat_sec:
                    done = i + 1
                    rate = (done / max(1, now - start))
                    remaining = (total_n - done) / rate if rate > 0 else None
                    live.update(build_eval_table(f"{desc} ({tag})", done, total_n, metric_name, metric_val, remaining))
                    last = now
            live.update(build_eval_table(f"{desc} ({tag})", total_n, total_n, metric_name, metric_val, 0.0))
    else:
        for ridx in rows:
            row = X[ridx].unsqueeze(0)
            Xw, Yw_reg, _, _ = build_windows_dataset(row, L)
            if Xw.numel() == 0:
                continue
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
            print(f"[{desc} ({tag})] avg MSE={float(np.mean(mse_list)) if mse_list else float('nan'):.6f}")
        else:
            if prob_buf:
                yprob = torch.cat(prob_buf, dim=0); ytrue = torch.cat(true_buf, dim=0)
                tp = ((yprob>=tau_for_cls).float() * ytrue).sum().item()
                fp = ((yprob>=tau_for_cls).float() * (1-ytrue)).sum().item()
                fn = (((yprob<tau_for_cls).float()) * ytrue).sum().item()
                prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
                f1 = 2*prec*rec / (prec+rec+1e-8)
                print(f"[{desc} ({tag})] F1@τ={tau_for_cls:.3f}={f1:.6f}")

    if task == "regress":
        return (float(np.mean(mse_list)) if mse_list else float("nan"),
                float(np.mean(mae_list)) if mae_list else float("nan"),
                None, None)
    else:
        yprob = torch.cat(prob_buf, dim=0) if prob_buf else torch.empty(0)
        ytrue = torch.cat(true_buf, dim=0) if true_buf else torch.empty(0)
        return None, None, ytrue, yprob
