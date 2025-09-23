# -*- coding: utf-8 -*-
"""
LLM-스타일 시계열 Transformer(llm_ts)로 3변량 CSV 롤링 평가
(row-batch 병렬화 + flexible plot-samples 버전)
"""

import argparse, os, math, csv, time
import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from ttm_flow.model import load_model_and_cfg

# rich optional
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


def read_csv_no_header(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[None, :] if data.ndim == 1 else data


def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)


def to_tensor(x: np.ndarray, device: str, dtype: str) -> torch.Tensor:
    return torch.from_numpy(x).to(
        device=device,
        dtype=(torch.float32 if dtype == "float32" else torch.float64),
    )


def eval_batch_llm_ts(model, X: torch.Tensor, context_len: int, row_bs: int = 128):
    """
    X: [N,C,T] (device)
    return preds, target (both [N,C,T])
    """
    N, C, T = X.shape
    preds = torch.full_like(X, float("nan"))
    target = X.clone()
    L = max(1, min(context_len, T - 1))

    preds[:, :, :L] = X[:, :, :L]

    with torch.no_grad():
        for t in range(L, T):
            for start in range(0, N, row_bs):
                end = min(start + row_bs, N)
                ctx = X[start:end, :, t - L : t]  # [B,C,L]
                pred_next, _ = model(ctx)  # [B,C]
                preds[start:end, :, t] = pred_next
    return preds, target


def compute_metrics(preds: torch.Tensor, target: torch.Tensor, context_len: int):
    mask = ~torch.isnan(preds)
    mask[:, :, :context_len] = False
    vpred = preds[mask]
    vtrue = target[mask]
    return F.mse_loss(vpred, vtrue).item(), F.l1_loss(vpred, vtrue).item()


def plot_sample(preds: torch.Tensor, target: torch.Tensor, out_png: str, title: str, row_idx: int):
    p = preds.cpu().numpy()
    t = target.cpu().numpy()
    T = t.shape[1]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(12, 4))
    for c in range(t.shape[0]):
        col = colors[c % len(colors)]
        plt.plot(range(T), t[c], color=col, linewidth=1.2, label=f"ch{c+1} true")
        m = ~np.isnan(p[c])
        if m.any():
            plt.plot(
                np.arange(T)[m],
                p[c, m],
                "--",
                color=col,
                linewidth=1.2,
                label=f"ch{c+1} pred",
            )
    plt.title(f"{title} (row={row_idx})")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def fmt_time(secs):
    if secs is None or math.isnan(secs):
        return "-"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_eval_table(done, N, mse, eta):
    from rich.table import Table
    from rich.panel import Panel

    t = Table(show_header=False, box=None)
    t.add_row("Row batches", f"{done}/{N}")
    t.add_row("avg MSE", f"{mse:.6f}")
    t.add_row("ETA", fmt_time(eta))
    return Panel(t, title="eval", border_style="magenta")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1")
    ap.add_argument("--csv2")
    ap.add_argument("--csv3")
    ap.add_argument("--backbone", default="llm_ts")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--context-len", type=int, required=True, dest="context_len")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-batch", type=int, default=128, help="row batch size for eval")

    ap.add_argument("--plot-samples", type=int, default=3)
    ap.add_argument(
        "--plot-mode",
        type=str,
        default="head",
        choices=["head", "random", "indices"],
    )
    ap.add_argument(
        "--plot-indices",
        type=str,
        default="",
        help="comma-separated indices when plot-mode=indices",
    )
    ap.add_argument("--plot-seed", type=int, default=777)

    args = ap.parse_args()

    ensure_outdir(args.out)
    A, B, C = [read_csv_no_header(p) for p in (args.csv1, args.csv2, args.csv3)]
    if not (A.shape == B.shape == C.shape):
        raise ValueError("CSV shapes mismatch")
    N, T = A.shape
    L = args.context_len
    if L >= T:
        L = T - 1
    X = np.stack([A, B, C], axis=1)  # [N,C,T]
    X_t = to_tensor(X, args.device, args.dtype)

    model, _ = load_model_and_cfg(backbone=args.backbone, in_channels=3)
    model = model.to(
        device=args.device,
        dtype=(torch.float32 if args.dtype == "float32" else torch.float64),
    )
    model.eval()
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)

    start = time.time()
    if USE_RICH:
        live = Live(
            build_eval_table(0, N, float("nan"), None),
            console=console,
            refresh_per_second=4,
        )
        live.start()
    else:
        live = None

    preds, target = eval_batch_llm_ts(model, X_t, L, row_bs=args.row_batch)
    mse, mae = compute_metrics(preds, target, L)

    # ---- plot sample rows ----
    K = max(0, args.plot_samples)
    if K > 0:
        if args.plot_mode == "random":
            rng = np.random.default_rng(args.plot_seed)
            idxs = rng.choice(N, size=min(K, N), replace=False)
        elif args.plot_mode == "indices":
            raw = args.plot_indices
            idxs = np.array(
                [int(s) for s in raw.split(",") if s.strip() != ""], dtype=int
            )
            idxs = idxs[(idxs >= 0) & (idxs < N)][:K]
        else:  # head
            idxs = np.arange(min(K, N))

        for i in idxs:
            png = os.path.join(args.out, f"row{i:04d}.png")
            plot_sample(preds[i], target[i], png, "eval", int(i))

    if live:
        live.update(build_eval_table(N, N, mse, 0.0))
        live.stop()
    else:
        print(f"[eval] mean MSE={mse:.6f} mean MAE={mae:.6f}")

    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(f"N={N},T={T},context={L}\n")
        f.write(f"mean MSE={mse:.6f}\n")
        f.write(f"mean MAE={mae:.6f}\n")
    print(f"[DONE] report saved {args.out}")


if __name__ == "__main__":
    main()
