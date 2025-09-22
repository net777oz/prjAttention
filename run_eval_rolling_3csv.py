# -*- coding: utf-8 -*-
"""
LLM-스타일 시계열 Transformer(llm_ts)로 3변량 CSV 롤링 평가
(…중략: 이전 설명 동일…)
"""

import argparse
import os
import math
import csv
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ttm_flow.model import load_model_and_cfg, extract_embeddings_ttm  # noqa


def read_csv_no_header(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    return data  # [N, T]


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def to_device_dtype(x: np.ndarray, device: str, dtype_str: str) -> torch.Tensor:
    dtype = torch.float32 if dtype_str == "float32" else torch.float64
    t = torch.from_numpy(x).to(device=device, dtype=dtype)
    return t


def eval_one_row_llm_ts(model, x_row: torch.Tensor, context_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    C, T = x_row.shape
    preds = torch.full((C, T), float("nan"), device=x_row.device, dtype=x_row.dtype)
    target = x_row.clone()
    context_len = max(1, min(context_len, T - 1))
    with torch.no_grad():
        for t in range(context_len, T):
            ctx = x_row[:, t - context_len : t].unsqueeze(0)
            pred_next, _ = model(ctx)
            preds[:, t] = pred_next.squeeze(0)
    return preds, target


def compute_metrics(preds: torch.Tensor, target: torch.Tensor, context_len: int) -> Tuple[float, float]:
    mask = ~torch.isnan(preds)
    mask[:, :context_len] = False
    valid_pred = preds[mask]
    valid_true = target[mask]
    mse = F.mse_loss(valid_pred, valid_true).item()
    mae = F.l1_loss(valid_pred, valid_true).item()
    return mse, mae


def plot_sample(preds: torch.Tensor, target: torch.Tensor, out_png: str, title: str):
    preds_np = preds.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    T = target_np.shape[1]

    # 색상 팔레트 (tru/pred 동일 색)
    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:red', 'tab:brown']

    plt.figure(figsize=(12, 4))
    for c in range(target_np.shape[0]):
        col = colors[c % len(colors)]
        # 실제 (실선)
        plt.plot(range(T), target_np[c], color=col, linewidth=1.2, label=f"ch{c+1} true")
        # 예측 (점선, 같은 색)
        mask = ~np.isnan(preds_np[c])
        if mask.any():
            plt.plot(np.arange(T)[mask], preds_np[c][mask],
                     color=col, linestyle="--", linewidth=1.2, label=f"ch{c+1} pred")

    plt.title(title)
    plt.xlabel("time (col index)")
    plt.ylabel("value")
    plt.legend(loc="upper right", fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", type=str, help="CSV for variable 1")
    ap.add_argument("--csv2", type=str, help="CSV for variable 2")
    ap.add_argument("--csv3", type=str, help="CSV for variable 3")
    ap.add_argument("--csv_a", type=str, help="(alias) CSV for variable 1")
    ap.add_argument("--csv_b", type=str, help="(alias) CSV for variable 2")
    ap.add_argument("--csv_c", type=str, help="(alias) CSV for variable 3")

    ap.add_argument("--backbone", type=str, default="llm_ts", help='Model backbone (default: "llm_ts")')
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--context-len", type=int, required=True, dest="context_len")

    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)

    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--plot-samples", type=int, default=3)

    args = ap.parse_args()

    csv1 = args.csv1 or args.csv_a
    csv2 = args.csv2 or args.csv_b
    csv3 = args.csv3 or args.csv_c
    if not (csv1 and csv2 and csv3):
        ap.error("You must provide 3 CSVs via --csv1/--csv2/--csv3 (or aliases --csv_a/--csv_b/--csv_c).")

    outdir = args.out or args.outdir
    if outdir is None:
        ap.error("Please specify output directory via --out (or --outdir).")
    ensure_outdir(outdir)

    A = read_csv_no_header(csv1)
    B = read_csv_no_header(csv2)
    C = read_csv_no_header(csv3)
    if not (A.shape == B.shape == C.shape):
        raise ValueError(f"CSV shapes must match. Got {A.shape}, {B.shape}, {C.shape}")

    N, T = A.shape
    if args.context_len >= T:
        print(f"[WARN] context_len({args.context_len}) >= T({T}) → auto-set to T-1")
        args.context_len = max(1, T - 1)
    if args.context_len < 1:
        print(f"[WARN] context_len({args.context_len}) < 1 → auto-set to 1")
        args.context_len = 1

    model, cfg = load_model_and_cfg(backbone=args.backbone, in_channels=3)
    model = model.to(device=args.device, dtype=(torch.float32 if args.dtype == "float32" else torch.float64))
    model.eval()

    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    A_t = to_device_dtype(A, args.device, args.dtype)
    B_t = to_device_dtype(B, args.device, args.dtype)
    C_t = to_device_dtype(C, args.device, args.dtype)

    all_mse, all_mae = [], []
    per_row_csv = os.path.join(outdir, "per_row_metrics.csv")
    if args.save_csv:
        with open(per_row_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["row_idx", "mse", "mae"])

    sample_count = 0
    for i in range(N):
        x_row = torch.stack([A_t[i], B_t[i], C_t[i]], dim=0)  # [C,T]
        preds, target = eval_one_row_llm_ts(model, x_row, args.context_len)
        mse, mae = compute_metrics(preds, target, args.context_len)
        all_mse.append(mse); all_mae.append(mae)

        if args.save_csv:
            with open(per_row_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([i, f"{mse:.6f}", f"{mae:.6f}"])

        if sample_count < max(0, args.plot_samples):
            png = os.path.join(outdir, f"row{i:04d}.png")
            plot_sample(preds, target, png, title=f"row {i} (context={args.context_len})")
            sample_count += 1

    mean_mse = float(np.mean(all_mse)) if all_mse else math.nan
    mean_mae = float(np.mean(all_mae)) if all_mae else math.nan

    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"backbone     : {args.backbone}\n")
        f.write(f"device/dtype : {args.device}/{args.dtype}\n")
        f.write(f"csvs         : {csv1}, {csv2}, {csv3}\n")
        f.write(f"N, T         : {N}, {T}\n")
        f.write(f"context_len  : {args.context_len}\n")
        f.write(f"mean MSE     : {mean_mse:.6f}\n")
        f.write(f"mean MAE     : {mean_mae:.6f}\n")

    print(f"[DONE] mean MSE={mean_mse:.6f}, mean MAE={mean_mae:.6f}")
    print(f"[OUT ] {outdir}")


if __name__ == "__main__":
    main()
