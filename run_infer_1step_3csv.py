# -*- coding: utf-8 -*-
"""
LLM-스타일 시계열 Transformer (llm_ts) 추론 전용 스크립트
- CSV(헤더/인덱스 없음): 행=아이템(N), 열=시간(T)
- 다변량 입력: --csv1/2/3 또는 --csv-list
- 리치 박스 UI로 진행률 + 지표 출력
- PNG 샘플 저장 + 리포트 텍스트 저장
"""

import argparse, os, time
import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from ttm_flow.model import load_model_and_cfg

# ===== Console UI (rich optional) =====
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


# -----------------------------
# 유틸
# -----------------------------
def ensure_outdir(p: str): os.makedirs(p, exist_ok=True)

def read_csv_no_header(path: str) -> np.ndarray:
    x = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return x[None, :] if x.ndim == 1 else x

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(dtype=torch.float32).contiguous()

def stack_multivar(csv_paths) -> torch.Tensor:
    mats = [to_tensor(read_csv_no_header(p)) for p in csv_paths]
    shapes = [tuple(m.shape) for m in mats]
    if len(set(shapes)) != 1:
        raise ValueError(f"CSV shapes mismatch: {shapes}")
    return torch.stack(mats, dim=1)  # [N,C,T]

def parse_csvs(args) -> torch.Tensor:
    if args.csv_list:
        paths = [p.strip() for p in args.csv_list.split(",") if p.strip()]
        return stack_multivar(paths)
    if args.csv1 and args.csv2 and args.csv3:
        return stack_multivar([args.csv1, args.csv2, args.csv3])
    if args.csv:
        M = to_tensor(read_csv_no_header(args.csv))
        N, T = M.shape
        return torch.stack([M[0::3], M[1::3], M[2::3]], dim=1) if (N >= 3 and N % 3 == 0) else M.unsqueeze(1)
    raise SystemExit("CSV 입력 필요: --csv-list 또는 --csv1/--csv2/--csv3 또는 --csv")


# -----------------------------
# 롤링 윈도우 벡터화
# -----------------------------
def build_windows_dataset(X: torch.Tensor, L: int):
    N, C, T = X.shape
    L = max(1, min(L, T-1))
    W = T - L
    Xw_full = X.unfold(2, size=L, step=1)   # [N,C,T-L+1,L]
    Xw = Xw_full[:, :, :W, :]               # [N,C,W,L]
    Yw = X[:, :, L:]                        # [N,C,W]
    Xw = Xw.permute(0,2,1,3).contiguous().view(N*W, C, L)
    Yw = Yw.permute(0,2,1).contiguous().view(N*W, C)
    return Xw, Yw


# -----------------------------
# 콘솔 표 UI
# -----------------------------
def fmt_time(secs: float) -> str:
    if secs is None or np.isnan(secs): return "-"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def build_eval_table(phase: str, i:int, N:int, avg_mse:float, eta_sec:float):
    t = Table(expand=True, show_header=False, pad_edge=False, box=None)
    t.add_row("Phase", phase)
    t.add_row("Row", f"{i}/{N}")
    t.add_row("avg MSE", f"{avg_mse:.6f}" if np.isfinite(avg_mse) else "nan")
    t.add_row("ETA", fmt_time(eta_sec))
    return Panel(t, title="eval", border_style="magenta")


# -----------------------------
# 평가 (리치 박스 UI)
# -----------------------------
@torch.no_grad()
def eval_model(model, X: torch.Tensor, L: int, desc="eval", heartbeat_sec=5):
    dev = next(model.parameters()).device
    N, C, T = X.shape
    last = time.time()
    mse_list = []
    start = time.time()

    if USE_RICH:
        with Live(build_eval_table(desc, 0, N, float("nan"), None),
                  refresh_per_second=4, console=console) as live:
            for i in range(N):
                row = X[i].unsqueeze(0)
                Xw, Yw = build_windows_dataset(row, L)
                if Xw.numel() == 0: continue
                Xw, Yw = Xw.to(dev), Yw.to(dev)
                pred, _ = model(Xw)
                mse_list.append(F.mse_loss(pred, Yw).item())

                now = time.time()
                if now - last >= heartbeat_sec:
                    avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
                    done = i+1
                    rate = done / max(1, now-start)
                    eta = (N-done)/rate if rate>0 else None
                    live.update(build_eval_table(desc, done, N, avg_mse, eta))
                    last = now
            # 최종
            avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
            live.update(build_eval_table(desc, N, N, avg_mse, 0.0))
    else:
        for i in range(N):
            row = X[i].unsqueeze(0)
            Xw, Yw = build_windows_dataset(row, L)
            if Xw.numel() == 0: continue
            Xw, Yw = Xw.to(dev), Yw.to(dev)
            pred, _ = model(Xw)
            mse_list.append(F.mse_loss(pred, Yw).item())
        avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
        print(f"[{desc}] avg MSE={avg_mse:.6f}")

    return float(np.mean(mse_list)) if mse_list else float("nan")


# -----------------------------
# 플롯
# -----------------------------
def plot_samples(model, X: torch.Tensor, L: int, outdir: str, k: int = 3):
    ensure_outdir(outdir)
    dev = next(model.parameters()).device
    N, C, T = X.shape
    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    for i in range(min(k, N)):
        row = X[i]
        preds = torch.full((C, T), float("nan"))
        Lc = max(1, min(L, T-1))
        Xw, _ = build_windows_dataset(row.unsqueeze(0), Lc)
        with torch.no_grad():
            pred, _ = model(Xw.to(dev))
        preds[:, Lc:] = pred.cpu().T
        x_np = row.numpy(); p_np = preds.numpy()
        plt.figure(figsize=(12,4))
        for c in range(C):
            col = colors[c % len(colors)]
            plt.plot(range(T), x_np[c], color=col, lw=1.2, label=f"ch{c+1} true")
            m = ~np.isnan(p_np[c])
            if m.any():
                plt.plot(np.arange(T)[m], p_np[c][m], "--", color=col, lw=1.2, label=f"ch{c+1} pred")
        plt.legend(ncol=3, fontsize=9)
        plt.title(f"row {i} (context={Lc})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"row{i:04d}.png")); plt.close()


# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--csv1"); ap.add_argument("--csv2"); ap.add_argument("--csv3")
    ap.add_argument("--csv-list")
    ap.add_argument("--backbone", default="llm_ts")
    ap.add_argument("--context-len", type=int, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot-samples", type=int, default=3)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    ensure_outdir(args.out)
    X = parse_csvs(args)
    N,C,T = X.shape

    if args.context_len >= T: args.context_len = T-1
    if args.context_len < 1: args.context_len = 1

    model,_ = load_model_and_cfg(backbone=args.backbone, in_channels=C)
    model = model.to(device=args.device, dtype=torch.float32)

    if args.ckpt and os.path.exists(args.ckpt):
        sd=torch.load(args.ckpt,map_location="cpu")
        if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        print(f"[INFO] checkpoint loaded: {args.ckpt}", flush=True)

    print(f"[INFO] start | N={N},C={C},T={T},context={args.context_len}", flush=True)

    # 평가
    mse = eval_model(model, X, args.context_len, desc="infer", heartbeat_sec=5)
    plot_samples(model, X, args.context_len, args.out, args.plot_samples)

    with open(os.path.join(args.out,"infer_report.txt"),"w") as f:
        f.write(f"MSE {mse:.6f}\n")

    print(f"[DONE] inference results saved in {args.out}", flush=True)


if __name__=="__main__":
    main()
