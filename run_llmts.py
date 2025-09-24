# -*- coding: utf-8 -*-
"""
LLM-스타일 시계열 Transformer (llm_ts) 통합 스크립트
- 모드: --mode {train, finetune, infer}
- CSV(헤더/인덱스 없음): 행=아이템(N), 열=시간(T)
- 다변량 입력: --csv1/2/3 또는 --csv-list (단일 CSV는 --csv)
- 교사강요(teacher-forcing) 평가 + 리치 박스 UI
- AMP, torch.compile 지원
"""

import argparse, os, time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
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
    console = None  # fallback to print


# -----------------------------
# 유틸
# -----------------------------
def set_seed(seed: int = 777):
    import torch.backends.cudnn as cudnn
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def ensure_outdir(p: str): os.makedirs(p, exist_ok=True)

def read_csv_no_header(path: str) -> np.ndarray:
    x = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return x[None, :] if x.ndim == 1 else x

def to_tensor(x: np.ndarray) -> torch.Tensor:
    # DataLoader 워커 안전한 CPU float32
    return torch.from_numpy(x).to(dtype=torch.float32).contiguous()

def stack_multivar(csv_paths) -> torch.Tensor:
    mats = [to_tensor(read_csv_no_header(p)) for p in csv_paths]  # CPU
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
        # 단일 CSV가 다변량 3개를 세 줄 간격으로 섞어둔 케이스 보정
        return torch.stack([M[0::3], M[1::3], M[2::3]], dim=1) if (N>=3 and N%3==0) else M.unsqueeze(1)
    raise SystemExit("CSV 입력 필요: --csv-list 또는 --csv1/--csv2/--csv3 또는 --csv")


# -----------------------------
# 롤링 윈도우 (teacher-forcing 학습/평가용)
# -----------------------------
def build_windows_dataset(X: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [N,C,T] -> Xw:[N*W,C,L], Yw:[N*W,C]
    unfold 창 개수는 T-L+1, 다음 시점 예측(T-L)에 맞춰 마지막 창 1개 drop
    """
    N, C, T = X.shape
    L = max(1, min(L, T-1))
    W = T - L
    Xw_full = X.unfold(dimension=2, size=L, step=1)   # [N,C,T-L+1,L]
    Xw = Xw_full[:, :, :W, :]                         # [N,C,W,L]
    Yw = X[:, :, L:]                                  # [N,C,W]
    Xw = Xw.permute(0,2,1,3).contiguous().view(N*W, C, L)  # [N*W,C,L]
    Yw = Yw.permute(0,2,1).contiguous().view(N*W, C)       # [N*W,C]
    return Xw, Yw


# -----------------------------
# 콘솔 표 UI
# -----------------------------
def fmt_time(secs: float) -> str:
    if secs is None or np.isnan(secs): return "-"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def build_train_table(ep:int, epochs:int, bi:int, num_batches:int,
                      done:int, total:int,
                      loss_cur:float, loss_avg:float, lr:float,
                      step_time:float, eta_sec:float, mem_mb:float,
                      bs:int, amp:bool, compiled:str|None):
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

def build_eval_table(phase: str, i:int, N:int, avg_mse:float, eta_sec:float):
    t = Table(expand=True, show_header=False, pad_edge=False, box=None)
    t.add_row("Phase", phase)
    t.add_row("Row", f"{i}/{N}")
    t.add_row("avg MSE", f"{avg_mse:.6f}" if np.isfinite(avg_mse) else "nan")
    t.add_row("ETA", fmt_time(eta_sec))
    return Panel(t, title="eval", border_style="magenta")


# -----------------------------
# 평가 (벡터화 + 하트비트 + 표UI)
# -----------------------------
@torch.no_grad()
def eval_model(model, X: torch.Tensor, L: int, desc="eval", heartbeat_sec=5):
    model.eval()  # ★ 중요: 평가 모드
    dev = next(model.parameters()).device
    N, C, T = X.shape
    last = time.time()
    mse_list, mae_list = [], []
    start = time.time()

    if USE_RICH:
        with Live(build_eval_table(desc, 0, N, float("nan"), None),
                  refresh_per_second=8, console=console) as live:
            for i in range(N):
                row = X[i].unsqueeze(0)
                Xw, Yw = build_windows_dataset(row, L)
                if Xw.numel() == 0:
                    continue
                Xw = Xw.to(dev, non_blocking=True)
                Yw = Yw.to(dev, non_blocking=True)
                pred, _ = model(Xw)
                mse_list.append(F.mse_loss(pred, Yw).item())
                mae_list.append(F.l1_loss(pred, Yw).item())

                now = time.time()
                if now - last >= heartbeat_sec:
                    avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
                    done = i + 1
                    rate = (done / max(1, now - start))
                    remaining = (N - done) / rate if rate > 0 else None
                    live.update(build_eval_table(desc, done, N, avg_mse, remaining))
                    last = now

            # 마지막 상태 한 번 더 업데이트
            avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
            live.update(build_eval_table(desc, N, N, avg_mse, 0.0))
    else:
        for i in range(N):
            row = X[i].unsqueeze(0)
            Xw, Yw = build_windows_dataset(row, L)
            if Xw.numel() == 0:
                continue
            Xw = Xw.to(dev, non_blocking=True)
            Yw = Yw.to(dev, non_blocking=True)
            pred, _ = model(Xw)
            mse_list.append(F.mse_loss(pred, Yw).item())
            mae_list.append(F.l1_loss(pred, Yw).item())
        avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
        print(f"[{desc}] avg MSE={avg_mse:.6f}")

    return (float(np.mean(mse_list)) if mse_list else float("nan"),
            float(np.mean(mae_list)) if mae_list else float("nan"))


# -----------------------------
# 플롯
# -----------------------------
def plot_samples(model, X: torch.Tensor, L: int, outdir: str, k: int = 3, prefix: str = "after"):
    ensure_outdir(outdir)
    model.eval()  # ★ 평가 모드
    dev = next(model.parameters()).device
    N, C, T = X.shape
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    for i in range(min(k, N)):
        row = X[i]  # CPU [C,T]
        preds = torch.full((C, T), float("nan"))  # CPU
        Lc = max(1, min(L, T-1))
        Xw, _ = build_windows_dataset(row.unsqueeze(0), Lc)  # [W,C,L] CPU
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
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{prefix}_row{i:04d}.png")); plt.close()


# -----------------------------
# 학습 루프 (리치 표 UI)
# -----------------------------
def train_all_epochs(model, dl: DataLoader, opt, scaler: GradScaler,
                     epochs:int, amp_enabled:bool=True,
                     log_every:int=50, heartbeat_sec:int=5, compiled:str|None=None):
    dev = next(model.parameters()).device
    total_steps = epochs * len(dl)
    step_counter = 0
    start = time.time()
    last = start
    avg_step_time = None

    def update_table(ep, bi, loss_cur, loss_avg, bs):
        nonlocal avg_step_time
        now = time.time()
        dt = now - last_times[0]
        last_times[0] = now
        if avg_step_time is None: avg_step_time = dt
        else: avg_step_time = 0.9*avg_step_time + 0.1*dt
        done = step_counter
        remaining = total_steps - done
        eta = remaining * (avg_step_time if avg_step_time else 0.0)
        lr = opt.param_groups[0].get("lr", 0.0)
        mem = (torch.cuda.memory_allocated()/ (1024**2)) if torch.cuda.is_available() else 0.0

        if USE_RICH:
            table = build_train_table(ep, epochs, bi, len(dl),
                                      done, total_steps, loss_cur, loss_avg, lr,
                                      avg_step_time if avg_step_time else 0.0,
                                      eta, mem, bs, amp_enabled, compiled)
            live.update(table)
        else:
            print(f"\r[train] ep {ep}/{epochs} batch {bi}/{len(dl)} "
                  f"done {done}/{total_steps} loss={loss_cur:.6f} avg={loss_avg:.6f} "
                  f"lr={lr:.2e} step={avg_step_time:.2f}s ETA={fmt_time(eta)} mem={int(mem)}MB",
                  end="", flush=True)

    if USE_RICH:
        live = Live(Panel("initializing...", title="train", border_style="cyan"),
                    refresh_per_second=10, console=console)
        live.start()
    else:
        live = None

    try:
        for ep in range(1, epochs+1):
            model.train()
            total, steps = 0.0, 0
            last_times = [time.time()]

            for bi, (xb, yb) in enumerate(dl):
                xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                if amp_enabled:
                    with autocast():
                        pred, _ = model(xb)
                        loss = F.mse_loss(pred, yb)
                    if not torch.isfinite(loss):
                        if (bi % max(1, log_every)) == 0:
                            print(f"\n[WARN] non-finite loss at ep {ep} batch {bi}, skip.", flush=True)
                        step_counter += 1
                        update_table(ep, bi, float("nan"), float("nan"), xb.size(0))
                        continue
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    pred, _ = model(xb); loss = F.mse_loss(pred, yb)
                    if not torch.isfinite(loss):
                        if (bi % max(1, log_every)) == 0:
                            print(f"\n[WARN] non-finite loss at ep {ep} batch {bi}, skip.", flush=True)
                        step_counter += 1
                        update_table(ep, bi, float("nan"), float("nan"), xb.size(0))
                        continue
                    loss.backward(); opt.step()

                total += loss.item(); steps += 1; step_counter += 1
                avg = total / steps

                # 주기/하트비트에 맞춰 갱신
                now = time.time()
                if ((bi % max(1, log_every)) == 0) or (now - last >= heartbeat_sec) or (step_counter == total_steps):
                    update_table(ep, bi, loss.item(), avg, xb.size(0))
                    last = now

            print(f"\n[EPOCH {ep}/{epochs}] avg_loss={total/max(1,steps):.6f}", flush=True)

    finally:
        if USE_RICH:
            live.stop()
        else:
            print()  # 줄바꿈


# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # 입력
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--csv1", type=str, default=None)
    ap.add_argument("--csv2", type=str, default=None)
    ap.add_argument("--csv3", type=str, default=None)
    ap.add_argument("--csv-list", type=str, default=None, help='Comma list: "a.csv,b.csv,..."')

    # 모드/모델/학습/출력
    ap.add_argument("--mode", type=str, required=True, choices=["train","finetune","infer"])
    ap.add_argument("--backbone", type=str, default="llm_ts")
    ap.add_argument("--context-len", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--plot-samples", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=50)

    # 속도/리소스 옵션
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (AMP)")
    ap.add_argument("--compile", type=str, default="", help='torch.compile mode: "", "reduce-overhead", "max-autotune"')

    # 체크포인트
    ap.add_argument("--ckpt", type=str, default=None, help="pretrained checkpoint for finetune/infer")

    args = ap.parse_args()
    set_seed(args.seed); ensure_outdir(args.out)

    # 데이터 로드 (CPU)
    X = parse_csvs(args)   # [N,C,T]
    N, C, T = X.shape

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

    # torch.compile
    compiled_mode = None
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile)
            compiled_mode = args.compile
            print(f"[INFO] torch.compile enabled: mode={args.compile}", flush=True)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}", flush=True)

    print(f"[INFO] start | mode={args.mode}, N={N},C={C},T={T},context={args.context_len},params={sum(p.numel() for p in model.parameters())}", flush=True)

    # ---- 모드 분기 ----
    if args.mode == "infer":
        if not args.ckpt: raise SystemExit("--mode infer 는 --ckpt 가 필요합니다.")
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

        bmse, bmae = eval_model(model, X, args.context_len, desc="infer", heartbeat_sec=5)
        print(f"[INFER] MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)
        plot_samples(model, X, args.context_len, args.out, k=args.plot_samples, prefix="infer")
        with open(os.path.join(args.out, "infer_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"infer MSE {bmse:.6f} MAE {bmae:.6f}\n")
        print(f"[DONE] infer saved to {args.out}", flush=True)
        return

    # train 또는 finetune
    if args.mode == "finetune":
        if not args.ckpt: raise SystemExit("--mode finetune 는 --ckpt 가 필요합니다.")
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    # BEFORE eval (초기 성능 파악)
    print("[INFO] starting BEFORE eval...", flush=True)
    bmse, bmae = eval_model(model, X, args.context_len, desc="before", heartbeat_sec=5)
    print(f"[BEFORE] MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)

    # 학습 준비
    Xw, Yw = build_windows_dataset(X, args.context_len)
    ds = TensorDataset(Xw, Yw)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=False, persistent_workers=True if args.num_workers>0 else False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # 학습
    train_all_epochs(model, dl, opt, scaler,
                     epochs=args.epochs, amp_enabled=args.amp,
                     log_every=args.log_every, heartbeat_sec=5, compiled=compiled_mode)

    # AFTER eval
    print("[INFO] starting AFTER eval...", flush=True)
    amse, amae = eval_model(model, X, args.context_len, desc="after", heartbeat_sec=5)
    print(f"[AFTER ] MSE={amse:.6f} MAE={amae:.6f}", flush=True)

    # 저장
    with open(os.path.join(args.out, "train_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"before MSE {bmse:.6f} MAE {bmae:.6f}\n")
        f.write(f"after  MSE {amse:.6f} MAE {amae:.6f}\n")
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    plot_samples(model, X, args.context_len, args.out, k=args.plot_samples, prefix="after")
    print(f"[DONE] model/report/plots saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
