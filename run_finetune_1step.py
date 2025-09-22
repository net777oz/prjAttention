# run_finetune_1step.py
# 1-스텝 파인튜닝: CSV 3개(행=아이템, 열=시간) -> (B,T,3) 로 결합해 롤링윈도우 학습/평가
# 필수 인자: --csv1 --csv2 --csv3
# 나머지는 전부 기본값(원하면 플래그로 덮어쓰기)

import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from ttm_flow.data import make_rolling_windows, load_triplet_csvs
from ttm_flow.model import load_model_and_cfg, extract_embeddings_ttm
from ttm_flow.pipeline import (
    eval_step,
    cluster_embeddings,
    finetune_1step,
    eval_1step_mse,
    _pick_prediction_tensor,
)
from ttm_flow.viz import umap_scatter_2d

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CUDA 튜닝(있으면 약간 이득)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ────────────────────────────
# freq_token 주입 어댑터(기본 0)
# ────────────────────────────
class TTMFreqAdapter(nn.Module):
    def __init__(self, base: nn.Module, default_token: int = 0):
        super().__init__()
        self.base = base
        self.default_token = int(default_token)

    def forward(self, x, **kwargs):
        B = x.size(0)
        ft = kwargs.get("freq_token", self.default_token)
        if torch.is_tensor(ft):
            if ft.ndim == 0:
                ft = ft.view(1).expand(B)
            elif ft.ndim == 1 and ft.size(0) != B:
                ft = ft.expand(B)
            ft = ft.to(dtype=torch.long, device=x.device)
        else:
            ft = torch.full((B,), int(ft), dtype=torch.long, device=x.device)
        kwargs["freq_token"] = ft
        return self.base(x, **kwargs)

def unwrap_model(m: nn.Module) -> nn.Module:
    return getattr(m, "base", m)


def scatter_pred_vs_true(y_true, y_pred, outpath: str, title: str):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    plt.figure(figsize=(5,5))
    plt.scatter(yt, yp, s=10, alpha=0.5)
    lim = (min(yt.min(), yp.min()), max(yt.max(), yp.max()))
    plt.plot(lim, lim, 'k--', linewidth=1)
    plt.xlabel("True (1-step)"); plt.ylabel("Pred (1-step)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    print(f"[plot] saved {outpath}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TTM-R2 (1-step) from 3 CSVs (C=3).")
    # ▶ 필수
    p.add_argument("--csv1", required=True, help="변수1 CSV (행=아이템, 열=시간)")
    p.add_argument("--csv2", required=True, help="변수2 CSV (행=아이템, 열=시간)")
    p.add_argument("--csv3", required=True, help="변수3 CSV (행=아이템, 열=시간)")
    # ▶ 선택(전부 기본값 존재; 주면 덮어쓰기)
    p.add_argument("--context_len", type=int, default=180, help="컨텍스트 길이(기본 180)")
    p.add_argument("--roll_step",   type=int, default=16,  help="롤링 윈도우 슬라이딩 간격(기본 16)")
    p.add_argument("--epochs",      type=int, default=300, help="기본 300")
    p.add_argument("--batch_size",  type=int, default=256, help="기본 256")
    p.add_argument("--lr",          type=float, default=2e-4, help="기본 2e-4")
    p.add_argument("--patience",    type=int, default=4,   help="얼리스탑 기본 4 (0=off)")
    p.add_argument("--num_workers", type=int, default=2,   help="기본 2")
    p.add_argument("--seed",        type=int, default=123, help="기본 123")
    p.add_argument("--freq_token",  type=int, default=0,   help="TTM freq_token(기본 0)")
    p.add_argument("--save_dir",    type=str, default="checkpoints", help="기본 checkpoints/")
    p.add_argument("--outdir",      type=str, default="plots",       help="기본 plots/")
    return p.parse_args()


def safe_load_ttm_model(model_id: str, context_len: int, pred_len: int):
    """pred_len=1로 로드. 가능한 경우 force_return='rolling' 사용."""
    try:
        model, cfg = load_model_and_cfg(
            model_id,
            context_len=context_len,
            pred_len=pred_len,
            device=DEVICE,
            force_return="rolling",
        )
        return model, cfg
    except (TypeError, ValueError):
        model, cfg = load_model_and_cfg(
            model_id,
            context_len=context_len,
            pred_len=pred_len,
            device=DEVICE,
        )
        return model, cfg


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) CSV → (B,T,3) 로드 (NaN은 기본 mean 처리: load_triplet_csvs 기본값)
    X_np = load_triplet_csvs(args.csv1, args.csv2, args.csv3)
    B, T, C = X_np.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    print(f"[data] loaded: items={B}, timesteps={T}, channels={C}")

    # 2) 컨텍스트 자동 보정 (짧으면 줄임), horizon=1 고정
    H = 1
    ctx = max(16, min(args.context_len, T - H))
    if ctx < args.context_len:
        print(f"[warn] context_len {args.context_len} -> {ctx} (timesteps={T}, horizon={H})")

    # 3) 롤링 윈도우 생성
    Xw, Yw = make_rolling_windows(X_np, context_len=ctx, horizon=H, step=args.roll_step)
    Xw_t = torch.from_numpy(Xw).to(torch.float32)
    Yw_t = torch.from_numpy(Yw).to(torch.float32)

    # 4) train/val split
    N = Xw_t.size(0)
    n_val = max(64, int(0.2 * N))
    n_tr  = N - n_val
    gen = torch.Generator().manual_seed(args.seed)
    X_train, X_val = random_split(Xw_t, [n_tr, n_val], generator=gen)
    y_train, y_val = random_split(Yw_t, [n_tr, n_val], generator=gen)
    Xtr = torch.stack([X_train[i] for i in range(len(X_train))], dim=0)
    ytr = torch.stack([y_train[i] for i in range(len(y_train))], dim=0)
    Xva = torch.stack([X_val[i]   for i in range(len(X_val))],   dim=0)
    yva = torch.stack([y_val[i]   for i in range(len(y_val))],   dim=0)

    # 5) 모델 로드(+freq_token 어댑터)
    base_model, _ = safe_load_ttm_model(MODEL_ID, context_len=ctx, pred_len=H)
    model = TTMFreqAdapter(base_model, default_token=args.freq_token)
    model.eval()

    # 6) 파인튜닝 전 MSE
    before_val = eval_1step_mse(model, Xva, yva)
    print(f"[before] 1-step val MSE: {before_val:.6f}")

    # 7) 파인튜닝
    best_path = os.path.join(args.save_dir, "ttm_r2_ft1step_best.pt")
    finetune_1step(
        model,
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clip=1.0,
        num_workers=args.num_workers,
        save_best_path=best_path,
        patience=args.patience,
    )

    # 8) 파인튜닝 후 MSE
    after_val = eval_1step_mse(model, Xva, yva)
    print(f"[after ] 1-step val MSE: {after_val:.6f}")

    # 9) 시각화(검증 일부)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
        Xb = Xva[:1024].to(DEVICE, non_blocking=True)
        out = model(Xb, return_dict=True)
        yhat = _pick_prediction_tensor(out)
        if yhat is None:
            raise RuntimeError("Predictions not found in model output.")
        yhat1 = yhat[:, 0:1, :].detach().cpu().numpy()

    scatter_pred_vs_true(
        yva[:1024].detach().cpu().numpy(),
        yhat1,
        os.path.join(args.outdir, "ft1step_val_scatter.png"),
        f"1-step fine-tune (val) — MSE {after_val:.4f}"
    )

    out_eval, _ = eval_step(model, Xb)
    emb = extract_embeddings_ttm(out_eval, Xb)
    labels, _ = cluster_embeddings(emb, k=4, seed=777)
    umap_scatter_2d(
        emb.detach().cpu().numpy(),
        labels,
        os.path.join(args.outdir, "ft1step_umap_2d.png"),
        f"UMAP 2D — 1-step FT (ctx={ctx})"
    )

    # 10) 체크포인트 저장 (어댑터 제외 실제 모델 가중치)
    last_path = os.path.join(args.save_dir, "ttm_r2_finetuned_1step.pt")
    torch.save(unwrap_model(model).state_dict(), last_path)
    print(f"[ckpt] saved {last_path}")
    print(f"[ckpt] best model (val): {best_path}")


if __name__ == "__main__":
    main()
