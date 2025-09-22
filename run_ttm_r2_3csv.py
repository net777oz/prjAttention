# run_ttm_r2_3csv.py
# 3-파일(C=3) 입력 전용. mode=train(파인튜닝) | infer(추론만)
# - CSV 3개(헤더/인덱스 없음, 행=아이템, 열=시간)를 (N_items, T_steps, 3)으로 결합
# - 롤링 윈도우로 1-step 예측 파이프라인 유지
# - train: before/after MSE, 체크포인트 저장, 산점도/UMAP 출력
# - infer: 체크포인트 로드 후 평가 및 산점도/UMAP만 출력(학습 없음)

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from ttm_flow.data import make_rolling_windows
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

# CUDA 튜닝
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def scatter_pred_vs_true(y_true, y_pred, outpath: str, title: str):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    plt.figure(figsize=(5, 5))
    plt.scatter(yt, yp, s=10, alpha=0.5)
    lim = (min(yt.min(), yp.min()), max(yt.max(), yp.max()))
    plt.plot(lim, lim, 'k--', linewidth=1)
    plt.xlabel("True (1-step)"); plt.ylabel("Pred (1-step)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    print(f"[plot] saved {outpath}")


def load_triplet_csvs(csv1: str, csv2: str, csv3: str) -> np.ndarray:
    """
    세 CSV 파일을 읽어 (N_items, T_steps, 3) float32 배열을 반환.
    가정:
      - 헤더/인덱스 없음, 구분자 ','
      - 세 파일의 shape 동일
    """
    def _load_one(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"CSV must be 2D (items x timesteps). Got {arr.shape} for {path}")
        return arr

    A = _load_one(csv1)
    B = _load_one(csv2)
    C = _load_one(csv3)

    if A.shape != B.shape or A.shape != C.shape:
        raise ValueError(f"CSV shapes must match: A={A.shape}, B={B.shape}, C={C.shape}")

    X_np = np.stack([A, B, C], axis=-1).astype(np.float32)  # (N, T, 3)
    return X_np


def parse_args():
    p = argparse.ArgumentParser(description="TTM-R2 1-step fine-tune/infer from three CSV inputs (C=3).")
    p.add_argument("--mode", choices=["train", "infer"], required=True,
                   help="train: 파인튜닝 수행, infer: 학습 없이 추론/평가만")
    p.add_argument("--csv1", required=True, help="변수1 CSV (행=아이템, 열=시간)")
    p.add_argument("--csv2", required=True, help="변수2 CSV (행=아이템, 열=시간)")
    p.add_argument("--csv3", required=True, help="변수3 CSV (행=아이템, 열=시간)")
    p.add_argument("--context_len", type=int, default=512, help="모델 컨텍스트 길이 (기본 512)")
    p.add_argument("--horizon", type=int, default=1, help="예측 스텝(이 스크립트는 1 권장)")
    p.add_argument("--roll_step", type=int, default=16, help="롤링 윈도우 슬라이딩 간격")
    # 학습 관련
    p.add_argument("--epochs", type=int, default=300, help="파인튜닝 에폭 수")
    p.add_argument("--batch_size", type=int, default=256, help="배치 크기")
    p.add_argument("--lr", type=float, default=2e-4, help="러닝레이트")
    p.add_argument("--patience", type=int, default=4, help="얼리스탑 patience (0=off)")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=123, help="랜덤 시드")
    # 체크포인트
    p.add_argument("--save_dir", type=str, default="checkpoints", help="체크포인트 저장 폴더")
    p.add_argument("--ckpt", type=str, default="",
                   help="infer 모드 또는 로드 강제 시 사용할 체크포인트 경로(.pt)")
    # 출력
    p.add_argument("--outdir", type=str, default="plots",
                   help="산점도/UMAP 이미지 저장 폴더")
    return p.parse_args()


def prepare_windows(X_np: np.ndarray, context_len: int, horizon: int, roll_step: int):
    N_items, T_steps, C = X_np.shape
    if T_steps < context_len + horizon:
        raise ValueError(
            f"Not enough time steps. Need at least context_len+horizon={context_len + horizon}, "
            f"but got T_steps={T_steps}."
        )
    Xw, Yw = make_rolling_windows(X_np, context_len=context_len, horizon=horizon, step=roll_step)
    Xw_t = torch.from_numpy(Xw).to(torch.float32)  # [N_win, context_len, C]
    Yw_t = torch.from_numpy(Yw).to(torch.float32)  # [N_win, horizon, C]
    return Xw_t, Yw_t


def split_train_val(Xw_t: torch.Tensor, Yw_t: torch.Tensor, seed: int):
    N = Xw_t.size(0)
    n_val = max(64, int(0.2 * N))
    n_train = N - n_val
    gen = torch.Generator().manual_seed(seed)
    X_train, X_val = random_split(Xw_t, [n_train, n_val], generator=gen)
    y_train, y_val = random_split(Yw_t, [n_train, n_val], generator=gen)
    Xtr = torch.stack([X_train[i] for i in range(len(X_train))], dim=0)
    ytr = torch.stack([y_train[i] for i in range(len(y_train))], dim=0)
    Xva = torch.stack([X_val[i]   for i in range(len(X_val))],   dim=0)
    yva = torch.stack([y_val[i]   for i in range(len(y_val))],   dim=0)
    return Xtr, ytr, Xva, yva


def maybe_load_ckpt(model: torch.nn.Module, ckpt_path: str):
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        print(f"[ckpt] loaded weights from {ckpt_path}")
    else:
        if ckpt_path:
            print(f"[ckpt] specified but not found: {ckpt_path} (skip loading)")
    return model


def safe_load_ttm_model(model_id: str, context_len: int, horizon: int):
    """
    pred_len=1(=horizon) + force_return='rolling'로 TTM을 안전 로드.
    1) ttm_flow.model.load_model_and_cfg 시도(가능하면 force_return 전달)
    2) 실패 시 tsfm_public.toolkit.get_model 로 폴백(여긴 device 인자 X)
       → 로드 후 model.to(DEVICE)
    """
    pred_len = max(1, int(horizon))

    # 1) 우선 ttm_flow의 헬퍼 시도
    try:
        model, cfg = load_model_and_cfg(
            model_id,
            context_len=context_len,
            pred_len=pred_len,
            device=DEVICE,
            force_return="rolling",   # 최신 시그니처에서만 동작
        )
        return model, cfg
    except TypeError:
        # 구버전 시그니처(=force_return 미지원)일 수 있음 → 아래로 폴백
        pass
    except ValueError as e:
        # "Set `force_return=rolling`" 등 → 폴백
        if "force_return=rolling" not in str(e):
            raise

    # 2) 직접 로드 (tsfm_public) — 여기서는 device 인자 금지!
    from tsfm_public.toolkit.get_model import get_model
    model = get_model(
        model_id,
        context_length=context_len,
        prediction_length=pred_len,
        force_return="rolling",
    )
    # 로드 후에 장치 이동
    model.to(DEVICE)
    cfg = {"context_length": context_len, "prediction_length": pred_len}
    return model, cfg


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 데이터 로드
    X_np = load_triplet_csvs(args.csv1, args.csv2, args.csv3)
    N_items, T_steps, C = X_np.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    print(f"[data] loaded: items={N_items}, timesteps={T_steps}, channels={C}")

    # 2) 컨텍스트 길이 자동 보정 (T_steps < context_len + horizon 인 경우 축소)
    H_train = args.horizon
    effective_ctx = max(16, min(args.context_len, T_steps - H_train))
    if effective_ctx < args.context_len:
        print(f"[warn] context_len {args.context_len} -> {effective_ctx} (자동 축소: timesteps={T_steps}, horizon={H_train})")

    # 3) 모델 로드: 1-step만 필요 → pred_len=1로 안전 로드(+ rolling 강제)
    model, _ = safe_load_ttm_model(MODEL_ID, context_len=effective_ctx, horizon=H_train)
    model.eval()

    # infer 모드: 체크포인트 로드 시도(있는 경우)
    if args.mode == "infer":
        default_best = os.path.join(args.save_dir, "ttm_r2_ft1step_best.pt")
        default_last = os.path.join(args.save_dir, "ttm_r2_finetuned_1step.pt")
        ckpt_try = args.ckpt or (default_best if os.path.isfile(default_best) else (default_last if os.path.isfile(default_last) else ""))
        if ckpt_try:
            maybe_load_ckpt(model, ckpt_try)

    # 4) 윈도우 생성/분할 (보정된 컨텍스트 길이 사용)
    Xw_t, Yw_t = prepare_windows(X_np, context_len=effective_ctx, horizon=H_train, roll_step=args.roll_step)
    Xtr, ytr, Xva, yva = split_train_val(Xw_t, Yw_t, seed=args.seed)

    # 공통: 평가/시각화용 배치
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    Xb_gpu = Xva[:1024].to(DEVICE, non_blocking=True)

    if args.mode == "train":
        # 5) 파인튜닝 전 MSE
        before_val = eval_1step_mse(model, Xva, yva)
        print(f"[before] 1-step val MSE: {before_val:.6f}")

        # 6) 파인튜닝
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

        # 7) 파인튜닝 후 MSE
        after_val = eval_1step_mse(model, Xva, yva)
        print(f"[after ] 1-step val MSE: {after_val:.6f}")

        # 8) 산점도/UMAP (파인튜닝 후 가중치 기준)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            out = model(Xb_gpu, return_dict=True)
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

        out_eval, _ = eval_step(model, Xb_gpu)
        emb = extract_embeddings_ttm(out_eval, Xb_gpu)
        labels, _ = cluster_embeddings(emb, k=4, seed=777)
        umap_scatter_2d(
            emb.detach().cpu().numpy(),
            labels,
            os.path.join(args.outdir, "ft1step_umap_2d.png"),
            f"UMAP 2D — 1-step FT (ctx={effective_ctx})"
        )

        # 9) 최종 체크포인트 저장(현재 가중치)
        last_path = os.path.join(args.save_dir, "ttm_r2_finetuned_1step.pt")
        torch.save(model.state_dict(), last_path)
        print(f"[ckpt] saved {last_path}")
        print(f"[ckpt] best model (val 기준): {best_path}")

    else:  # infer
        # 5) 학습 없이 바로 MSE 평가
        val_mse = eval_1step_mse(model, Xva, yva)
        print(f"[infer] 1-step val MSE: {val_mse:.6f}")

        # 6) 산점도/UMAP (현재 가중치 기준)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            out = model(Xb_gpu, return_dict=True)
            yhat = _pick_prediction_tensor(out)
            if yhat is None:
                raise RuntimeError("Predictions not found in model output.")
            yhat1 = yhat[:, 0:1, :].detach().cpu().numpy()

        scatter_pred_vs_true(
            yva[:1024].detach().cpu().numpy(),
            yhat1,
            os.path.join(args.outdir, "infer_val_scatter.png"),
            f"1-step inference (val) — MSE {val_mse:.4f}"
        )

        out_eval, _ = eval_step(model, Xb_gpu)
        emb = extract_embeddings_ttm(out_eval, Xb_gpu)
        labels, _ = cluster_embeddings(emb, k=4, seed=777)
        umap_scatter_2d(
            emb.detach().cpu().numpy(),
            labels,
            os.path.join(args.outdir, "infer_umap_2d.png"),
            f"UMAP 2D — 1-step Inference (ctx={effective_ctx})"
        )


if __name__ == "__main__":
    main()
