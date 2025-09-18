# run_finetune_1step.py
# 1-스텝 전용 파인튜닝: 롤링 윈도우로 학습하고, before/after 평가 및 PNG 출력
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from ttm_flow.data import gen_mock_batch, make_rolling_windows
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
    plt.figure(figsize=(5,5))
    plt.scatter(yt, yp, s=10, alpha=0.5)
    lim = (min(yt.min(), yp.min()), max(yt.max(), yp.max()))
    plt.plot(lim, lim, 'k--', linewidth=1)
    plt.xlabel("True (1-step)"); plt.ylabel("Pred (1-step)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    print(f"[plot] saved {outpath}")

def main():
    # -----------------------------
    # 0) 모의 데이터 & 모델
    # -----------------------------
    B, C = 32, 3
    T_ctx   = 512    # 모델 컨텍스트 길이
    H_model = 96     # 모델의 기본 예측 길이(R2)
    H_train = 1      # 파인튜닝은 1-step만
    T_long  = 640    # 롤링윈도우 확보용(≥ T_ctx + H_train)

    # mock data (길게 생성)
    X_np, _ = gen_mock_batch(B, T_long, C=C, missing_prob=0.05, seed=777)

    # 모델 로드 (컨텍스트는 512로 고정)
    model, cfg = load_model_and_cfg(MODEL_ID, context_len=T_ctx, pred_len=H_model, device=DEVICE)
    model.eval()

    # -----------------------------
    # 1) 롤링 윈도우 생성 (컨텍스트 512 → 다음 1-step)
    # -----------------------------
    Xw, Yw = make_rolling_windows(X_np, context_len=T_ctx, horizon=H_train, step=16)

    # 텐서(일단 CPU 유지 — DataLoader가 CPU→GPU 전송 최적화)
    Xw_t = torch.from_numpy(Xw).to(torch.float32)  # [N, 512, C]
    Yw_t = torch.from_numpy(Yw).to(torch.float32)  # [N, 1,   C]

    # 학습/검증 분할 (CPU 텐서 상태 유지)
    N = Xw_t.size(0)
    n_val = max(64, int(0.2 * N))
    n_train = N - n_val
    X_train, X_val = random_split(Xw_t, [n_train, n_val], generator=torch.Generator().manual_seed(123))
    y_train, y_val = random_split(Yw_t, [n_train, n_val], generator=torch.Generator().manual_seed(123))

    # 텐서로 모아두기(여전히 CPU)
    Xtr = torch.stack([X_train[i] for i in range(len(X_train))], dim=0)  # CPU
    ytr = torch.stack([y_train[i] for i in range(len(y_train))], dim=0)  # CPU
    Xva = torch.stack([X_val[i]   for i in range(len(X_val))],   dim=0)  # CPU
    yva = torch.stack([y_val[i]   for i in range(len(y_val))],   dim=0)  # CPU

    # -----------------------------
    # 2) 파인튜닝 전 1-step MSE
    # -----------------------------
    before_val = eval_1step_mse(model, Xva, yva)
    print(f"[before] 1-step val MSE: {before_val:.6f}")

    # -----------------------------
    # 3) 1-step 파인튜닝 (🔟 에폭)
    # -----------------------------
    os.makedirs("checkpoints", exist_ok=True)
    finetune_1step(
        model,
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        epochs=300,                 # ✅ 10 에폭
        batch_size=256,            # GPU 여유되면 256~512 추천
        lr=2e-4,
        clip=1.0,
        num_workers=2,
        save_best_path="checkpoints/ttm_r2_ft1step_best.pt",  # 베스트 저장
        patience=4,                # 4 에폭 동안 개선 없으면 얼리스탑(원하면 0으로 끄기)
    )

    # -----------------------------
    # 4) 파인튜닝 후 1-step MSE & 산점도
    # -----------------------------
    after_val = eval_1step_mse(model, Xva, yva)
    print(f"[after ] 1-step val MSE: {after_val:.6f}")

    # 산점도(검증셋 일부) 저장 (GPU/AMP로 추론 후 CPU로 내림)
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
        "plots/ft1step_val_scatter.png",
        f"1-step fine-tune (val) — MSE {after_val:.4f}"
    )

    # -----------------------------
    # 5) 임베딩 UMAP(선택) — 파인튜닝 후 상태
    # -----------------------------
    out_eval, _ = eval_step(model, Xb)  # 위에서 만든 Xb 재사용(GPU)
    emb = extract_embeddings_ttm(out_eval, Xb)
    labels, _ = cluster_embeddings(emb, k=4, seed=777)
    umap_scatter_2d(emb.detach().cpu().numpy(), labels, "plots/ft1step_umap_2d.png",
                    "UMAP 2D — 1-step FT (val subset)")

    # -----------------------------
    # 6) 최종 체크포인트 저장 (현재 가중치)
    # -----------------------------
    torch.save(model.state_dict(), "checkpoints/ttm_r2_finetuned_1step.pt")
    print("[ckpt] saved checkpoints/ttm_r2_finetuned_1step.pt")
    print("[ckpt] best model (val 기준)는 checkpoints/ttm_r2_ft1step_best.pt 에 저장됨")

if __name__ == "__main__":
    main()
