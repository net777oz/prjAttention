# run_demo.py
# - 옵션(--ckpt)으로 파인튜닝된 .pt 가중치 불러오기
# - 모의 데이터 생성(X: [B,T,C], y_true: [B,H,C])
# - 예측 실행, PNG 저장(입력/정답/예측을 채널별 서브플롯), 임베딩 UMAP 저장

import argparse
import os
import torch

from ttm_flow.data import gen_mock_batch, gen_future_labels_from_series
from ttm_flow.model import load_model_and_cfg, extract_embeddings_ttm
from ttm_flow.pipeline import eval_step, cluster_embeddings
from ttm_flow.viz import plot_batch_examples, umap_scatter_2d

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CUDA 튜닝(있으면 살짝 이득)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="(optional) path to finetuned .pt checkpoint (e.g., checkpoints/ttm_r2_ft1step_best.pt)",
    )
    parser.add_argument("--B", type=int, default=32, help="batch size for demo data")
    parser.add_argument("--T", type=int, default=512, help="context length")
    parser.add_argument("--H", type=int, default=96, help="prediction horizon")
    parser.add_argument("--C", type=int, default=3, help="num of channels")
    parser.add_argument("--n_examples", type=int, default=3, help="how many samples to save as PNG")
    args = parser.parse_args()

    B, T, H, C = args.B, args.T, args.H, args.C

    # -----------------------------
    # 1) 모의 데이터 생성
    #    X: [B, T, C], y_true: [B, H, C]
    # -----------------------------
    X_np, _ = gen_mock_batch(B, T, C=C, missing_prob=0.05, seed=42)
    y_true_np = gen_future_labels_from_series(X_np, H)  # [B,H,C]

    X = torch.from_numpy(X_np).to(torch.float32).to(DEVICE, non_blocking=True)

    # -----------------------------
    # 2) 모델 로드 (+옵션 체크포인트)
    # -----------------------------
    model, cfg = load_model_and_cfg(
        MODEL_ID, context_len=T, pred_len=H, device=DEVICE
    )
    if args.ckpt is not None and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        print(f"[ckpt] loaded finetuned weights from {args.ckpt}")

    model.eval()

    # -----------------------------
    # 3) 예측 실행
    # -----------------------------
    with torch.no_grad():
        out, yhat = eval_step(model, X)  # yhat: [B,H,C] or None
    if yhat is None:
        raise RuntimeError("Model did not return predictions.")
    print(f"[eval] prediction shape: {tuple(yhat.shape)}")

    # -----------------------------
    # 4) 시각화 저장 (PNG) — 아까 스타일로 복귀!
    #     입력(검정) + 정답(초록) + 예측(빨강 점선), 채널별 서브플롯, 샘플별 PNG
    # -----------------------------
    os.makedirs("plots", exist_ok=True)
    yhat_np = yhat.detach().cpu().numpy()  # [B,H,C]

    # 아까 스타일 함수 사용
    plot_batch_examples(
        X_np,                 # [B,T,C]
        y_true_np,            # [B,H,C]
        yhat_np,              # [B,H,C]
        n_examples=args.n_examples,
        outdir="plots",
        tag="direct"          # 파일명 접두사: direct_sample_0.png ...
    )

    # -----------------------------
    # 5) 임베딩 추출 → KMeans → UMAP 2D 저장
    # -----------------------------
    emb = extract_embeddings_ttm(out, X)  # torch.Tensor [B, d*]
    labels, _ = cluster_embeddings(emb, k=4, seed=777)
    umap_scatter_2d(
        emb.detach().cpu().numpy(),
        labels,
        outpath="plots/umap_2d.png",
        title="UMAP 2D — TTM embeddings",
    )

if __name__ == "__main__":
    main()
