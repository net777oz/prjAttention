# run_demo.py
# - CSV 3종(--csv1/--csv2/--csv3) 또는 모의데이터로 X:[B,T,3] 구성
# - H(예측 길이) 자유롭게 설정 가능 (예: H=5)
# - TinyTimeMixer 일부 리비전에서 필요한 freq_token을 자동 주입(옵션 --freq_token, 기본 0)
# - NaN 처리는 기본(mean)로 고정
# - 예측 실행, PNG 저장(입력/정답/예측: 채널별 서브플롯), 임베딩 UMAP 저장

import argparse
import os
import torch
import torch.nn as nn

from ttm_flow.data import (
    gen_mock_batch,
    gen_future_labels_from_series,
    load_triplet_csvs,   # CSV 로더 (B,T,3 반환; nan_policy 기본 mean)
)
from ttm_flow.model import load_model_and_cfg, extract_embeddings_ttm
from ttm_flow.pipeline import eval_step, cluster_embeddings
from ttm_flow.viz import plot_batch_examples, umap_scatter_2d

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CUDA 튜닝
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ---------- freq_token 어댑터 (기본 0, CLI에서 변경 가능) ----------
class TTMFreqAdapter(nn.Module):
    """
    TinyTimeMixer 리비전이 forward(freq_token=...)을 필수로 요구할 때,
    호출될 때 자동으로 freq_token을 (B,) LongTensor로 주입해주는 래퍼.
    """
    def __init__(self, base: nn.Module, default_token: int = 0):
        super().__init__()
        self.base = base
        self.default_token = default_token

    def forward(self, x, **kwargs):
        B = x.size(0)
        if "freq_token" in kwargs:
            ft = kwargs["freq_token"]
            if torch.is_tensor(ft):
                if ft.ndim == 0:
                    ft = ft.view(1).expand(B)
                elif ft.ndim == 1 and ft.size(0) != B:
                    ft = ft.expand(B)
                ft = ft.to(dtype=torch.long, device=x.device)
            else:
                ft = torch.full((B,), int(ft), dtype=torch.long, device=x.device)
            kwargs["freq_token"] = ft
        else:
            kwargs["freq_token"] = torch.full(
                (B,), self.default_token, dtype=torch.long, device=x.device
            )
        return self.base(x, **kwargs)

def unwrap_model(m: nn.Module) -> nn.Module:
    return getattr(m, "base", m)


# ---------- 안전 로더 ----------
def safe_load_ttm_model(model_id: str, context_len: int, pred_len: int):
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
    parser = argparse.ArgumentParser(
        description="TTM-R2 demo with CSV or mock data (freq_token CLI, NaN mean fill)"
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="(optional) finetuned .pt checkpoint path")

    # 모의데이터 (CSV 미사용 시)
    parser.add_argument("--B", type=int, default=32, help="batch size (mock mode)")
    parser.add_argument("--T", type=int, default=512, help="context length (mock mode)")
    parser.add_argument("--H", type=int, default=96, help="prediction horizon")
    parser.add_argument("--C", type=int, default=3, help="num of channels (mock mode)")
    parser.add_argument("--n_examples", type=int, default=3, help="how many samples to save as PNG")

    # CSV 입력
    parser.add_argument("--csv1", type=str, default=None, help="변수1 CSV (행=아이템, 열=시간)")
    parser.add_argument("--csv2", type=str, default=None, help="변수2 CSV (행=아이템, 열=시간)")
    parser.add_argument("--csv3", type=str, default=None, help="변수3 CSV (행=아이템, 열=시간)")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    parser.add_argument("--skiprows", type=int, default=0, help="CSV header rows to skip")

    # freq_token CLI 옵션
    parser.add_argument("--freq_token", type=int, default=0,
                        help="freq_token 값 (기본 0, 예: --freq_token 7)")

    args = parser.parse_args()
    use_csv = args.csv1 and args.csv2 and args.csv3

    # -----------------------------
    # 1) 데이터 적재
    # -----------------------------
    if use_csv:
        X_np = load_triplet_csvs(
            args.csv1, args.csv2, args.csv3,
            delimiter=args.delimiter,
            skiprows=args.skiprows,
        )
        B, T, C = X_np.shape
        H = args.H
        if H > T:
            print(f"[warn] H({H}) > T({T}) → H=T로 자동 축소")
            H = T
        y_true_np = gen_future_labels_from_series(X_np, H)
        print(f"[data] CSV mode: X={X_np.shape}, y_true={y_true_np.shape}")
    else:
        B, T, H, C = args.B, args.T, args.H, args.C
        X_np, _ = gen_mock_batch(B, T, C=C, missing_prob=0.05, seed=42)
        y_true_np = gen_future_labels_from_series(X_np, H)
        print(f"[data] MOCK mode: X={X_np.shape}, y_true={y_true_np.shape}")

    X = torch.from_numpy(X_np).to(torch.float32).to(DEVICE, non_blocking=True)

    # -----------------------------
    # 2) 모델 로드 (+옵션 체크포인트)
    # -----------------------------
    model, cfg = safe_load_ttm_model(MODEL_ID, context_len=T, pred_len=H)
    model = TTMFreqAdapter(model, default_token=args.freq_token)

    if args.ckpt is not None and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=DEVICE)
        unwrap_model(model).load_state_dict(state, strict=False)
        model.to(DEVICE)
        print(f"[ckpt] loaded finetuned weights from {args.ckpt}")

    model.eval()

    # -----------------------------
    # 3) 예측 실행
    # -----------------------------
    with torch.no_grad():
        out, yhat = eval_step(model, X)
    if yhat is None:
        raise RuntimeError("Model did not return predictions.")

    if yhat.shape[1] > H:
        yhat = yhat[:, :H, :]
    print(f"[eval] prediction shape (sliced to H): {tuple(yhat.shape)}")

    # -----------------------------
    # 4) 시각화 저장
    # -----------------------------
    os.makedirs("plots", exist_ok=True)
    yhat_np = yhat.detach().cpu().numpy()
    plot_batch_examples(
        X_np, y_true_np, yhat_np,
        n_examples=args.n_examples,
        outdir="plots",
        tag=("csv" if use_csv else "direct")
    )

    # -----------------------------
    # 5) 임베딩 추출 → KMeans → UMAP 2D 저장
    # -----------------------------
    emb = extract_embeddings_ttm(out, X)
    labels, _ = cluster_embeddings(emb, k=4, seed=777)
    umap_scatter_2d(
        emb.detach().cpu().numpy(),
        labels,
        outpath="plots/umap_2d.png",
        title="UMAP 2D — TTM embeddings",
    )


if __name__ == "__main__":
    main()
