# ttm_flow/pipeline.py
# - TinyTimeMixer(TTM-R2) 파이프라인 유틸
# - eval_step / cluster_embeddings
# - 1-step 전용 평가(eval_1step_mse)와 1-step 파인튜닝 루프(finetune_1step; AMP, ckpt, early-stop)

from typing import Optional
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader


def _pick_prediction_tensor(out):
    """
    모델 출력 객체에서 예측 텐서를 찾아 반환.
    (TTM 계열은 키가 버전에 따라 다를 수 있으므로 안전하게 탐색)
    """
    for key in ("predictions", "prediction", "forecast", "prediction_outputs"):
        if hasattr(out, key):
            val = getattr(out, key)
            if isinstance(val, dict):
                for k2 in ("predictions", "prediction", "forecast", "yhat"):
                    if k2 in val:
                        return val[k2]
            return val
    return None


@torch.no_grad()
def eval_step(model, X: torch.Tensor):
    """
    평가 모드 forward.
    - output_hidden_states=True 로 임베딩 추출 준비
    - 예측 텐서 탐색
    """
    model.eval()
    out = model(X, output_hidden_states=True, return_dict=True)
    yhat = _pick_prediction_tensor(out)
    return out, yhat


def cluster_embeddings(emb_t: torch.Tensor, k: int = 4, seed: int = 777):
    """
    torch 텐서를 numpy로 변환해 KMeans 수행.
    """
    emb = emb_t.detach().cpu().numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(emb)
    return labels, km


def eval_1step_mse(model, X: torch.Tensor, y1: torch.Tensor) -> float:
    """
    1-step 전용 MSE 평가. y1 shape: [N, 1, C]
    - 모델 디바이스로 X/y를 옮겨 device mismatch 방지
    - AMP로 추론 가속 (torch.amp)
    """
    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
        Xb = X.to(device, non_blocking=True)
        yb = y1.to(device, non_blocking=True)
        out = model(Xb, return_dict=True)
        yhat = _pick_prediction_tensor(out)
        if yhat is None:
            raise RuntimeError("Predictions not found.")
        yhat1 = yhat[:, 0:1, :]
        loss = F.mse_loss(yhat1, yb)
    return loss.item()


def finetune_1step(
    model,
    X_train: torch.Tensor,  # [N, T, C]  (CPU 텐서로 전달)
    y_train: torch.Tensor,  # [N, 1, C]  (CPU 텐서로 전달)
    X_val: Optional[torch.Tensor] = None,  # (CPU)
    y_val: Optional[torch.Tensor] = None,  # (CPU)
    *,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 2e-4,
    clip: float = 1.0,
    num_workers: int = 2,
    save_best_path: Optional[str] = None,  # 베스트(최저 val) 체크포인트 저장 경로
    patience: int = 0,                     # 얼리 스탑 patience(에폭). 0이면 비활성화
):
    """
    TTM-R2는 96-step을 내지만, loss는 첫 1-step에만 걸어 미세조정.
    - DataLoader엔 CPU 텐서를 넣고(pin_memory 사용), 루프 내부에서 GPU로 올립니다.
    - AMP + GradScaler 사용 (torch.amp).
    - 선택: save_best_path에 베스트(val MSE) 가중치 저장, patience>0이면 얼리 스탑.
    """
    device = next(model.parameters()).device
    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),  # CPU→GPU 전송 최적화
        num_workers=num_workers,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    best_val = float("inf")
    best_saved = False
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)   # 여기서 GPU 이동
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                out = model(xb, return_dict=True)
                yhat = _pick_prediction_tensor(out)
                if yhat is None:
                    raise RuntimeError("Predictions not found during training.")
                yhat1 = yhat[:, 0:1, :]
                # 이상치 안정화를 위해 Huber + MSE 혼합
                loss = 0.7 * F.huber_loss(yhat1, yb, delta=0.8) + 0.3 * F.mse_loss(yhat1, yb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()

            running += loss.detach().item() * xb.size(0)

        train_mse = running / len(ds)
        msg = f"[ft-1step] epoch {ep}/{epochs}  train_loss≈{train_mse:.6f}"

        # --- 검증 / 체크포인트 / 얼리스탑 ---
        if X_val is not None and y_val is not None:
            val_mse = eval_1step_mse(model, X_val, y_val)  # 내부에서 디바이스 정합
            improved = val_mse < best_val - 1e-12
            if improved:
                best_val = val_mse
                no_improve = 0
                if save_best_path is not None:
                    torch.save(model.state_dict(), save_best_path)
                    best_saved = True
            else:
                no_improve += 1

            msg += f"  |  val_mse={val_mse:.6f}"
            if patience > 0 and no_improve >= patience:
                print(msg)
                stop_note = f"[early-stop] patience reached ({patience}). Best val MSE: {best_val:.6f}"
                if save_best_path and not best_saved:
                    # 에지 케이스 보호: 개선이 한 번도 없었을 때 현재 가중치라도 저장
                    torch.save(model.state_dict(), save_best_path)
                    stop_note += " (current weights saved)"
                print(stop_note)
                return

        print(msg)
