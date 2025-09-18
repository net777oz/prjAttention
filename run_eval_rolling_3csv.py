# -*- coding: utf-8 -*-
"""
run_eval_rolling_3csv.py

3개 CSV(각각 N×T)를 받아 채널=3의 다변량 시계열(N, T, 3) 구성 후,
t=1..T-1에 대해 "1..t열(실데이터) → (t+1)열 1스텝 예측"을 반복 수행하고 평가합니다.

- 모델 입력은 (B, L, C) 형태 (B=N, C=3). L은 context_length로 패딩/절단.
- 매 스텝 t에서 마지막 L개(또는 t개 전체)가 컨텍스트가 되고, 예측은 1스텝만 사용합니다.
- 출력:
  - preds_stepwise.npy : (T-1, N, C_out)  t=1..T-1에 대한 (t+1)열 예측
  - metrics_stepwise.csv : 스텝별/채널별 MAE/MSE/MAPE 및 전체 집계
  - summary.json : 전체 평균 지표 등 메타 정보
  - (옵션) preds_stepwise.csv : 2D 평탄화 저장

Usage:
python run_eval_rolling_3csv.py \
  --csv1 data/varA.csv \
  --csv2 data/varB.csv \
  --csv3 data/varC.csv \
  --ckpt checkpoints/ttm_r2_ft1step_best.pt \
  --out artifacts/rolling_eval \
  --device cuda:0 \
  --save-csv

주의:
- 세 CSV의 (행=N, 열=T)가 다르면 최소 크기에 맞춰 자동 절단합니다.
- 모델이 여러 스텝(prediction_length>1)을 내더라도 첫 1스텝만 평가에 사용합니다.
- 모델의 context_length보다 t가 작으면 좌측 zero-padding으로 맞춥니다.
"""

import os, sys, argparse, json, importlib, inspect
import numpy as np
import pandas as pd
import torch


# ----------------------------
# 1) get_model 동적 탐색 + 폴백
# ----------------------------
def _resolve_get_model():
    """
    get_model 함수 자동 탐색:
    1) 같은 폴더의 get_model 모듈
    2) run_demo 모듈 내부의 get_model
    3) utils.get_model (관례적 경로)
    4) 폴백: HuggingFace 최소 로더 (모델 클래스 차이로 실패할 수 있음)
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    if curdir not in sys.path:
        sys.path.append(curdir)

    # 1) get_model.py (같은 폴더)
    try:
        mod = importlib.import_module("get_model")
        if hasattr(mod, "get_model"):
            print("[INFO] Using get_model from: get_model.py")
            return mod.get_model
    except Exception:
        pass

    # 2) run_demo.py 내부 함수
    try:
        mod = importlib.import_module("run_demo")
        if hasattr(mod, "get_model"):
            print("[INFO] Using get_model from: run_demo.py")
            return mod.get_model
    except Exception:
        pass

    # 3) utils/get_model.py
    try:
        mod = importlib.import_module("utils.get_model")
        if hasattr(mod, "get_model"):
            print("[INFO] Using get_model from: utils/get_model.py")
            return mod.get_model
    except Exception:
        pass

    # 4) 폴백: 간이 HF 로더
    print("[WARN] Could not find get_model; falling back to a minimal HF loader.")
    def _fallback_get_model(model_id: str = "ibm-granite/granite-timeseries-ttm-r2",
                            device: str = "cpu"):
        try:
            # AutoModel로는 일부 시계열 모델이 맞지 않을 수 있음
            from transformers import AutoModel
            m = AutoModel.from_pretrained(model_id)
            m = m.to(device)
            m.eval()
            return m
        except Exception as e:
            raise RuntimeError(
                "Fallback loader failed. Please expose your project's get_model() "
                "or install the correct timeseries model class."
            ) from e
    return _fallback_get_model

_get_model = _resolve_get_model()


# ----------------------------
# 2) 유틸 함수들
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv1", required=True, help="변수1 CSV (N×T)")
    p.add_argument("--csv2", required=True, help="변수2 CSV (N×T)")
    p.add_argument("--csv3", required=True, help="변수3 CSV (N×T)")
    p.add_argument("--ckpt", default="", help="파인튜닝 가중치(.pt)")
    p.add_argument("--model-id", default="ibm-granite/granite-timeseries-ttm-r2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="artifacts/rolling_eval")
    p.add_argument("--dtype", default="float32", choices=["float32","float64"])
    p.add_argument("--context-len", type=int, default=0,
                   help="0=모델 기본값 사용. >0이면 해당 길이로 강제(좌측 zero-padding/우측 절단).")
    p.add_argument("--save-csv", action="store_true", help="예측도 CSV로 저장")
    return p.parse_args()

def load_csv(path):
    df = pd.read_csv(path, header=None)
    return df.values

def to_device(x: np.ndarray, device: torch.device, dtype: str) -> torch.Tensor:
    if dtype == "float64":
        t = torch.from_numpy(x.astype(np.float64))
        return t.to(device=device, dtype=torch.float64)
    else:
        t = torch.from_numpy(x.astype(np.float32))
        return t.to(device=device, dtype=torch.float32)

def get_context_len_from_model(model, fallback=512):
    """
    granite 계열은 보통 config/context_length가 존재.
    가능성 높은 키들을 순회하면서 방어적으로 획득.
    """
    try:
        # 직접 속성
        for key in ["context_length", "context_len", "ctx_len", "prediction_context_length"]:
            if hasattr(model, key):
                return int(getattr(model, key))
        # config 내부
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for key in ["context_length", "context_len", "ctx_len", "prediction_context_length"]:
                if hasattr(cfg, key):
                    return int(getattr(cfg, key))
    except Exception:
        pass
    return fallback

def _to_tensor_first_step(output) -> torch.Tensor:
    """
    다양한 모델 출력 형태 방어:
    - Tensor (N, P, C) → 첫 스텝 out[:,0,:]
    - Tensor (N, C)     → 그대로
    - Tuple/Dict        → 안에 Tensor 꺼내 시도
    """
    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            return output[:, 0, :]
        elif output.ndim == 2:
            return output
        else:
            raise RuntimeError(f"Unexpected tensor shape: {tuple(output.shape)}")

    # tuple: (pred, ...) 형태 방어
    if isinstance(output, (tuple, list)) and len(output) > 0:
        for item in output:
            if isinstance(item, torch.Tensor):
                return _to_tensor_first_step(item)
        raise RuntimeError("Tuple/list output has no tensor.")

    # dict: {"predictions": Tensor, ...}류
    if isinstance(output, dict):
        # 우선순위 키 후보
        for key in ["predictions", "prediction", "logits", "output", "y_hat", "yhat"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return _to_tensor_first_step(output[key])
        # 첫 텐서 값 탐색
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return _to_tensor_first_step(v)
        raise RuntimeError("Dict output has no tensor.")

    raise RuntimeError(f"Unsupported model output type: {type(output)}")

def predict_one_step(model, x_ctx: torch.Tensor) -> torch.Tensor:
    """
    x_ctx: (N, L, C)
    return: (N, C_out)  → 모델 출력의 첫 타임스텝만 사용
    """
    with torch.no_grad():
        out = model(x_ctx)
        yhat = _to_tensor_first_step(out)
    return yhat

def mae(a, b): return np.mean(np.abs(a - b))
def mse(a, b): return np.mean((a - b) ** 2)
def mape(a, b, eps=1e-12):
    denom = np.clip(np.abs(b), eps, None)
    return np.mean(np.abs((a - b) / denom)) * 100.0


# ----------------------------
# 3) 메인
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) 데이터 로드
    X1 = load_csv(args.csv1)
    X2 = load_csv(args.csv2)
    X3 = load_csv(args.csv3)

    # 2) (행, 열) 최소로 정합
    n_rows = min(X1.shape[0], X2.shape[0], X3.shape[0])
    T      = min(X1.shape[1], X2.shape[1], X3.shape[1])
    if (X1.shape != (n_rows, T)) or (X2.shape != (n_rows, T)) or (X3.shape != (n_rows, T)):
        print(f"[WARN] Shape mismatch: X1={X1.shape}, X2={X2.shape}, X3={X3.shape} → truncate to ({n_rows}, {T})")
        X1, X2, X3 = X1[:n_rows, :T], X2[:n_rows, :T], X3[:n_rows, :T]

    if T < 2:
        raise ValueError(f"T(열 수)={T} < 2 이면 예측 대상이 없습니다. 최소 2열 이상 필요합니다.")

    # 3) 스택: (N, T, C=3)
    X = np.stack([X1, X2, X3], axis=-1)  # float 캐스팅은 to_device에서 처리

    # 4) 장치/정밀도
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    # 5) 모델 로드
    model = _get_model(model_id=args.model_id, device=str(device))

    # 5-1) 체크포인트 적용(선택, 실패해도 경고만)
    if args.ckpt and os.path.exists(args.ckpt):
        try:
            sd = torch.load(args.ckpt, map_location=device)
            # state_dict 직접이거나 {'state_dict': ...} 형태 대응
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)
            print(f"[ckpt] Loaded weights from {args.ckpt}")
        except Exception as e:
            print(f"[WARN] Failed to load ckpt {args.ckpt}: {e}")

    # 6) context_length 결정
    model_ctx_len = get_context_len_from_model(model, fallback=512)
    L = args.context_len if args.context_len > 0 else model_ctx_len
    print(f"[INFO] Using context_length = {L}")

    N, T, C = X.shape
    preds_all = np.zeros((T-1, N, C), dtype=np.float64)  # C_out==C 가정(대부분 동일)
    rows = []

    # 7) rolling one-step 평가 루프
    for t in range(1, T):  # 예측 목표는 시점 t (0-based), 컨텍스트는 [0..t-1]
        # 컨텍스트: 마지막 L 스텝 사용
        start = max(0, t - L)
        x_ctx_np = X[:, start:t, :]           # (N, l, C)   where l = t-start <= L
        l = x_ctx_np.shape[1]

        # 좌측 zero-padding (l<L) → (N, L, C)
        if l < L:
            pad = np.zeros((N, L - l, C), dtype=x_ctx_np.dtype)
            x_ctx_np = np.concatenate([pad, x_ctx_np], axis=1)

        # 텐서로 변환
        x_ctx = to_device(x_ctx_np, device, args.dtype)  # (N, L, C)

        # 1스텝 예측
        try:
            yhat = predict_one_step(model, x_ctx)        # (N, C_out)
        except Exception as e:
            raise RuntimeError(f"Model forward failed at step t={t}. "
                               f"Context shape={tuple(x_ctx.shape)}. Error: {e}") from e

        yhat_np = yhat.detach().cpu().numpy().astype(np.float64)

        # 정답: 실제 (t)열
        y_true_np = X[:, t, :].astype(np.float64)       # (N, C)

        # 저장
        # C_out != C 인 경우, 공통 차원만 평가
        C_eval = min(yhat_np.shape[1], y_true_np.shape[1])
        preds_all[t-1, :, :C_eval] = yhat_np[:, :C_eval]

        # 스텝별 지표
        step_metrics = {"t": t, "target_col": t+1}  # 1-based 컬럼 표기를 위해 t+1
        # 채널별
        for cidx in range(C_eval):
            step_metrics[f"MAE_c{cidx}"]  = mae(yhat_np[:, cidx], y_true_np[:, cidx])
            step_metrics[f"MSE_c{cidx}"]  = mse(yhat_np[:, cidx], y_true_np[:, cidx])
            step_metrics[f"MAPE_c{cidx}"] = mape(yhat_np[:, cidx], y_true_np[:, cidx])
        # 전체 집계(평가 대상 채널 평균)
        step_metrics["MAE_all"]  = mae(yhat_np[:, :C_eval], y_true_np[:, :C_eval])
        step_metrics["MSE_all"]  = mse(yhat_np[:, :C_eval], y_true_np[:, :C_eval])
        step_metrics["MAPE_all"] = mape(yhat_np[:, :C_eval], y_true_np[:, :C_eval])
        rows.append(step_metrics)

        if t % 10 == 0 or t == T-1:
            print(f"[eval] step {t}/{T-1} done")

    # 8) 저장
    np.save(os.path.join(args.out, "preds_stepwise.npy"), preds_all)
    if args.save_csv:
        # (T-1, N, C) → 2D 평탄화: 행=스텝×샘플, 열=채널
        flat = preds_all.reshape((T-1)*N, C)
        cols = [f"c{c}" for c in range(C)]
        dfp = pd.DataFrame(flat, columns=cols)
        dfp.to_csv(os.path.join(args.out, "preds_stepwise.csv"), index=False)

    dfm = pd.DataFrame(rows)
    dfm.to_csv(os.path.join(args.out, "metrics_stepwise.csv"), index=False)

    # 요약
    summary = {
        "input_shape": [int(N), int(T), int(C)],
        "context_length_used": int(L),
        "csvs": [args.csv1, args.csv2, args.csv3],
        "model_id": args.model_id,
        "ckpt": args.ckpt,
        "device": str(device),
        "dtype": args.dtype,
        "overall": {
            "MAE_all_mean": float(dfm["MAE_all"].mean()),
            "MSE_all_mean": float(dfm["MSE_all"].mean()),
            "MAPE_all_mean": float(dfm["MAPE_all"].mean()),
        }
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {args.out}/preds_stepwise.npy ({preds_all.shape})")
    print(f"[OK] Saved: {args.out}/metrics_stepwise.csv  (T-1 rows)")
    print(f"[OK] Saved: {args.out}/summary.json")
    if args.save_csv:
        print(f"[OK] Saved: {args.out}/preds_stepwise.csv")


if __name__ == "__main__":
    main()
