# quick_check_ttm_r2_with_granite.py (수정본)
import torch
from tsfm_public.toolkit.get_model import get_model

MODEL_CARD = "ibm-granite/granite-timeseries-ttm-r2"
CONTEXT_LEN = 512
PRED_LEN = 96

model = get_model(
    model_path=MODEL_CARD,
    model_name="ttm",
    context_length=CONTEXT_LEN,
    prediction_length=PRED_LEN,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

B, T, C = 2, CONTEXT_LEN, 3
x = torch.randn(B, T, C, device=device)

with torch.no_grad():
    out = model(x, return_dict=True, output_hidden_states=True)

print("OK: forward 성공")

# ✅ 안전하게 키 확인
keys = [k for k in dir(out) if not k.startswith("_")]
print("out keys:", keys)

# ✅ 예측 텐서(있으면) 확인
for k in ("prediction", "predictions", "forecast"):
    if hasattr(out, k):
        yhat = getattr(out, k)
        print(f"{k} shape:", tuple(yhat.shape))
        break

# ✅ 임베딩은 hidden_states에서 마지막 계층을 사용
hs = getattr(out, "hidden_states", None)
if hs is None:
    raise RuntimeError("hidden_states가 없습니다. forward에 output_hidden_states=True를 확인하세요.")
last = hs[-1]  # [B, T, d]
print("hidden_states[-1] shape:", tuple(last.shape))
