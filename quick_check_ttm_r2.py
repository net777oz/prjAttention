# quick_check_ttm_r2.py
from transformers import AutoConfig, AutoModel
m = "ibm-granite/granite-timeseries-ttm-r2"
cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)  # ← 여기서 에러가 없어야 정상
print("config ok:", type(cfg), getattr(cfg, "model_type", None))
model = AutoModel.from_pretrained(m, trust_remote_code=True)
print("model ok:", type(model))
