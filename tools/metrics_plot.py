# metrics_plot.py
# 001~032 + (64,96,128,160,192) infer 결과의 metrics.json을 모아
# 6개 지표(precision/recall/f1/accuracy/roc_auc/pr_auc)를 prevalence로 나눈 값으로 플롯/저장

import os, json
import pandas as pd
import matplotlib.pyplot as plt
import math

BASE = "./artifacts"
PREFIX = "infer_classify_SPa_"
START, END = 1, 32  # 001 ~ 032
EXTRA = [64, 96, 128, 160, 192]

rows, missing = [], []

def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def _append_row(tag, path):
    global rows, missing
    if not os.path.isfile(path):
        missing.append(tag); return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics_at_default", {}) or {}
        rows.append({
            "run": tag,
            "precision": _to_float_or_nan(m.get("precision")),
            "recall":    _to_float_or_nan(m.get("recall")),
            "f1":        _to_float_or_nan(m.get("f1")),
            "accuracy":  _to_float_or_nan(m.get("accuracy")),
            "roc_auc":   _to_float_or_nan(data.get("roc_auc")),
            "pr_auc":    _to_float_or_nan(data.get("pr_auc")),
            "prevalence":_to_float_or_nan(data.get("prevalence")),  # ← 반드시 읽어옴
        })
    except Exception as e:
        print(f"[WARN] {tag} 읽기 실패: {e}")
        missing.append(tag)

# 001..032
for n in range(START, END + 1):
    tag = f"{n:03d}"
    _append_row(tag, os.path.join(BASE, f"{PREFIX}{tag}", "metrics.json"))

# 064..192 (추가)
for n in EXTRA:
    tag = f"{n:03d}"
    _append_row(tag, os.path.join(BASE, f"{PREFIX}{tag}", "metrics.json"))

# 데이터프레임
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("수집된 메트릭이 없습니다. 경로/파일을 확인하세요.")
df = df.sort_values("run", key=lambda s: s.map(int)).set_index("run")

# prevalence로 나누기 (0 또는 NaN이면 결과 NaN)
def _div_over_prev(col):
    prev = df["prevalence"]
    num = df[col]
    out = num.copy()
    mask = (prev > 0) & (~prev.isna()) & (~num.isna())
    out.loc[~mask] = float("nan")
    out.loc[mask] = num.loc[mask] / prev.loc[mask]
    return out

metrics = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
for m in metrics:
    df[f"{m}_over_prev"] = _div_over_prev(m)

# 요약 CSV 저장(원래값 + prevalence + 나눈값)
out_csv = "metrics_summary.csv"
df.to_csv(out_csv, index=True)
print(f"[OK] 요약 저장: {out_csv}")

# 그래프: 3x2, y축은 자동 스케일 (0..1 고정 제거)
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
titles = ["Precision/prev", "Recall/prev", "F1/prev", "Accuracy/prev", "ROC AUC/prev", "PR AUC/prev"]
cols   = [f"{m}_over_prev" for m in metrics]

for ax, col, title in zip(axes.flat, cols, titles):
    ax.plot(df.index, df[col], marker="o")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
out_png = "metrics_trends.png"
plt.savefig(out_png, dpi=150)
print(f"[OK] 그래프 저장: {out_png}")

if missing:
    print(f"[INFO] 누락된 러닝(run): {', '.join(sorted(set(missing), key=lambda x:int(x)))}")
