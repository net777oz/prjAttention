# tools/metrics_plot.py
# train_classify_ctx1..31_* 폴더의 metrics.json을 모아
# raw(왼쪽 y축 0~1) vs (/prev)(오른쪽 y축 0부터) 를 한 플롯에 오버레이
# CSV도 함께 저장

import os, json
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.abspath("./artifacts")
METRICS_FILE = "metrics.json"
START, END = 1, 31  # ctx1~ctx31 (1자리엔 0패딩 없음)

DIR_TPL_MAIN = "finetune_classify_ctx{X}_ch1_alpha0.80_pwglobal_llm_ts_seed777_splitgroup"
DIR_TPL_FALLBACK = "finetune_classify_ctx{X:02d}_ch1_alpha0.80_pwglobal_llm_ts_seed777_splitgroup"

rows, missing = [], []

def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def _read_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("metrics_at_default", {}) or {}
    return {
        "precision": _to_float_or_nan(m.get("precision")),
        "recall":    _to_float_or_nan(m.get("recall")),
        "f1":        _to_float_or_nan(m.get("f1")),
        "accuracy":  _to_float_or_nan(m.get("accuracy")),
        "roc_auc":   _to_float_or_nan(data.get("roc_auc")),
        "pr_auc":    _to_float_or_nan(data.get("pr_auc")),
        "prevalence":_to_float_or_nan(data.get("prevalence")),
    }

def _append_row(run_idx: int):
    # 비패딩 → 0패딩 순서로 탐색
    d = DIR_TPL_MAIN.format(X=run_idx)
    p = os.path.join(BASE, d, METRICS_FILE)
    if not os.path.isfile(p):
        d2 = DIR_TPL_FALLBACK.format(X=run_idx)
        p2 = os.path.join(BASE, d2, METRICS_FILE)
        if os.path.isfile(p2):
            p = p2
        else:
            missing.append(run_idx)
            return
    try:
        r = _read_metrics(p)
        r["run"] = run_idx
        rows.append(r)
    except Exception as e:
        print(f"[WARN] ctx{run_idx} 읽기 실패: {e}")
        missing.append(run_idx)

# ---- 수집
for n in range(START, END + 1):
    _append_row(n)

df = pd.DataFrame(rows)
if df.empty:
    print("[ERROR] 수집된 메트릭이 없습니다.")
    print(f"- BASE: {BASE} (exists={os.path.isdir(BASE)})")
    raise SystemExit(1)

df = df.sort_values("run").set_index("run")

# ---- (/prev) 파생열
def _over_prev(col):
    prev = df["prevalence"]
    num = df[col]
    out = num.copy()
    mask = (prev > 0) & (~prev.isna()) & (~num.isna())
    out.loc[~mask] = float("nan")
    out.loc[mask] = num.loc[mask] / prev.loc[mask]
    return out

metrics = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
for m in metrics:
    df[f"{m}_over_prev"] = _over_prev(m)

# ---- CSV 저장 (원본 + /prev + prevalence)
out_csv = "metrics_summary.csv"
df.to_csv(out_csv, index=True)
print(f"[OK] 요약 저장: {out_csv}")

# ---- 플롯: raw vs /prev 오버레이(좌/우 y축)
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
titles = ["Precision", "Recall", "F1", "Accuracy", "ROC AUC", "PR AUC"]
cols_raw   = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
cols_prev  = [f"{m}_over_prev" for m in metrics]
x_ticks = [f"ctx{r}" for r in df.index]

for ax, col_raw, col_prev, title in zip(axes.flat, cols_raw, cols_prev, titles):
    # 왼쪽 축: raw (0..1 고정)
    line_raw, = ax.plot(x_ticks, df[col_raw], marker="o", label=f"{title} (raw)")
    ax.set_title(title + " — raw vs /prev")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)

    # 오른쪽 축: /prev (하한 0, 상한 자동)
    ax_r = ax.twinx()
    line_prev, = ax_r.plot(x_ticks, df[col_prev], linestyle="--", marker="x", label=f"{title}/prev")
    ax_r.set_ylim(bottom=0)
    ax_r.set_ylabel("/prev")

    # 범례(양축 핸들 합치기)
    handles = [line_raw, line_prev]
    labels  = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="best")

plt.tight_layout()
out_png = "metrics_overlay.png"
plt.savefig(out_png, dpi=150)
print(f"[OK] 그래프 저장: {out_png}")

# 누락 알림
if missing:
    miss_tags = ", ".join([f"ctx{m}" for m in sorted(set(missing))])
    print(f"[INFO] 누락된 러닝: {miss_tags}")
