# metrics_plot.py
# 001~032까지 infer 결과의 metrics.json을 모아 4개 지표 그래프와 요약 CSV 생성

import os, json
import pandas as pd
import matplotlib.pyplot as plt

BASE = "../artifacts"
PREFIX = "infer_classify_SPa_"
START, END = 1, 32  # 001 ~ 032

rows = []
missing = []

for n in range(START, END + 1):
    tag = f"{n:03d}"
    path = os.path.join(BASE, f"{PREFIX}{tag}", "metrics.json")
    if not os.path.isfile(path):
        missing.append(tag)
        continue
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics_at_default", {})
        rows.append({
            "run": tag,
            "precision": float(m.get("precision", "nan")),
            "recall": float(m.get("recall", "nan")),
            "f1": float(m.get("f1", "nan")),
            "accuracy": float(m.get("accuracy", "nan")),
        })
    except Exception as e:
        print(f"[WARN] {tag} 읽기 실패: {e}")
        missing.append(tag)

# 데이터프레임 생성 및 정렬(001,002,...,032)
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("수집된 메트릭이 없습니다. 경로/파일을 확인하세요.")
df = df.sort_values("run", key=lambda s: s.map(int)).set_index("run")

# 요약 저장
out_csv = "metrics_summary.csv"
df.to_csv(out_csv, index=True)
print(f"[OK] 요약 저장: {out_csv}")

# 그래프 (2x2 서브플롯)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
metrics = ["precision", "recall", "f1", "accuracy"]
titles = ["Precision", "Recall", "F1 Score", "Accuracy"]

for ax, metric, title in zip(axes.flat, metrics, titles):
    ax.plot(df.index, df[metric], marker="o")
    ax.set_title(title)
    ax.set_ylim(0, 1)  # 확률 지표 범위
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
out_png = "metrics_trends.png"
plt.savefig(out_png, dpi=150)
print(f"[OK] 그래프 저장: {out_png}")

if missing:
    print(f"[INFO] 누락된 러닝(run): {', '.join(missing)}")
