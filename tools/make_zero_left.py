import numpy as np
import re
from collections import Counter
from pathlib import Path

src_csv = "../data/LPa.csv"
dst_csv = "../data/LP0a.csv"

# 허용: 숫자/부호/소수점/지수표기(e/E)
NUM_RE = re.compile(r"^[ \t]*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?[ \t]*$")

def clean_line(line: str) -> str:
    # 개행 통일 + BOM 제거 + 제어문자(개행/탭 제외) 제거
    line = line.replace("\r\n", "\n").replace("\r", "\n")
    line = line.replace("\ufeff", "")
    line = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", line)
    return line

def parse_numeric_row(raw_line: str, delim: str = ","):
    parts = [p.strip() for p in raw_line.split(delim)]
    vals = []
    nan_count = 0
    for p in parts:
        if p == "" or not NUM_RE.match(p):
            vals.append(np.nan)
            nan_count += 1
        else:
            try:
                vals.append(float(p))
            except Exception:
                vals.append(np.nan)
                nan_count += 1
    return vals, nan_count

def mode_or_max(lengths):
    # 최빈수(동점이면 가장 큰 값) 선택
    if not lengths:
        return 0
    cnt = Counter(lengths)
    most = max(cnt.values())
    candidates = [k for k, v in cnt.items() if v == most]
    return max(candidates)

# 1) 원본 텍스트 읽기(utf-8-sig로만; 손상 바이트는 무시하지 않음)
raw = Path(src_csv).read_text(encoding="utf-8-sig")
raw = clean_line(raw)

# 2) 라인 단위 파싱(숫자 아닌 셀은 NaN)
rows_raw = [ln for ln in raw.split("\n") if ln.strip() != ""]
parsed = []
row_nan_counts = []
row_lengths = []

for ln in rows_raw:
    vals, n_nan = parse_numeric_row(ln, ",")
    if len(vals) == 0:
        continue
    parsed.append(vals)
    row_nan_counts.append(n_nan)
    row_lengths.append(len(vals))

if not parsed:
    raise RuntimeError("입력 CSV에서 유효한 행을 하나도 찾지 못했습니다.")

# 3) 열 수 통일: 최빈 열수에 맞춰 강제
expected_cols = mode_or_max(row_lengths)
coerced = []
coerced_rows = 0
for vals in parsed:
    if len(vals) == expected_cols:
        coerced.append(vals)
    elif len(vals) < expected_cols:
        coerced.append(vals + [np.nan] * (expected_cols - len(vals)))
        coerced_rows += 1
    else:  # 길면 자르기
        coerced.append(vals[:expected_cols])
        coerced_rows += 1

arr = np.array(coerced, dtype=np.float32)
nan_total = np.isnan(arr).sum()

# 4) NaN → 0 치환(필요하면 평균/중앙값으로 바꿔도 됨)
arr = np.nan_to_num(arr, nan=0.0)

# 5) 왼쪽으로 제로 패딩 28열 추가
pad_cols = 28
pad = np.zeros((arr.shape[0], pad_cols), dtype=np.float32)
padded = np.hstack([pad, arr])

print(f"[INFO] rows_in={len(rows_raw)}, used={arr.shape[0]}")
print(f"[INFO] expected_cols={expected_cols}, coerced_rows={coerced_rows}")
print(f"[INFO] NaN->0 replaced: {int(nan_total)} cells")
print("원본 shape:", (arr.shape[0], arr.shape[1]))
print("패딩 후 shape:", padded.shape)

# 6) 저장
np.savetxt(dst_csv, padded, delimiter=",", fmt="%.6f")
print(f"[OK] 저장 완료: {dst_csv}")
