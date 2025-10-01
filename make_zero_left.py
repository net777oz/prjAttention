import numpy as np

src_csv = "data/LDa.csv"
dst_csv = "data/LD0a.csv"

# BOM 제거 안전하게 utf-8-sig로 읽기
arr = np.loadtxt(src_csv, delimiter=",", dtype=np.float32, encoding="utf-8-sig")

n_rows, n_cols = arr.shape
pad_cols = 28

# (행, 31) 크기의 제로 배열 생성
pad = np.zeros((n_rows, pad_cols), dtype=np.float32)

# 왼쪽(시작점)에 제로 열 추가
padded = np.hstack([pad, arr])

print("원본 shape:", arr.shape)
print("패딩 후 shape:", padded.shape)

# 저장
np.savetxt(dst_csv, padded, delimiter=",", fmt="%.6f")
print(f"저장 완료: {dst_csv}")

