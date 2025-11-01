import torch
import numpy as np

src_csv = "data/LP0a.csv"

# BOM 제거: utf-8-sig로 읽기
with open(src_csv, "r", encoding="utf-8-sig") as f:
    arr = np.loadtxt(f, delimiter=",", dtype=np.float32)

x = torch.from_numpy(arr)              # shape: (행, 열)
zeros = torch.zeros_like(x)            # 동일 크기 0 텐서 (dtype/shape 동일)

print("원본 텐서 크기:", x.shape)
print("제로 텐서 크기:", zeros.shape)

# 필요 시 CSV로 저장
np.savetxt("data/zeros.csv", zeros.numpy(), delimiter=",", fmt="%.6f")
