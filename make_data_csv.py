#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3변량 모의 시계열을 CSV로 생성해 저장합니다.
- 컬럼: time, ch0, ch1, ch2
- 기본 출력: data/train.csv, data/val.csv, data/test.csv
- 여러 샘플을 만들면 train_0.csv, train_1.csv ... 개별 파일도 함께 저장

사용 예:
  python make_data_csv.py --T 1024 --C 3 --seed 123 --train 2 --val 1 --test 1
"""

import argparse
import os
import numpy as np
import pandas as pd


def gen_mock_batch(
    B: int,
    T: int,
    C: int = 3,
    missing_prob: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    모의 다변량 시계열 생성.
    반환: X [B, T, C]
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, T, dtype=np.float32)  # 파형 길이

    X = np.zeros((B, T, C), dtype=np.float32)
    for b in range(B):
        phases = rng.uniform(0, np.pi, size=C)
        scales = rng.uniform(0.8, 1.2, size=C)
        freqs  = rng.uniform(0.2, 1.0, size=C)

        series = []
        for c in range(C):
            s = (
                np.sin(freqs[c] * t + phases[c])
                + 0.5 * np.cos(0.5 * freqs[c] * t + 2 * phases[c])
            )
            s = scales[c] * s
            series.append(s)
        base = np.stack(series, axis=-1)
        noise = 0.08 * rng.standard_normal(size=base.shape).astype(np.float32)
        X[b] = base + noise

        if missing_prob > 0.0:
            mask = rng.random((T, C)) < missing_prob
            X[b][mask] = np.nan  # 필요시 아래에서 채움

    # KMeans/UMAP 등과 함께 쓰려면 NaN이 없도록 처리
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return X


def save_series_to_csv(x: np.ndarray, out_path: str):
    """
    단일 시계열 x: [T, C] -> CSV 저장 (time, ch0..ch{C-1})
    """
    T, C = x.shape
    df = pd.DataFrame(x, columns=[f"ch{i}" for i in range(C)])
    df.insert(0, "time", np.arange(T, dtype=np.int64))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[data] saved -> {out_path}  (rows={T}, cols={C+1})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024, help="시계열 길이")
    ap.add_argument("--C", type=int, default=3, help="채널 수(기본 3)")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드")
    ap.add_argument("--missing_prob", type=float, default=0.0, help="결측 삽입 확률(기본 0.0)")
    ap.add_argument("--train", type=int, default=1, help="train 샘플 개수(B)")
    ap.add_argument("--val", type=int, default=1, help="val 샘플 개수(B)")
    ap.add_argument("--test", type=int, default=1, help="test 샘플 개수(B)")
    ap.add_argument("--outdir", type=str, default="data", help="CSV 저장 폴더")
    args = ap.parse_args()

    # Train
    if args.train > 0:
        X_tr = gen_mock_batch(args.train, args.T, C=args.C, missing_prob=args.missing_prob, seed=args.seed)
        for i in range(args.train):
            save_series_to_csv(X_tr[i], os.path.join(args.outdir, f"train_{i}.csv"))
        # 대표 본으로 train.csv 도 저장(첫 번째 샘플)
        save_series_to_csv(X_tr[0], os.path.join(args.outdir, "train.csv"))

    # Val
    if args.val > 0:
        X_va = gen_mock_batch(args.val, args.T, C=args.C, missing_prob=args.missing_prob, seed=args.seed + 1)
        for i in range(args.val):
            save_series_to_csv(X_va[i], os.path.join(args.outdir, f"val_{i}.csv"))
        save_series_to_csv(X_va[0], os.path.join(args.outdir, "val.csv"))

    # Test
    if args.test > 0:
        X_te = gen_mock_batch(args.test, args.T, C=args.C, missing_prob=args.missing_prob, seed=args.seed + 2)
        for i in range(args.test):
            save_series_to_csv(X_te[i], os.path.join(args.outdir, f"test_{i}.csv"))
        save_series_to_csv(X_te[0], os.path.join(args.outdir, "test.csv"))


if __name__ == "__main__":
    main()
