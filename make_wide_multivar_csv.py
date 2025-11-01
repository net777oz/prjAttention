#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3변량 모의 시계열을 '행=아이템, 열=타임스텝' 형태(헤더/인덱스 없음)의 CSV 3개로 저장합니다.
- 출력: data/varA.csv, data/varB.csv, data/varC.csv
- 각 파일 shape: [rows, cols] (기본 4000 x 200)
- 같은 (row, col)은 3개 파일에서 동일 시간축에 대응

사용 예)
  python make_wide_multivar_csv.py --rows 4000 --cols 200 --seed 42 --outdir data
  python make_wide_multivar_csv.py --rows 5000 --cols 256 --anomaly_frac 0.05 --spikes 3
"""

import argparse
import os
import numpy as np


def gen_base_series(t: np.ndarray, freq: float, phase: float, scale: float) -> np.ndarray:
    # 기본 파형: 서로 다른 주파수/위상/스케일 + 서서히 변하는 저주파 성분
    s = (
        np.sin(freq * t + phase)
        + 0.5 * np.cos(0.5 * freq * t + 2 * phase)
        + 0.1 * np.sin(0.05 * t)
    )
    return (scale * s).astype(np.float32)


def add_noise(x: np.ndarray, rng: np.random.Generator, sigma: float = 0.08) -> np.ndarray:
    return (x + sigma * rng.standard_normal(size=x.shape)).astype(np.float32)


def inject_spikes_inplace(x: np.ndarray, rng: np.random.Generator, n_spikes: int, scale: float):
    """x: (cols,), 임의 위치에 스파이크/딥 주입 (제자리 수정)"""
    if n_spikes <= 0:
        return
    cols = x.shape[0]
    idx = rng.choice(cols, size=min(n_spikes, cols), replace=False)
    signs = rng.choice([-1.0, 1.0], size=idx.size)
    std = float(x.std() + 1e-6)
    x[idx] += (scale * std) * signs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000, help="아이템 수(행)")
    ap.add_argument("--cols", type=int, default=200, help="타임스텝 수(열)")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드")
    ap.add_argument("--outdir", type=str, default="data", help="출력 폴더")
    # 잡음/이상치 옵션
    ap.add_argument("--noise_sigma", type=float, default=0.08, help="가우시안 잡음 표준편차")
    ap.add_argument("--anomaly_frac", type=float, default=0.0, help="이상치 주입할 행의 비율 (0~1)")
    ap.add_argument("--spikes", type=int, default=0, help="이상치 스파이크 개수(행당)")
    ap.add_argument("--spike_scale", type=float, default=3.0, help="스파이크 세기(표준편차 배수)")
    # 저장 포맷
    ap.add_argument("--fmt", type=str, default="%.6f", help="CSV 실수 포맷 (numpy.savetxt)")
    args = ap.parse_args()

    R, T = args.rows, args.cols
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 공통 시간축(열) — 모든 변수에 동일
    # (절대시간이 필요 없다면 t만 쓰고, 실제 timestamp는 별도로 관리)
    t = np.linspace(0, 8 * np.pi, T, dtype=np.float32)

    # 출력 배열 준비: (rows, cols)
    varA = np.empty((R, T), dtype=np.float32)
    varB = np.empty((R, T), dtype=np.float32)
    varC = np.empty((R, T), dtype=np.float32)

    # 어떤 행들에 이상치를 넣을지 선택
    n_anom_rows = int(R * max(0.0, min(1.0, args.anomaly_frac)))
    anom_rows = set(rng.choice(R, size=n_anom_rows, replace=False).tolist())

    # 행(아이템) 단위로 생성
    for i in range(R):
        # 아이템별 랜덤 파라미터(세 변수는 서로 상관을 갖도록 구성)
        base_freq = rng.uniform(0.25, 0.95)
        phase_a   = rng.uniform(0, np.pi)
        scale_a   = rng.uniform(0.8, 1.2)

        # A: 기준 시그널
        a = gen_base_series(t, base_freq, phase_a, scale_a)
        a = add_noise(a, rng, sigma=args.noise_sigma)

        # B: A의 영향을 받되, 약간 다른 성분 섞기
        phase_b = phase_a + rng.uniform(-0.4, 0.4)
        scale_b = scale_a * rng.uniform(0.85, 1.15)
        b_core  = gen_base_series(t, base_freq * rng.uniform(0.9, 1.1), phase_b, scale_b)
        b = 0.60 * a + 0.40 * b_core
        b = add_noise(b, rng, sigma=args.noise_sigma)

        # C: A, B와 상관 + 독자 성분
        phase_c = phase_a + rng.uniform(-0.6, 0.6)
        scale_c = scale_a * rng.uniform(0.8, 1.2)
        c_core  = gen_base_series(t, base_freq * rng.uniform(0.8, 1.2), phase_c, scale_c)
        c = 0.30 * a - 0.20 * b + 0.90 * c_core
        c = add_noise(c, rng, sigma=args.noise_sigma)

        # 이상치 행이면 스파이크 주입
        if i in anom_rows and args.spikes > 0:
            inject_spikes_inplace(a, rng, args.spikes, args.spike_scale)
            inject_spikes_inplace(b, rng, args.spikes, args.spike_scale)
            inject_spikes_inplace(c, rng, args.spikes, args.spike_scale)

        varA[i] = a
        varB[i] = b
        varC[i] = c

    # 헤더/인덱스 없이 저장 (행=아이템, 열=타임스텝)
    pathA = os.path.join(args.outdir, "varA.csv")
    pathB = os.path.join(args.outdir, "varB.csv")
    pathC = os.path.join(args.outdir, "varC.csv")

    np.savetxt(pathA, varA, fmt=args.fmt, delimiter=",")
    np.savetxt(pathB, varB, fmt=args.fmt, delimiter=",")
    np.savetxt(pathC, varC, fmt=args.fmt, delimiter=",")
    print(f"[ok] saved: {pathA}  shape={varA.shape}")
    print(f"[ok] saved: {pathB}  shape={varB.shape}")
    print(f"[ok] saved: {pathC}  shape={varC.shape}")

    # 메모: 학습/평가 시에는 원하는 행(row_idx)을 선택해서
    #  T 길이 벡터로 읽은 뒤, [T,3]로 스택해 사용:
    #   row = row_idx
    #   x = np.stack([varA[row], varB[row], varC[row]], axis=-1)  # [T, 3]
    #   # 또는 CSV에서 바로:
    #   # a = np.loadtxt(pathA, delimiter=",", dtype=np.float32)[row]
    #   # b = np.loadtxt(pathB, delimiter=",", dtype=np.float32)[row]
    #   # c = np.loadtxt(pathC, delimiter=",", dtype=np.float32)[row]
    #   # x = np.stack([a,b,c], axis=-1)

if __name__ == "__main__":
    main()
