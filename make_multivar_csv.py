#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동일한 타임스탬프에 정렬된 3변량 모의 시계열을
- varA.csv (time,value)
- varB.csv (time,value)
- varC.csv (time,value)
- multivar.csv (time,varA,varB,varC)
로 생성합니다.

예)
  python make_multivar_csv.py --T 2048 --freq H --start "2025-01-01 00:00" --seed 123
"""

import argparse
import os
import numpy as np
import pandas as pd


def make_time_index(T: int, start: str, freq: str) -> pd.DatetimeIndex:
    # 예: freq="H"(시간), "D"(일), "min"(분)
    return pd.date_range(start=start, periods=T, freq=freq)


def gen_channel(t: np.ndarray, base_freq: float, phase: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    # 서로 다른 주파수/위상/스케일 + 약간의 잡음 + 저주파 추세
    s = (
        np.sin(base_freq * t + phase)
        + 0.5 * np.cos(0.5 * base_freq * t + 2 * phase)
        + 0.1 * np.sin(0.05 * t)  # 느린 추세 성분
    )
    s = scale * s
    noise = 0.08 * rng.standard_normal(size=t.shape).astype(np.float32)
    return (s + noise).astype(np.float32)


def inject_anomalies(x: np.ndarray, rng: np.random.Generator, n_spikes: int = 4, spike_scale: float = 3.0) -> None:
    # 임의 위치에 스파이크/딥 추가 (in-place)
    if n_spikes <= 0: return
    T = x.shape[0]
    idx = rng.choice(T, size=min(n_spikes, T), replace=False)
    signs = rng.choice([-1.0, 1.0], size=idx.shape[0])
    x[idx] += signs * spike_scale * x.std()


def maybe_nan(x: np.ndarray, rng: np.random.Generator, missing_prob: float) -> None:
    if missing_prob <= 0.0: return
    mask = rng.random(x.shape) < missing_prob
    x[mask] = np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024, help="시계열 길이")
    ap.add_argument("--start", type=str, default="2025-01-01 00:00", help="시작 시각 (YYYY-MM-DD HH:MM)")
    ap.add_argument("--freq", type=str, default="H", help="빈도 (예: H=시간, D=일, min=분)")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드")
    ap.add_argument("--missing_prob", type=float, default=0.0, help="결측 비율 (0~1)")
    ap.add_argument("--with_anomaly", action="store_true", help="이상치(스파이크/딥) 주입")
    ap.add_argument("--outdir", type=str, default="data", help="출력 폴더")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 공통 시간축
    idx = make_time_index(args.T, args.start, args.freq)
    t = np.linspace(0, 8 * np.pi, args.T, dtype=np.float32)

    # 각 변수별로 약간씩 다른 파라미터
    pars = [
        dict(name="varA", base_freq=rng.uniform(0.3, 0.9), phase=rng.uniform(0, np.pi), scale=rng.uniform(0.8, 1.2)),
        dict(name="varB", base_freq=rng.uniform(0.2, 1.0), phase=rng.uniform(0, np.pi), scale=rng.uniform(0.8, 1.2)),
        dict(name="varC", base_freq=rng.uniform(0.25, 0.95), phase=rng.uniform(0, np.pi), scale=rng.uniform(0.8, 1.2)),
    ]

    series = {}
    for p in pars:
        x = gen_channel(t, p["base_freq"], p["phase"], p["scale"], rng)
        if args.with_anomaly:
            inject_anomalies(x, rng, n_spikes=4, spike_scale=3.0)
        maybe_nan(x, rng, args.missing_prob)
        # 모델/후처리 호환을 위해 NaN/Inf 정리
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        series[p["name"]] = x

        # 개별 CSV: (time,value)
        df_single = pd.DataFrame({"time": idx, "value": x})
        df_single.to_csv(os.path.join(args.outdir, f"{p['name']}.csv"), index=False)

    print(f"[ok] saved {len(pars)} single-var CSVs -> {args.outdir}/varA.csv, varB.csv, varC.csv")

    # 멀티버리엇 CSV도 함께 저장: (time,varA,varB,varC)
    df_multi = pd.DataFrame({"time": idx})
    for k, v in series.items():
        df_multi[k] = v
    df_multi.to_csv(os.path.join(args.outdir, "multivar.csv"), index=False)
    print(f"[ok] saved multivariate CSV -> {args.outdir}/multivar.csv")


if __name__ == "__main__":
    main()
