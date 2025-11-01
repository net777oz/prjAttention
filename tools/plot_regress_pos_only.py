# -*- coding: utf-8 -*-
"""
plot_regress_pos_only.py — 회귀 예측 시각화(양성 필터 지원, 유닛 분리 지원)
- 입력: after_pred.csv (또는 동일 포맷)
  * 기본 가정: 앞의 두 개 숫자열이 y_true, y_pred (이름이 y_true, y_pred면 그대로 사용)
  * (선택) unit 열이 있으면 유닛별로 분할 가능
- 필터: --threshold 로 y_true > thr 만 선택 (thr<0면 전체)
- 저장: <pred_csv의 폴더>/plots/ 아래에 PNG 저장

기본값:
  • ts-mode=facet (실측/예측을 원 단위로 나란히)
  • scatter-mode=hex (빈도 표현)
  • 히스토그램 범위 auto(percentile 0.5~99.5)
  • by-unit=auto (unit 열 있으면 사용, 없으면 RUL 점프(증가)로 자동 분할)

추가:
  • --ts-raw: 시계열을 원 단위 그대로 그림(마스크/스케일링 없음)
  • size 스캐터: 큰 원일수록 투명(겹침 가독성↑)
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────── 유틸 ───────────────────────────

def _ensure_plots_dir(pred_path: Path) -> Path:
    out_dir = pred_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _read_pred_csv(path: Path):
    """
    CSV에서 (y_true, y_pred, unit or None)을 반환.
    - 'y_true','y_pred' 컬럼이 있으면 그대로 사용.
    - 없으면 앞에서부터 '숫자형'인 2개 컬럼을 자동 선택.
    - 'unit' 컬럼이 있으면 함께 반환(없으면 None).
    """
    df = pd.read_csv(path)
    cols = list(df.columns)
    unit = df["unit"].to_numpy() if "unit" in cols else None

    if ("y_true" in cols) and ("y_pred" in cols):
        yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy()
        yp = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy()
    else:
        # 숫자형 2개 자동 탐지
        num_cols = []
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= len(s) * 0.5:  # 절반 이상이 숫자면 숫자열로 간주
                num_cols.append(c)
        if len(num_cols) < 2:
            raise SystemExit("CSV에서 숫자형 2개 열을 찾지 못했습니다. y_true,y_pred 열명을 쓰거나 숫자열을 앞에 두세요.")
        yt = pd.to_numeric(df[num_cols[0]], errors="coerce").to_numpy()
        yp = pd.to_numeric(df[num_cols[1]], errors="coerce").to_numpy()

    keep = ~(np.isnan(yt) | np.isnan(yp))
    yt = yt[keep]; yp = yp[keep]
    if unit is not None:
        unit = unit[keep]
    return yt, yp, unit

def _pos_filter(y_true, y_pred, unit, thr):
    """y_true > thr 필터. thr<0면 전체."""
    if thr is None or float(thr) < 0:
        idx = np.arange(len(y_true))
    else:
        idx = np.where(y_true > float(thr))[0]
    yt = y_true[idx]; yp = y_pred[idx]
    uu = (unit[idx] if unit is not None else None)
    return yt, yp, uu

def _split_by_unit(y_true, y_pred, unit=None, jump_thr=20.0):
    """
    유닛별 그룹으로 분할.
    - unit 배열이 있으면 그것대로 그룹핑.
    - 없으면 y_true의 '증가' 지점을 경계로 자동 분할(리셋 감지).
    반환: [(yt_g, yp_g, unit_id), ...]
    """
    y = np.asarray(y_true); p = np.asarray(y_pred)

    if unit is not None:
        u = np.asarray(unit)
        uniq = np.unique(u)
        groups = []
        for uid in uniq:
            m = (u == uid)
            groups.append((y[m], p[m], int(uid)))
        return groups

    # unit이 없으면 RUL 증가(리셋) 지점으로 분할
    if len(y) <= 1:
        return [(y, p, 1)]
    jumps = np.where(np.diff(y) > float(jump_thr))[0]  # 증가(리셋) 지점
    cuts = np.r_[-1, jumps, len(y)-1]
    groups = []
    for i in range(len(cuts)-1):
        s = cuts[i] + 1
        e = cuts[i+1] + 1
        groups.append((y[s:e], p[s:e], i+1))
    return groups

# ─────────────────────────── 플롯 ───────────────────────────

def plot_timeseries_overlay_raw(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_true, label="Actual", linewidth=1.5)
    ax.plot(y_pred, label="Pred", linewidth=1.2)
    ax.set_title("Timeseries (raw units, overlay)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value (raw)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_timeseries_facet_units(groups, out_path, max_units=12):
    """
    유닛별로 (Actual, Pred) 같은 축에 겹쳐 그려서 한 유닛당 한 행으로 표시.
    """
    K = min(len(groups), max_units)
    fig, axes = plt.subplots(K, 1, figsize=(10, 2.8*K), sharex=False)
    if K == 1:
        axes = [axes]
    for ax, (ytg, ypg, uid) in zip(axes, groups[:K]):
        ax.plot(ytg, label="Actual", linewidth=1.3)
        ax.plot(ypg, label="Pred", linewidth=1.1)
        ax.set_title(f"unit={uid}")
        ax.set_ylabel("RUL")
    axes[-1].set_xlabel("cycle (relative)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Timeseries per unit (raw)", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_scatter_hex(y_true, y_pred, out_path, gridsize=40):
    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=gridsize, mincnt=1)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title("Pred vs Actual (hexbin, pos-only)")
    ax.set_xlabel("Actual (>thr)")
    ax.set_ylabel("Predicted")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_scatter_size(y_true, y_pred, out_path,
                      size_round=0.0, size_scale=20.0,
                      alpha_min=0.15, alpha_max=0.65):
    """동일 좌표 집계하여 버블 크기 = 빈도. 큰 버블일수록 투명."""
    x = y_true.copy()
    y = y_pred.copy()
    if size_round > 0:
        x = np.round(x / size_round) * size_round
        y = np.round(y / size_round) * size_round
    pts = np.vstack([x, y]).T
    uniq, inv, cnts = np.unique(pts, axis=0, return_inverse=True, return_counts=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    # 값 범위로 대각선
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    # 점 찍기(큰 버블일수록 투명)
    cmin, cmax = cnts.min(), cnts.max()
    for u, c in zip(uniq, cnts):
        mask = (x == u[0]) & (y == u[1])
        if cmax > cmin:
            a = alpha_max - ((c - cmin) / (cmax - cmin)) * (alpha_max - alpha_min)
        else:
            a = alpha_max
        ax.scatter(x[mask], y[mask], s=float(c) * size_scale, alpha=float(a), edgecolors="none")
    ax.set_title("Pred vs Actual (size, pos-only)")
    ax.set_xlabel("Actual (>thr)")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_residual_hist(y_true, y_pred, out_path, rule="auto", low_pct=0.5, high_pct=99.5, bins=100):
    res = y_pred - y_true
    if rule == "auto":
        lo = np.percentile(res, low_pct)
        hi = np.percentile(res, high_pct)
        rng = (float(lo), float(hi))
    else:
        rng = None
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(res, bins=bins, range=rng)
    ax.set_title("Residuals (pos-only)")
    ax.set_xlabel("Residual (Pred-Actual)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_calibration_by_ybins(y_true, y_pred, out_path, n_bins=18):
    # y기준 등간격 bin → 각 bin에서 y, pred 평균 비교
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    bins = np.linspace(np.min(y), np.max(y), n_bins + 1)
    idx = np.digitize(y, bins) - 1
    y_means, p_means, xs = [], [], []
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        y_means.append(np.mean(y[m]))
        p_means.append(np.mean(p[m]))
        xs.append(b)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs, y_means, label="Actual (bin mean)")
    ax.plot(xs, p_means, label="Pred (bin mean)")
    ax.set_title("Calibration by y bins (pos-only)")
    ax.set_xlabel("Bins (low→high)")
    ax.set_ylabel("Mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ─────────────────────────── 메인 ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="after_pred.csv 경로")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="y_true > thr만 사용 (음수면 전체). 기본 -1")
    ap.add_argument("--ts-mode", choices=["overlay", "facet"], default="facet",
                    help="시계열 표현 방식 (기본 facet)")
    ap.add_argument("--ts-raw", action="store_true",
                    help="시계열을 원 단위로 그대로 그림(마스크/스케일링 없음).")
    ap.add_argument("--by-unit", choices=["auto","yes","no"], default="auto",
                    help="유닛별로 분리해서 그림(auto: unit 열이 있으면 사용, 없으면 점프 감지).")
    ap.add_argument("--max-units", type=int, default=12,
                    help="facet로 보여줄 최대 유닛 수(많으면 무거움).")
    ap.add_argument("--split-jump", type=float, default=20.0,
                    help="unit 열이 없을 때 y_true 증가가 이 값보다 크면 새 유닛 시작으로 간주.")
    ap.add_argument("--scatter-mode", choices=["hex", "size"], default="hex",
                    help="산점도 표현 방식 (기본 hex)")
    # size scatter 세부
    ap.add_argument("--size-round", type=float, default=0.0, help="좌표 라운딩 간격(0=비활성)")
    ap.add_argument("--size-scale", type=float, default=20.0, help="버블 크기 스케일")
    ap.add_argument("--size-alpha-min", type=float, default=0.15, help="가장 큰 버블의 최소 알파")
    ap.add_argument("--size-alpha-max", type=float, default=0.65, help="가장 작은 버블의 최대 알파")
    # 히스토그램 자동 범위
    ap.add_argument("--hist-rule", choices=["auto", "full"], default="auto",
                    help="히스토그램 범위 자동화(기본 auto)")
    ap.add_argument("--low-pct", type=float, default=0.5)
    ap.add_argument("--high-pct", type=float, default=99.5)
    ap.add_argument("--bins", type=int, default=100)
    args = ap.parse_args()

    pred_path = Path(args.pred).expanduser().resolve()
    out_dir = _ensure_plots_dir(pred_path)

    # 데이터 읽기 + 필터
    y_true_all, y_pred_all, unit_all = _read_pred_csv(pred_path)
    y_true, y_pred, unit = _pos_filter(y_true_all, y_pred_all, unit_all, args.threshold)
    if y_true.size == 0:
        raise SystemExit("선택된 데이터가 없습니다. --threshold를 낮추거나 CSV를 확인하세요.")

    # 유닛 분리 여부 결정
    use_by_unit = (args.by_unit == "yes") or (args.by_unit == "auto" and unit is not None)

    # 유닛별 그룹 분할(없으면 점프 감지)
    groups = _split_by_unit(y_true, y_pred, (unit if use_by_unit else None), jump_thr=args.split_jump)

    # ── Timeseries
    if args.ts_mode == "overlay":
        if len(groups) > 1:
            # 경계 끊기 위해 NaN 삽입 후 한 줄로 overlay
            yt_cat, yp_cat = [], []
            for i, (ytg, ypg, _) in enumerate(groups):
                if i > 0:
                    yt_cat.append(np.nan); yp_cat.append(np.nan)
                yt_cat.extend(ytg); yp_cat.extend(ypg)
            plot_timeseries_overlay_raw(np.array(yt_cat), np.array(yp_cat), out_dir / "ts_overlay_raw.png")
        else:
            plot_timeseries_overlay_raw(groups[0][0], groups[0][1], out_dir / "ts_overlay_raw.png")
    else:
        # facet: 유닛별로 행을 나눠서 그림
        plot_timeseries_facet_units(groups, out_dir / "ts_facet_units.png", max_units=args.max_units)

    # ── Scatter (전체 pos-only 집계)
    if args.scatter_mode == "hex":
        plot_scatter_hex(y_true, y_pred, out_dir / "scatter_hex.png", gridsize=40)
    else:
        plot_scatter_size(
            y_true, y_pred, out_dir / "scatter_size.png",
            size_round=args.size_round, size_scale=args.size_scale,
            alpha_min=args.size_alpha_min, alpha_max=args.size_alpha_max
        )

    # ── Residual histogram
    plot_residual_hist(
        y_true, y_pred, out_dir / "residuals.png",
        rule=args.hist_rule, low_pct=args.low_pct, high_pct=args.high_pct, bins=args.bins
    )

    # ── Calibration
    plot_calibration_by_ybins(y_true, y_pred, out_dir / "calibration_by_y.png", n_bins=18)

    print(f"[OK] saved plots to: {out_dir}")

if __name__ == "__main__":
    main()
