# tools/plot_regress_pos_only.py
# 회귀 결과(pred.csv 등)에서 y_true > threshold 구간만 골라 시각화
# 기본: 타임시리즈=facet(위: Actual, 아래: Pred), 스캐터=hexbin(색=빈도)

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CAND_TRUE = ["y_true","y","target","label","gt","truth"]
CAND_PRED = ["y_pred","pred","yhat","y_hat","prediction"]

def _guess_cols(df: pd.DataFrame):
    y = next((c for c in CAND_TRUE if c in df.columns), None)
    p = next((c for c in CAND_PRED if c in df.columns), None)
    if not y or not p:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num) >= 2:
            y = y or num[0]; p = p or num[1]
    if not y or not p:
        raise ValueError(f"y/pred 컬럼을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}")
    return y, p

def _auto_hist_range(arr: np.ndarray, low_pct=0.5, high_pct=99.5):
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    pad = 0.02 * max(1e-12, hi - lo)
    return lo - pad, hi + pad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="예측 결과 CSV (예: run_xxx/pred.csv)")
    ap.add_argument("--threshold", type=float, default=0.0, help="y_true > threshold 필터")
    ap.add_argument("--outdir", default=None, help="출력 디렉토리(기본: pred.csv와 같은 폴더)")

    # 히스토그램 옵션
    ap.add_argument("--bins", type=int, default=None, help="정수 지정 시 고정 bin 개수")
    ap.add_argument("--hist-rule", dest="hist_rule", default="auto",
                    help="auto/fd/doane/scott/sturges/stone/rice/sqrt 중 선택(문자 규칙은 Matplotlib가 처리)")
    ap.add_argument("--low-pct", type=float, default=0.5, help="히스토그램 표시 하한 퍼센타일")
    ap.add_argument("--high-pct", type=float, default=99.5, help="히스토그램 표시 상한 퍼센타일")
    ap.add_argument("--save-full", action="store_true", help="전체 범위 히스토그램도 추가 저장")

    # 타임시리즈: 기본 facet
    ap.add_argument("--ts-mode", choices=["twin","overlay","facet"], default="facet",
                    help="twin: 좌/우 y축 분리 겹치기 | overlay: Pred를 0~2로 스케일해 한 축에 | facet: 위/아래 두 패널")
    ap.add_argument("--ts-y2-fixed-02", action="store_true",
                    help="twin 모드에서 우측 y축을 0~2로 고정")

    # 스캐터: 기본 hex
    ap.add_argument("--scatter-mode", choices=["hex","size","plain"], default="hex",
                    help="hex: hexbin 색=빈도 | size: 마커 크기/투명도=빈도 | plain: 일반 산점도")
    # size 모드용 (큰 점일수록 더 연하게)
    ap.add_argument("--size-alpha-min", type=float, default=0.25, help="최소 불투명도(가장 연함 한계)")
    ap.add_argument("--size-alpha-max", type=float, default=0.85, help="최대 불투명도(가장 진함 한계)")
    ap.add_argument("--size-round", type=int, default=2, help="좌표 라운딩 자릿수(집계용)")

    args = ap.parse_args()

    # 입출력
    df = pd.read_csv(args.pred)
    ycol, pcol = _guess_cols(df)
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.pred)) or "."
    os.makedirs(outdir, exist_ok=True)

    # time-like 정렬
    for tcol in ["t","time","step","idx","index","window_idx"]:
        if tcol in df.columns:
            df = df.sort_values(tcol).reset_index(drop=True); break

    # 필터링
    d = df[df[ycol] > args.threshold].copy()
    if d.empty:
        raise SystemExit(f"y_true > {args.threshold} 조건을 만족하는 행이 없습니다.")

    y = d[ycol].to_numpy()
    p = d[pcol].to_numpy()
    r = p - y

    # ───────────────── 1) 스캐터 (빈도 시각화) ─────────────────
    plt.figure()
    if args.scatter_mode == "hex":
        # 색 진할수록 count↑. alpha로 겹침 과도함을 완화.
        hb = plt.hexbin(y, p, gridsize=35, mincnt=1, alpha=0.85)
        plt.colorbar(hb, label="Count")
    elif args.scatter_mode == "size":
        # 좌표를 버킷팅해 동일 좌표 빈도를 크기/투명도로 표현
        ay = np.round(y, args.size_round); ap = np.round(p, args.size_round)
        coords, counts = np.unique(np.stack([ay, ap], axis=1), axis=0, return_counts=True)
        # 크기: 루트 스케일로 과도한 겹침 완화
        sizes = 10 + 12*np.sqrt(counts.astype(float))
        # 알파: count↑ -> 더 투명(겹침 시 아래가 보이도록)
        cmin, cmax = counts.min(), counts.max()
        if cmax == cmin:
            alphas = np.full_like(counts, (args.size_alpha_min+args.size_alpha_max)/2, dtype=float)
        else:
            norm = (counts - cmin) / (cmax - cmin)
            alphas = args.size_alpha_max - norm * (args.size_alpha_max - args.size_alpha_min)
        base_rgb = np.array(mcolors.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][0]))
        colors = np.concatenate([np.tile(base_rgb, (len(alphas),1)), alphas[:,None]], axis=1)
        plt.scatter(coords[:,0], coords[:,1], s=sizes, c=colors)
    else:
        plt.scatter(y, p, s=10, alpha=0.6)

    mn, mx = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel(f"Actual (>{args.threshold})"); plt.ylabel("Predicted")
    plt.title("Pred vs Actual (pos-only)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pred_pos_scatter.png"), dpi=150)
    plt.close()

    # ───────────────── 2) 타임시리즈 (y축 처리) ─────────────────
    if args.ts_mode == "facet":
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(y, label="Actual")
        ax1.set_ylabel("Actual"); ax1.set_ylim(0, 2.05); ax1.legend()

        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        ax2.plot(p, label="Pred")
        if args.ts_y2_fixed_02:
            ax2.set_ylim(0, 2.05)
        ax2.set_ylabel("Pred"); ax2.set_xlabel("Filtered index"); ax2.legend()

        fig.suptitle("Timeseries (pos-only, facet)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "pred_pos_timeseries.png"), dpi=150)
        plt.close(fig)

    elif args.ts_mode == "overlay":
        # Pred를 0~2로 선형 스케일해 한 축 위에 겹치기
        p_min, p_max = float(np.min(p)), float(np.max(p))
        p_scaled = (p - p_min) / max(1e-12, (p_max - p_min)) * 2.0
        plt.figure()
        plt.plot(y, label="Actual")
        plt.plot(p_scaled, label="Pred (scaled to 0~2)")
        plt.ylim(0, 2.05)
        plt.xlabel("Filtered index"); plt.ylabel("Value (0~2)")
        plt.title("Timeseries (pos-only, overlay)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pred_pos_timeseries.png"), dpi=150)
        plt.close()

    else:  # twin
        fig, ax1 = plt.subplots()
        l1, = ax1.plot(y, label="Actual")
        ax1.set_ylabel("Actual"); ax1.set_ylim(0, 2.05)
        ax2 = ax1.twinx()
        l2, = ax2.plot(p, label="Pred")
        if args.ts_y2_fixed_02:
            ax2.set_ylim(0, 2.05)
        ax1.set_xlabel("Filtered index")
        fig.suptitle("Timeseries (pos-only)")
        ax1.legend([l1, l2], ["Actual", "Pred"], loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "pred_pos_timeseries.png"), dpi=150)
        plt.close(fig)

    # ───────────────── 3) 잔차 히스토그램 (자동 범위) ─────────────────
    lo, hi = _auto_hist_range(r, args.low_pct, args.high_pct)
    bins = args.bins if (args.bins and args.bins > 0) else args.hist_rule

    if args.save_full:
        plt.figure()
        plt.hist(r, bins=bins)
        plt.xlabel("Residual (Pred-Actual)"); plt.ylabel("Count")
        plt.title("Residuals (FULL)"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "residuals_pos_hist_full.png"), dpi=150)
        plt.close()

    plt.figure()
    plt.hist(r, bins=bins, range=(lo, hi))
    plt.xlabel("Residual (Pred-Actual)"); plt.ylabel("Count")
    plt.title("Residuals (pos-only, auto-range)"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "residuals_pos_hist.png"), dpi=150)
    plt.close()

    # ───────────────── 4) 보정(구간 평균) ─────────────────
    try:
        k = 20
        bins_for_cal = np.linspace(y.min(), y.max(), k+1)
        idx = np.digitize(y, bins_for_cal) - 1
        Yb, Pb = [], []
        for b in range(k):
            m = idx == b
            if np.any(m):
                Yb.append(np.mean(y[m])); Pb.append(np.mean(p[m]))
        if len(Yb) >= 2:
            plt.figure()
            plt.plot(Yb, label="Actual (bin mean)")
            plt.plot(Pb, label="Pred (bin mean)")
            plt.xlabel("Bins (low→high)"); plt.ylabel("Mean")
            plt.title("Calibration by y bins (pos-only)")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "calibration_pos.png"), dpi=150)
            plt.close()
    except Exception as e:
        print("[WARN] calibration skipped:", e)

    # ───────────────── 5) 요약 저장 ─────────────────
    mae = float(np.mean(np.abs(r)))
    rmse = float(np.sqrt(np.mean(r**2)))
    ss_res = float(np.sum(r**2))
    ss_tot = float(np.sum((y - y.mean())**2)) if len(y) > 1 else np.nan
    r2 = 1.0 - ss_res/ss_tot if np.isfinite(ss_tot) and ss_tot > 0 else np.nan

    pd.Series({
        "n": int(len(y)),
        "threshold": float(args.threshold),
        "mae_pos": mae, "rmse_pos": rmse, "r2_pos": r2,
        "y_true_mean_pos": float(y.mean()),
        "y_pred_mean_pos": float(p.mean()),
        "hist_rule": str(args.hist_rule),
        "hist_range": [lo, hi],
        "low_pct": float(args.low_pct), "high_pct": float(args.high_pct),
        "ts_mode": args.ts_mode, "scatter_mode": args.scatter_mode,
        "ts_y2_fixed_02": bool(args.ts_y2_fixed_02),
    }).to_json(os.path.join(outdir, "metrics_pos_only.json"), indent=2)

    print("[OK] Saved plots/metrics →", outdir)

if __name__ == "__main__":
    main()
