# tools/cmapss_to_channels.py
# CMAPSS 원본 txt(train/test/RUL) → 채널별 CSV 바로 생성
# - train: RUL = T_end - cycle
# - test : RUL = (T_last - cycle) + RUL_truth
# - cap 적용(0이면 미적용), 입력만 z-score(μ/σ from train)
# - 패딩: pad ∈ {ffill, zero, nan}, pad-side ∈ {left, right}
# - 출력:
#     <out>/train/{rul,set1,set2,set3,s1..}.csv
#     <out>/test/{rul,set1,set2,set3,s1..}.csv
#     <out>/channels_order.txt
#
# 예시:
#   python tools/cmapss_to_channels.py --root ./data/CMAPSS --fd FD001 \
#     --out ./data/cmapss_fd001_ch_right --cap 0 --pad ffill --pad-side right --save-scaler

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────── 입출력 ───────────────────────────

def _read_fd_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    assert df.shape[1] >= 5, f"Unexpected column count in {path} (got {df.shape[1]})"
    n_feat = df.shape[1] - 2
    n_set = min(3, n_feat)
    n_sens = n_feat - n_set
    cols = ["unit", "cycle"] + [f"set{i+1}" for i in range(n_set)] + [f"s{i+1}" for i in range(n_sens)]
    df.columns = cols
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    feat_cols = [c for c in df.columns if c not in ("unit", "cycle")]
    df[feat_cols] = df[feat_cols].astype(float)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)

def _read_rul_txt(path: Path) -> pd.Series:
    return pd.read_csv(path, sep=r"\s+", header=None, names=["RUL_truth"])["RUL_truth"].astype(float)

# ─────────────────────────── RUL 생성/가공 ───────────────────────────

def _make_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    mx = df.groupby("unit")["cycle"].max().rename("T_end")
    out = df.merge(mx, on="unit", how="left")
    out["rul"] = out["T_end"] - out["cycle"]
    out.drop(columns=["T_end"], inplace=True)
    return out

def _make_rul_test(df_test: pd.DataFrame, rul_truth: pd.Series) -> pd.DataFrame:
    units = np.sort(df_test["unit"].unique())
    assert len(units) == len(rul_truth), f"RUL lines({len(rul_truth)}) != test units({len(units)})"
    truth_map = {u: float(rul_truth.iloc[i]) for i, u in enumerate(units)}
    mx = df_test.groupby("unit")["cycle"].max().rename("T_last")
    out = df_test.merge(mx, on="unit", how="left")
    out["rul"] = out.apply(lambda r: (r["T_last"] - r["cycle"]) + truth_map[int(r["unit"])], axis=1)
    out.drop(columns=["T_last"], inplace=True)
    return out

def _cap_rul(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap and cap > 0:
        df["rul"] = np.minimum(df["rul"].values, float(cap))
    return df

# ─────────────────────────── 스케일링 ───────────────────────────

def _fit_scaler(train_df: pd.DataFrame, in_cols: list[str]):
    mu = train_df[in_cols].mean()
    sd = train_df[in_cols].std(ddof=0).replace(0, 1.0)
    return mu, sd

def _apply_scaler(df: pd.DataFrame, in_cols: list[str], mu: pd.Series, sd: pd.Series):
    df.loc[:, in_cols] = df[in_cols].astype(float)
    df.loc[:, in_cols] = (df[in_cols] - mu) / sd
    return df

# ─────────────────────────── 패딩 ───────────────────────────

def _pad_seq(arr: np.ndarray, T_out: int, pad: str, side: str) -> np.ndarray:
    """side='left' → 좌측 패딩, side='right' → 우측 패딩."""
    T = len(arr)
    out = np.empty((T_out,), dtype=float)
    if pad == "zero":
        out[:] = 0.0
    else:
        out[:] = np.nan

    if side == "left":
        # [pad ... pad, arr]
        out[-T:] = arr
        if pad == "ffill" and T > 0:
            first = arr[0]
            mask = np.isnan(out)
            out[mask] = first
    else:
        # [arr, pad ... pad]
        out[:T] = arr
        if pad == "ffill" and T > 0:
            last = arr[-1]
            mask = np.isnan(out)
            out[mask] = last
    return out

# ─────────────────────────── 채널 CSV 생성 ───────────────────────────

def _emit_channel_csvs(
    df: pd.DataFrame,
    channels: list[str],
    out_dir: Path,
    pad: str,
    pad_side: str,
    split_name: str
):
    """엔진별 1행, 시간축 t0000..t{T_out-1}. 각 채널을 개별 CSV로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    len_by_u = df.groupby("unit")["cycle"].max().astype(int)
    T_out = int(len_by_u.max())
    units = df["unit"].unique()

    for ch in ["rul"] + channels:
        rows = []
        for u in units:
            d = df[df["unit"] == u].sort_values("cycle")
            seq = d[ch].to_numpy(dtype=float)
            pad_seq = _pad_seq(seq, T_out, pad, pad_side)
            row = {f"{ch}_t{t:04d}": pad_seq[t] for t in range(T_out)}
            row["unit"] = int(u)
            row["T"] = int(len_by_u.loc[u])
            rows.append(row)
        ch_df = pd.DataFrame(rows).set_index("unit")
        ch_df.to_csv(out_dir / f"{ch}.csv")
    (out_dir / "channels_order.txt").write_text("\n".join(["rul"] + channels), encoding="utf-8")
    print(f"[OK] saved {split_name} channels → {out_dir} (T_out={T_out}, channels={1+len(channels)})")

# ─────────────────────────── 메인 ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CMAPSS 폴더 (train_FD001.txt, test_FD001.txt, RUL_FD001.txt ...)")
    ap.add_argument("--fd", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--out", required=True, help="출력 루트 폴더 (여기 아래 train/, test/ 생성)")
    ap.add_argument("--cap", type=int, default=125, help="RUL cap(0=미적용)")
    ap.add_argument("--pad", choices=["nan", "zero", "ffill"], default="ffill", help="패딩 값(기본 ffill)")
    ap.add_argument("--pad-side", choices=["left", "right"], default="right",
                    help="패딩을 어디에 둘지(left/right). 기본 right")
    ap.add_argument("--save-scaler", action="store_true", help="입력 채널 μ/σ CSV 저장")
    args = ap.parse_args()

    root = Path(args.root)
    fd = args.fd
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # 로드
    tr = _read_fd_txt(root / f"train_{fd}.txt")
    te = _read_fd_txt(root / f"test_{fd}.txt")
    rul_truth = _read_rul_txt(root / f"RUL_{fd}.txt")

    # RUL 부여
    tr_r = _make_rul_train(tr)
    te_r = _make_rul_test(te, rul_truth)

    # cap
    tr_r = _cap_rul(tr_r, args.cap)
    te_r = _cap_rul(te_r, args.cap)

    # 입력 목록 (unit,cycle,rul 제외 전부)
    in_cols = [c for c in tr_r.columns if c not in ("unit", "cycle", "rul")]

    # 스케일링 (입력만, train 통계)
    mu, sd = _fit_scaler(tr_r, in_cols)
    tr_s = _apply_scaler(tr_r.copy(), in_cols, mu, sd)
    te_s = _apply_scaler(te_r.copy(), in_cols, mu, sd)

    # 채널별 CSV 생성 (우측/좌측 패딩 선택)
    _emit_channel_csvs(tr_s, in_cols, train_dir, args.pad, args.pad_side, split_name="train")
    _emit_channel_csvs(te_s, in_cols, test_dir,  args.pad, args.pad_side, split_name="test")

    # 스케일러 저장(선택)
    if args.save_scaler:
        pd.DataFrame({"mu": mu, "sd": sd}).to_csv(out_root / f"{fd}_scaler_inputs.csv")

    info = {
        "fd": fd,
        "train_units": int(tr["unit"].nunique()),
        "test_units": int(te["unit"].nunique()),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "channels_out": 1 + len(in_cols),
        "features_in": len(in_cols),
        "cap": int(args.cap),
        "pad": args.pad,
        "pad_side": args.pad_side,
    }
    print("[OK] CMAPSS→channels done:", info)

    # 편의: 학습/추론용 --csv 라인 프린트
    train_list = ",".join((train_dir / f"{ch}.csv").as_posix() for ch in ["rul"] + in_cols)
    test_list  = ",".join((test_dir  / f"{ch}.csv").as_posix() for ch in ["rul"] + in_cols)
    print("\n# train --csv (copy/paste):")
    print(train_list)
    print("\n# test  --csv (copy/paste):")
    print(test_list)

if __name__ == "__main__":
    main()
