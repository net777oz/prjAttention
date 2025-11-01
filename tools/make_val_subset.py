#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_val_subset.py
--------------------------------------------------------------------
목적:
- groups.csv 기반으로 train/val을 "unit 단위"로 재현하여,
  선택한 split 서브셋만 별도 폴더에 저장한다.
- 이렇게 만든 폴더를 --mode infer로 평가하면, 학습 시 분할과 동일하게
  데이터 누수 없이 성능을 측정할 수 있다.

주요 기능(개선):
- --mode {val,train}        : 검증뿐 아니라 학습 서브셋도 생성 가능
- --units <txt>             : 유닛 ID 목록 파일로 정확한 분할 재현(학습 때 저장한 목록 재사용)
- --val-ratio/--seed        : 학습 시와 동일 파라미터로 재현 (units 파일이 없을 때 사용)
- --context/--rule/--thr    : 생성된 서브셋에 대해 Δ=+1 라벨 구간(열 L 이후)의 양성률 즉시 리포트
- sources.csv (존재 시)     : 동일 행 인덱스로 필터링하여 함께 생성
- CSV/헤더/정합 검사        : 행 수/열 수 불일치, groups 파싱 잡음에 안전

사용 예:
  # 1) 검증 서브셋 만들기 (학습 때 val_ratio=0.2, seed=777을 썼다면 그대로)
  python tools/make_val_subset.py \
    --data ./data/cmapss_all_full \
    --val-ratio 0.2 --seed 777 \
    --mode val \
    --out ./data/cmapss_all_full_val \
    --context 31 --rule nonzero

  # 2) 학습 때 저장해둔 val_units.txt로 정확히 재현
  python tools/make_val_subset.py \
    --data ./data/cmapss_all_full \
    --units ./data/cmapss_all_full_val/val_units.txt \
    --mode val \
    --out ./data/cmapss_all_full_val_re

참고:
- Δ=+1 라벨을 쓴다면, L=context 이후(main[:, L:])에 양성이 "0이 아닌지" 꼭 확인.
"""

import argparse, csv, os, sys, random
from typing import List, Tuple

NUM_AUX = 23

# ---------------------------
# helpers
# ---------------------------
def read_groups(path: str) -> Tuple[List[int], List[int]]:
    """groups.csv에서 unit_id, valid_len을 읽는다.
    헤더/여분 컬럼(예: fd)이 있어도 동작."""
    unit_ids, valid_lens = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        # header가 실제 데이터일 수도 있으니 유연하게 처리
        def _try_row(row):
            try:
                uid = int(row[1]); vlen = int(float(row[2]))
                return uid, vlen
            except Exception:
                return None
        if header and len(header) >= 3:
            maybe = _try_row(header)
            if maybe:
                uid, vlen = maybe
                unit_ids.append(uid); valid_lens.append(vlen)
        for row in r:
            if not row: 
                continue
            maybe = _try_row(row)
            if not maybe:
                raise ValueError(f"Invalid groups row: {row}")
            uid, vlen = maybe
            unit_ids.append(uid); valid_lens.append(vlen)
    return unit_ids, valid_lens

def derive_split(units: List[int], val_ratio=0.2, seed=777):
    """unit의 집합을 셔플해서 train/val로 나눈다(그룹 분할 재현)."""
    uniq = sorted(set(units))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    val_n = max(1, int(round(n * float(val_ratio))))
    val_units = set(uniq[:val_n])
    train_units = set(uniq[val_n:])
    return train_units, val_units

def read_matrix(path: str) -> List[List[str]]:
    """숫자 정밀도 보존을 위해 문자열 그대로 읽는다."""
    mat = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if row: mat.append(row)
    return mat

def write_matrix(path: str, rows: List[List[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)

def read_units_txt(path: str) -> List[int]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            ids.append(int(s))
    return ids

def compute_label_stats(main_path: str, keep_idx: List[int], L: int, rule: str, thr: float):
    """Δ=+1 라벨 구간(main[:, L:])에서 양성률을 계산한다."""
    import numpy as np
    # 메모리 효율을 위해 전체 로드 후 행 슬라이스
    m = []
    with open(main_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i in keep_idx:
                m.append([float(x) for x in row])
    if not m:
        return 0, 0, 0.0, 0
    arr = np.array(m, dtype=np.float32)  # [K, T]
    T = arr.shape[1]
    if L >= T:
        return 0, arr.size, 0.0, T
    tail = arr[:, L:]                    # Δ=+1일 때 라벨이 가리키는 열 집합
    if rule == "nonzero":
        pos = int((tail != 0).sum())
    else:  # le
        pos = int((tail <= thr).sum())
    total = int(tail.size)
    rate = (pos / total) if total else 0.0
    return pos, total, rate, T

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Create train/val subset (group split) for inference without leakage."
    )
    ap.add_argument("--data", required=True,
                    help="원본 데이터 폴더 (main.csv, aux01..aux23.csv, groups.csv)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="단위: unit 그룹 분할 비율 (units 파일이 없을 때 사용)")
    ap.add_argument("--seed", type=int, default=777,
                    help="그룹 분할 셔플 시드 (units 파일이 없을 때 사용)")
    ap.add_argument("--mode", choices=["val","train"], default="val",
                    help="만들 서브셋 종류 (기본: val)")
    ap.add_argument("--units", type=str, default=None,
                    help="선택: 특정 유닛ID 목록 파일(한 줄 하나). 주어지면 이 집합만 선택")
    ap.add_argument("--out", required=True,
                    help="서브셋 출력 폴더")
    ap.add_argument("--context", type=int, default=None,
                    help="선택: Δ=+1 라벨 구간 점검을 위한 L(context). 지정 시 라벨 양성률 출력")
    ap.add_argument("--rule", choices=["nonzero","le"], default="nonzero",
                    help="라벨 양성 정의: nonzero 또는 '≤ thr' (le)")
    ap.add_argument("--thr", type=float, default=0.0,
                    help="--rule le일 때 임계값(예: RUL≤30)")
    args = ap.parse_args()

    # 파일 체크
    main_p   = os.path.join(args.data, "main.csv")
    groups_p = os.path.join(args.data, "groups.csv")
    if not os.path.isfile(main_p) or not os.path.isfile(groups_p):
        print(f"[ERROR] main.csv or groups.csv not found in {args.data}", file=sys.stderr)
        sys.exit(1)

    unit_ids, valid_lens = read_groups(groups_p)
    if len(unit_ids) == 0:
        print("[ERROR] groups.csv is empty", file=sys.stderr); sys.exit(1)

    # 선택 유닛 결정
    if args.units:
        # 목록 파일을 그대로 사용
        pick_units = set(read_units_txt(args.units))
        if not pick_units:
            print(f"[ERROR] units file empty: {args.units}", file=sys.stderr); sys.exit(1)
        # mode가 train이면 목록의 보수집합을 취할 수 있게 해도 되지만,
        # 여기선 주어진 목록 그대로 사용(일관성 유지).
        selected_units = pick_units
    else:
        # 학습 시 분할 재현(랜덤 시드/비율)
        train_u, val_u = derive_split(unit_ids, args.val_ratio, args.seed)
        selected_units = val_u if args.mode == "val" else train_u

    # 행 인덱스 선택
    keep_idx = [i for i, u in enumerate(unit_ids) if u in selected_units]
    if len(keep_idx) == 0:
        print("[ERROR] no rows selected. Check --mode/--val-ratio/--seed or --units.", file=sys.stderr)
        sys.exit(1)

    def slice_rows(file_in: str, file_out: str):
        rows = read_matrix(file_in)
        # 정합: 행 수 일치
        if len(rows) != len(unit_ids):
            print(f"[ERROR] row count mismatch in {file_in}: rows={len(rows)} vs groups={len(unit_ids)}",
                  file=sys.stderr); sys.exit(1)
        picked = [rows[i] for i in keep_idx]
        write_matrix(file_out, picked)

    os.makedirs(args.out, exist_ok=True)

    # main + aux 채널 저장
    slice_rows(main_p, os.path.join(args.out, "main.csv"))
    for k in range(1, NUM_AUX+1):
        fn = f"aux{k:02d}.csv"
        in_p = os.path.join(args.data, fn)
        if not os.path.isfile(in_p):
            print(f"[ERROR] missing {in_p}", file=sys.stderr); sys.exit(1)
        slice_rows(in_p, os.path.join(args.out, fn))

    # groups.csv 새로 쓰기 (row_idx 재부여)
    with open(os.path.join(args.out, "groups.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","unit_id","valid_len"])
        for new_i, old_i in enumerate(keep_idx):
            w.writerow([new_i, unit_ids[old_i], valid_lens[old_i]])

    # sources.csv(있으면)도 동일하게 필터링
    src_in = os.path.join(args.data, "sources.csv")
    if os.path.isfile(src_in):
        with open(src_in, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            rows = [row for row in r if row]  # row_idx,source,orig_row_idx,orig_unit_id,orig_valid_len
        # header 없는 케이스도 허용하지 않음(명시적 형식 가정). 행수 정합은 보장 못함.
        # row_idx를 믿고 필터링
        row_idx_col = 0
        rows_map = {int(row[row_idx_col]): row for row in rows if row}
        picked = [rows_map[i] for i in keep_idx if i in rows_map]
        with open(os.path.join(args.out, "sources.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row_idx","source","orig_row_idx","orig_unit_id","orig_valid_len"])
            for new_i, old_i in enumerate(keep_idx):
                if old_i in rows_map:
                    r = rows_map[old_i]
                    r2 = [new_i] + r[1:]  # row_idx만 갱신
                    w.writerow(r2)

    # 유닛 목록 저장(편의)
    tag = args.mode
    with open(os.path.join(args.out, f"{tag}_units.txt"), "w", encoding="utf-8") as f:
        for u in sorted(set(unit_ids[i] for i in keep_idx)):
            f.write(str(u) + "\n")

    print(f"[OK] made {args.mode} subset at: {args.out}")
    print(f" - {args.mode} units: {len(set(unit_ids[i] for i in keep_idx))}")
    print(f" - rows kept: {len(keep_idx)}")
    print(f" - files: main.csv, aux01..aux23.csv, groups.csv, {tag}_units.txt"
          + (", sources.csv" if os.path.isfile(src_in) else ""))

    # (선택) Δ=+1 라벨 구간 점검
    if args.context is not None:
        pos, total, rate, T = compute_label_stats(main_p, keep_idx, args.context, args.rule, args.thr)
        print(f"[debug] label@Δ=+1 (context L={args.context}, rule={args.rule}, thr={args.thr}) "
              f"pos={pos}/{total} rate={rate:.6f} (T={T})")
        if total > 0 and pos == 0:
            print("[WARN] 선택된 서브셋에서 Δ=+1 라벨 구간이 전부 0입니다. "
                  "패딩/정렬 또는 이진 규칙을 점검하세요.", file=sys.stderr)

if __name__ == "__main__":
    main()
