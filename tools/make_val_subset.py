#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_val_subset.py
- groups.csv + (val_ratio, seed)로 train/val을 재현하고
  val 유닛만 뽑아 별도 폴더에 CSV를 저장한다.
- 이렇게 만든 폴더로 --mode infer 를 돌리면, 학습 시의 validation과
  동일 분할로 "오염 없이" 평가할 수 있다.
"""

import argparse, csv, os, sys, random

NUM_AUX = 23

def read_groups(path):
    unit_ids, valid_lens = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row: continue
            unit_ids.append(int(row[1]))
            valid_lens.append(int(float(row[2])))
    return unit_ids, valid_lens

def derive_split(units, val_ratio=0.2, seed=777):
    uniq = sorted(set(units))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    val_n = max(1, int(round(n * float(val_ratio))))
    val_units = set(uniq[:val_n])
    train_units = set(uniq[val_n:])
    return train_units, val_units

def read_matrix(path):
    mat = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if row: mat.append(row)
    return mat  # as strings keep precision

def write_matrix(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="원본 데이터 폴더 (main.csv, aux01..aux23.csv, groups.csv)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", required=True, help="val 전용 하위 폴더 경로")
    args = ap.parse_args()

    groups_p = os.path.join(args.data, "groups.csv")
    if not os.path.isfile(groups_p):
        print(f"[ERROR] groups.csv not found in {args.data}", file=sys.stderr); sys.exit(1)

    unit_ids, valid_lens = read_groups(groups_p)
    train_u, val_u = derive_split(unit_ids, args.val_ratio, args.seed)

    # 원본 행 인덱스 중, val 유닛에 해당하는 인덱스 모으기
    keep_idx = [i for i,u in enumerate(unit_ids) if u in val_u]

    # 행 슬라이스 함수
    def slice_rows(file_in, file_out):
        rows = read_matrix(file_in)
        picked = [rows[i] for i in keep_idx]
        write_matrix(file_out, picked)

    os.makedirs(args.out, exist_ok=True)
    # main + aux 채널 저장
    slice_rows(os.path.join(args.data, "main.csv"), os.path.join(args.out, "main.csv"))
    for k in range(1, NUM_AUX+1):
        fn = f"aux{k:02d}.csv"
        slice_rows(os.path.join(args.data, fn), os.path.join(args.out, fn))

    # groups.csv 새로 쓰기 (row_idx 재부여)
    with open(os.path.join(args.out, "groups.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_idx","unit_id","valid_len"])
        for new_i, old_i in enumerate(keep_idx):
            w.writerow([new_i, unit_ids[old_i], valid_lens[old_i]])

    # 유닛 목록도 남겨두면 편함
    with open(os.path.join(args.out, "val_units.txt"), "w", encoding="utf-8") as f:
        for u in sorted(val_u):
            f.write(str(u) + "\n")

    print(f"[OK] made val subset at: {args.out}")
    print(f" - val units: {len(val_u)}  (train units: {len(train_u)})")
    print(f" - rows kept: {len(keep_idx)}")
    print(f" - files: main.csv, aux01..aux23.csv, groups.csv, val_units.txt")

if __name__ == "__main__":
    main()
