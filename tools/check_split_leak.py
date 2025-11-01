#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split Leakage Checker for AttentionProject

목적:
- 학습 시 내부 분할(특히 --split-mode group)에서
  Train/Validation/Test 간 "유닛(엔진 ID)" 누수가 없는지 점검합니다.

지원 모드:
1) groups.csv 한 개만 있을 때(분할 전 데이터):
   - --split-mode group, --val-ratio, --seed 로 동일 규칙으로 "예상 분할"을 재현하여
     (train_units, val_units)를 산출하고 교집합이 비어있는지 확인.

2) 분할 결과 유닛 목록이 따로 있을 때:
   - --train-units 파일, --val-units 파일(각각 한 줄당 정수 unit_id)
     → 두 집합의 교집합을 직접 검증.

3) 폴더별 groups.csv 두 개가 있을 때:
   - --train-groups, --val-groups 로 각 폴더의 groups.csv를 지정
     → 두 폴더의 유닛 교집합을 직접 검증.
"""

import argparse
import csv
import os
import sys
import random

def read_groups_csv(path):
    """
    groups.csv 형식: row_idx,unit_id,valid_len(, ...optional)
    """
    units = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            try:
                unit_id = int(row[1])
            except Exception:
                # 헤더/이상행은 스킵
                continue
            units.append(unit_id)
    return units

def read_units_list(path):
    """
    한 줄당 정수 unit_id
    """
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s.add(int(line))
    return s

def derive_group_split(units, val_ratio=0.2, seed=777):
    """
    units: [unit_id,...] (중복 허용 입력 → 내부에서 set)
    단순 셔플 후 앞쪽을 val로, 나머지를 train으로.
    (학습 코드가 동일 규칙을 쓴다는 가정하에 재현 가능)
    """
    uniq = sorted(set(units))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    val_n = max(1, int(round(n * float(val_ratio))))
    val_units = set(uniq[:val_n])
    train_units = set(uniq[val_n:])
    return train_units, val_units

def summarize(train_units, val_units, label_train="train", label_val="val", show_overlap=20):
    inter = train_units & val_units
    print(f"[INFO] {label_train} units: {len(train_units)}")
    print(f"[INFO] {label_val}   units: {len(val_units)}")
    print(f"[CHECK] intersection({label_train} ∩ {label_val}) = {len(inter)}")
    if inter:
        sample = sorted(list(inter))[:show_overlap]
        print(f"[ALERT] Leakage suspected! Sample overlaps (up to {show_overlap}): {sample}")
    else:
        print("[OK] No unit overlap detected.")

def main():
    ap = argparse.ArgumentParser(description="Leakage check for AttentionProject splits")
    # 모드 1: 분할 전 원본 groups.csv + 규칙으로 재현
    ap.add_argument("--groups", type=str, help="분할 전 데이터의 groups.csv 경로")
    ap.add_argument("--split-mode", type=str, default="group",
                    choices=["group","item","time","window"],
                    help="스플릿 모드(재현용). 현재는 group만 신뢰 가능.")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="검증 비율 (재현용)")
    ap.add_argument("--seed", type=int, default=777, help="재현용 시드")

    # 모드 2: 유닛 목록 파일 직접 비교
    ap.add_argument("--train-units", type=str, help="train 유닛 ID 목록 파일(한 줄당 정수)")
    ap.add_argument("--val-units", type=str, help="val 유닛 ID 목록 파일(한 줄당 정수)")

    # 모드 3: 폴더별 groups.csv 비교
    ap.add_argument("--train-groups", type=str, help="train 폴더의 groups.csv 경로")
    ap.add_argument("--val-groups", type=str, help="val 폴더의 groups.csv 경로")

    # 표시 옵션
    ap.add_argument("--show-overlap", type=int, default=20, help="교집합 샘플 표시 개수")
    args = ap.parse_args()

    # 우선순위: (2) → (3) → (1)
    if args.train_units and args.val_units:
        tr = read_units_list(args.train_units)
        va = read_units_list(args.val_units)
        summarize(tr, va, "train(file)", "val(file)", args.show_overlap)
        return

    if args.train_groups and args.val_groups:
        tr = set(read_groups_csv(args.train_groups))
        va = set(read_groups_csv(args.val_groups))
        summarize(tr, va, "train(groups)", "val(groups)", args.show_overlap)
        return

    if args.groups:
        if args.split_mode != "group":
            print("[WARN] 현재 스크립트는 'group' 분할 재현만 신뢰할 수 있습니다. "
                  "다른 모드는 학습 코드와 불일치할 수 있습니다.",
                  file=sys.stderr)
        units = set(read_groups_csv(args.groups))
        tr, va = derive_group_split(list(units), val_ratio=args.val_ratio, seed=args.seed)
        summarize(tr, va, "train(derived)", "val(derived)", args.show_overlap)
        return

    print("[ERROR] 실행 모드를 선택하세요.\n"
          "  - (권장) --groups <path>  (분할 전 groups.csv로 재현)\n"
          "  - 또는 --train-groups <path> --val-groups <path>\n"
          "  - 또는 --train-units <file> --val-units <file>\n", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
