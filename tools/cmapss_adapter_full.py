#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
C-MAPSS → AttentionProject 24채널 레이아웃 어댑터 (main 1 + aux 23)
-------------------------------------------------------------------

기능 개요
- NASA C-MAPSS (FD001~FD004) 학습 TXT를 읽어 [N, C, T] 형태로 변환
- 타깃(main.csv)은 **RUL 기반 이진 라벨**:
    y_t = 1  if RUL_t ≤ rul_thr
          0  otherwise
- 입력(aux01..23.csv)은 운전조건 3개 + 센서 20개 (s1..s21 중 제외 목록 뺀 20)
- 유닛별 길이 불균등 → 공통 길이 T_max로 패딩
  *기본은 align='right'로 **왼쪽 패딩**(미래 보존)*
- FD 단일/복수 합본 모두 지원

실무 팁
- 윈도우 라벨이 "윈도우 다음 1칸(Δ=+1)"이면, **미래 칸을 보존**해야 함.
  align='right'(왼쪽 0패딩)으로 만들어야 `main[:, L:]`가 모두 0이 되는 사태를 피할 수 있음.
- 스크립트 끝의 디버그 요약을 반드시 확인:
  *[debug] label@Δ=+1 (context=L) pos=.../total=... rate=...*
  → pos가 0이면, 파이프라인(윈도우/스플릿/이진화) 앞서 데이터부터 재확인 필요.

입력 (root 예: ./data/CMAPSS):
  train_FD001.txt, train_FD002.txt, ...

출력 (out 예: ./data/cmapss_all_full):
  main.csv             : [N, T]  (0/1 라벨)
  aux01.csv..aux23.csv : [N, T]  (입력 채널)
  groups.csv           : row_idx, unit_id, valid_len, fd

사용 예
  # FD001 단일
  python cmapss_adapter_full.py --root ./data/CMAPSS --fd FD001 --out ./data/cmapss_fd001_full

  # FD001~FD004 합본
  python cmapss_adapter_full.py --root ./data/CMAPSS --fds FD001,FD002,FD003,FD004 \
         --out ./data/cmapss_all_full --align right --rul-thr 30 --context 31
"""

import argparse, os, sys, csv
from collections import defaultdict
from typing import Dict, List, Tuple


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="C-MAPSS → AttentionProject 24ch adapter (main+aux* CSVs)"
    )
    p.add_argument("--root", type=str, required=True,
                   help="C-MAPSS 원본 TXT 폴더 (예: ./data/CMAPSS)")
    p.add_argument("--fd", type=str, default=None,
                   help="단일 FD 세트명 (FD001|FD002|FD003|FD004). "
                        "멀티는 --fds 사용.")
    p.add_argument("--fds", type=str, default=None,
                   help="복수 FD 세트: 콤마로 구분 (예: FD001,FD002,FD003,FD004)")
    p.add_argument("--rul-thr", type=int, default=30,
                   help="RUL 임계 (≤ thr ⇒ 1)")
    p.add_argument("--out", type=str, required=True,
                   help="출력 폴더 (예: ./data/cmapss_all_full)")
    p.add_argument("--exclude-sensors", type=str, default="s1",
                   help="제외할 센서들 (콤마구분, 예: 's1' 또는 's1,s5')")
    p.add_argument("--align", choices=["right","left"], default="right",
                   help="패딩 정렬 방식: right=왼쪽 0패딩(미래 보존, 권장), "
                        "left=오른쪽 0패딩(미래 소실 위험)")
    p.add_argument("--context", type=int, default=None,
                   help="(선택) 윈도우 길이 L. 지정 시 Δ=+1 라벨 구간 양성비를 디버그 출력")
    return p.parse_args()


# ----------------------------
# IO helpers
# ----------------------------
def read_train_txt(path: str) -> Dict[int, List[Tuple[int, List[float], List[float]]]]:
    """
    C-MAPSS 학습 TXT 한 파일을 파싱하여 unit별 시퀀스로 반환.
    반환: unit -> [(cycle, settings(3), sensors(21)) ...]  (cycle 오름차순)
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 26:
                continue  # 비정상 라인 스킵
            vals = list(map(float, parts))
            unit  = int(vals[0])
            cycle = int(vals[1])
            setting = vals[2:5]   # 3
            sensors = vals[5:26]  # 21 (s1..s21)
            rows.append((unit, cycle, setting, sensors))

    data = defaultdict(list)
    for unit, cycle, setting, sensors in rows:
        data[unit].append((cycle, setting, sensors))
    for u in data:
        data[u].sort(key=lambda x: x[0])
    return data


def write_csv(path: str, mat: List[List[float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in mat:
            w.writerow([f"{v:.6f}" for v in row])


# ----------------------------
# Core building
# ----------------------------
def pad_to_Tmax(seq: List[float], T_max: int, align: str) -> List[float]:
    """한 유닛의 길이 T를 T_max로 패딩하여 반환.
    align='right' → 왼쪽 0패딩 (미래 보존)
    align='left'  → 오른쪽 0패딩 (미래 소실 위험)"""
    T = len(seq)
    if T == T_max:
        return seq[:]
    pad_len = T_max - T
    if align == "right":
        return [0.0]*pad_len + seq
    else:
        return seq + [0.0]*pad_len


def main():
    args = parse_args()

    # FD 목록 해석
    if args.fds:
        fds = [fd.strip().upper() for fd in args.fds.split(",") if fd.strip()]
    elif args.fd:
        fds = [args.fd.strip().upper()]
    else:
        print("[ERROR] --fd 또는 --fds 중 하나는 지정해야 합니다.", file=sys.stderr)
        sys.exit(1)

    # 채널 구성
    setting_names = ["setting1", "setting2", "setting3"]
    exclude = set([s.strip().lower() for s in args.exclude_sensors.split(",") if s.strip()])
    sensor_all = [f"s{i}" for i in range(1, 22)]  # s1..s21
    sensor_keep = [s for s in sensor_all if s.lower() not in exclude]
    if len(sensor_keep) != 20:
        if len(sensor_keep) < 20:
            print(f"[WARN] sensors kept = {len(sensor_keep)}; expected 20. Will pad by reusing last.",
                  file=sys.stderr)
            while len(sensor_keep) < 20:
                sensor_keep.append(sensor_keep[-1] if sensor_keep else "s21")
        else:
            print(f"[WARN] sensors kept = {len(sensor_keep)}; expected 20. Will trim to 20.",
                  file=sys.stderr)
            sensor_keep = sensor_keep[:20]
    aux_names = setting_names + sensor_keep  # 23
    s_name_to_idx = {f"s{i}": (i-1) for i in range(1, 22)}

    print(f"[INFO] FDs={fds}, rul_thr={args.rul_thr}, align={args.align}")
    print(f"[INFO] AUX channels (23) = {', '.join(aux_names)}")

    # ---------- FD별 파싱 후 합본 ----------
    # 합본 규칙: FD 구분을 유지한 채 row(유닛)들을 이어붙임
    # units_meta: [(fd, unit_id, valid_len), ...]
    all_main_rows: List[List[float]] = []
    all_aux_rows_by_ch: List[List[List[float]]] = [[] for _ in range(23)]
    units_meta: List[Tuple[int, int, int, str]] = []  # (row_idx, unit_id, valid_len, fd)

    # FD마다 유닛을 순회해 시퀀스 수집
    for fd in fds:
        train_path = os.path.join(args.root, f"train_{fd}.txt")
        if not os.path.isfile(train_path):
            print(f"[ERROR] not found: {train_path}", file=sys.stderr)
            sys.exit(1)

        data = read_train_txt(train_path)
        if not data:
            print(f"[ERROR] no rows parsed from {train_path}", file=sys.stderr)
            sys.exit(1)

        units = sorted(data.keys())
        for u in units:
            series = data[u]  # [(cycle, setting(3), sensors(21))]
            T = len(series)
            if T <= 0:
                continue

            # --- RUL 시퀀스(마지막 RUL=1) & 이진화 ---
            # 인덱스 i=0..T-1에 대해 RUL_i = T - i
            rul_seq = [T - i for i in range(T, 0, -1)]
            y = [1.0 if r <= args.rul_thr else 0.0 for r in rul_seq]
            all_main_rows.append(y)

            # --- aux 23채널 생성 ---
            settings_all = [st for _, st, _ in series]  # [T,3]
            s1_seq = [st[0] for st in settings_all]
            s2_seq = [st[1] for st in settings_all]
            s3_seq = [st[2] for st in settings_all]

            sensors_all = [sens for _, _, sens in series]  # [T,21]
            ch_seqs = [
                s1_seq, s2_seq, s3_seq,
                *([[sens[s_name_to_idx[name]] for sens in sensors_all] for name in sensor_keep])
            ]
            assert len(ch_seqs) == 23

            for ch_i in range(23):
                all_aux_rows_by_ch[ch_i].append(ch_seqs[ch_i])

            row_idx = len(all_main_rows) - 1
            units_meta.append((row_idx, u, T, fd))

    # ---------- 공통 길이 T_max 계산 & 패딩 ----------
    # 기본 정책: align='right' → 왼쪽으로 0패딩(미래 보존)
    T_max = max(len(row) for row in all_main_rows) if all_main_rows else 0
    if T_max <= 0:
        print("[ERROR] empty dataset after parsing.", file=sys.stderr)
        sys.exit(1)

    main_mat = [pad_to_Tmax(row, T_max, args.align) for row in all_main_rows]
    aux_mats = []
    for ch_i in range(23):
        aux_mats.append([pad_to_Tmax(row, T_max, args.align) for row in all_aux_rows_by_ch[ch_i]])

    # ---------- 저장 ----------
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "main.csv"), main_mat)
    for ch_i in range(23):
        write_csv(os.path.join(args.out, f"aux{ch_i+1:02d}.csv"), aux_mats[ch_i])

    with open(os.path.join(args.out, "groups.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","unit_id","valid_len","fd"])
        for row_idx, u, L, fd in units_meta:
            w.writerow([row_idx, u, L, fd])

    print("[OK] saved:")
    print(f" - main.csv       : {len(main_mat)} x {len(main_mat[0])}")
    for ch_i in range(23):
        print(f" - aux{ch_i+1:02d}.csv  : {len(aux_mats[ch_i])} x {len(aux_mats[ch_i][0])}")
    print(f" - groups.csv     : {len(units_meta)} rows")

    # ---------- (선택) 디버그: Δ=+1 라벨 구간 양성비 ----------
    if args.context is not None:
        L = int(args.context)
        T = len(main_mat[0])
        # 라벨 지점은 s+L (Δ=+1), 유효하려면 s+L < T
        # 전체 라벨 수 = N * (T - L) (패딩 포함 단순 계산)
        import numpy as np
        m = np.array(main_mat, dtype=np.float32)  # [N, T]
        if L >= T:
            print(f"[debug] context L={L} ≥ T={T} → Δ=+1 라벨 불가", file=sys.stderr)
        else:
            tail = m[:, L:]  # 라벨 위치가 가리키는 열들
            total = tail.size
            pos = int(np.count_nonzero(tail))
            rate = (pos / total) if total else 0.0
            print(f"[debug] label@Δ=+1 (context=L={L})  pos={pos} / total={total}  rate={rate:.6f}")
            if pos == 0:
                print("[WARN] Δ=+1 라벨 구간이 전부 0입니다. "
                      "align='right'(기본)인지, 전처리/패딩 정책을 재확인하세요.", file=sys.stderr)


if __name__ == "__main__":
    main()
