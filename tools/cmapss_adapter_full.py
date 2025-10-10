#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
C-MAPSS → prjAttention 24채널 레이아웃 어댑터 (main 1 + aux 23)
-------------------------------------------------------------------

📘 기능 개요:
이 스크립트는 NASA C-MAPSS 터보팬 엔진 데이터셋을
AttentionProject에서 사용하는 [N,C,T] 시계열 입력 구조로 변환합니다.

핵심 목적은 **RUL(남은 수명, Remaining Useful Life)** 기반의
이진 분류 타깃을 생성하는 것입니다.

즉, 각 유닛(unit_id)의 주기별 RUL을 계산하여
  y_t = 1  (RUL_t ≤ rul_thr → 고장 임박 상태)
  y_t = 0  (RUL_t > rul_thr → 정상 상태)
로 변환하고, 이를 main.csv로 저장합니다.

따라서 main.csv는 "RUL 기반 이진 타깃"이며,
이는 전체적으로 RUL 예측 문제를 단순화한 **이진 상태 예측 모델**로 사용됩니다.

입력 (root 폴더 예: ./data/CMAPSS):
  - train_FD001.txt, train_FD002.txt, ...
  - test_FD001.txt,  RUL_FD001.txt  (현재 스크립트는 train만 사용)
형식: 공백 구분, 헤더 없음, 26열
  unit, cycle, setting1, setting2, setting3, s1..s21

출력 (out 폴더 예: ./data/cmapss_fd001_full):
  - main.csv             : [N,T]  타깃(0/1), 규칙 y_t = 1 if RUL_t <= rul_thr else 0
  - aux01.csv..aux23.csv : [N,T] 입력 23채널 (setting1~3 + 센서 20개)
  - groups.csv           : row_idx,unit_id,valid_len (메타)

규칙:
  - 한 유닛(unit_id) = 한 행(row), 시간은 열 방향(T)
  - 유닛마다 길이가 다르므로 우측 0-패딩으로 T_max에 맞춤
"""

import argparse, os, sys, csv, math
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="C-MAPSS → prjAttention 24ch adapter")
    p.add_argument("--root", type=str, required=True,
                   help="C-MAPSS txt들이 있는 폴더 (예: ./data/CMAPSS)")
    p.add_argument("--fd", type=str, default="FD001",
                   help="FD 세트명 (FD001|FD002|FD003|FD004)")
    p.add_argument("--rul-thr", type=int, default=30,
                   help="RUL 임계 (≤ thr ⇒ 1)")
    p.add_argument("--out", type=str, required=True,
                   help="출력 폴더 (예: ./data/cmapss_fd001_full)")
    p.add_argument("--exclude-sensors", type=str, default="s1",
                   help="제외할 센서들 (콤마구분, 예: 's1' 또는 's1,s5')")
    return p.parse_args()

def read_train_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: 
                continue
            if len(parts) < 26:
                # 비정상 라인은 스킵
                continue
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
        data[u].sort(key=lambda x: x[0])  # cycle 오름차순
    return data  # unit -> [(cycle, setting(3), sensors(21))...]

def pad_right(rows, T_max):
    out, lens = [], []
    for r in rows:
        L = len(r); lens.append(L)
        if L < T_max:
            out.append(r + [0.0]*(T_max - L))
        else:
            out.append(r[:T_max])
    return out, lens

def write_csv(path, mat):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in mat:
            # 시간은 열 방향 → 그대로 한 행씩 기록
            w.writerow([f"{v:.6f}" for v in row])

def main():
    args = parse_args()

    train_path = os.path.join(args.root, f"train_{args.fd}.txt")
    if not os.path.isfile(train_path):
        print(f"[ERROR] not found: {train_path}", file=sys.stderr)
        sys.exit(1)

    data = read_train_txt(train_path)
    if not data:
        print("[ERROR] no rows parsed from train file.", file=sys.stderr)
        sys.exit(1)

    # ----- 채널 구성 -----
    # 운전조건 3채널은 항상 포함
    setting_names = ["setting1", "setting2", "setting3"]

    # 센서 21개 중 제외 목록을 뺀 20개 사용 (기본: s1 제외)
    exclude = set([s.strip().lower() for s in args.exclude_sensors.split(",") if s.strip()])
    sensor_all = [f"s{i}" for i in range(1, 22)]  # s1..s21
    sensor_keep = [s for s in sensor_all if s.lower() not in exclude]
    if len(sensor_keep) != 20:
        # 사용자가 제외 목록을 바꿔서 20개가 아니면 경고 후 자동 슬라이스
        # (prj 규칙: 총 aux 채널 = 23 = 3 settings + 20 sensors)
        if len(sensor_keep) < 20:
            # 부족하면 알파벳 순으로 남은 것 채움(사실상 불가, 여기선 경고만)
            print(f"[WARN] sensors kept = {len(sensor_keep)}; expected 20. Will pad by reusing last.",
                  file=sys.stderr)
            while len(sensor_keep) < 20:
                sensor_keep.append(sensor_keep[-1])
        else:
            print(f"[WARN] sensors kept = {len(sensor_keep)}; expected 20. Will trim to 20.",
                  file=sys.stderr)
            sensor_keep = sensor_keep[:20]

    # 채널 이름 나열 (디버깅 메시지)
    aux_names = setting_names + sensor_keep  # 총 23
    print(f"[INFO] FD={args.fd}, rul_thr={args.rul_thr}")
    print(f"[INFO] AUX channels (23) = {', '.join(aux_names)}")

    # ----- 길이 조사 (T_max) -----
    units = sorted(data.keys())
    T_max = 0
    for u in units:
        T_max = max(T_max, len(data[u]))

    # ----- 타깃(main) 및 aux 생성 -----
    main_rows = []               # 1ch (target)
    aux_rows_by_ch = [[] for _ in range(23)]  # 23ch

    meta = []  # (row_idx, unit_id, valid_len)

    # 센서 이름 → 인덱스 매핑 (s1..s21 → 0..20)
    s_name_to_idx = {f"s{i}": (i-1) for i in range(1, 22)}

    for row_idx, u in enumerate(units):
        series = data[u]  # [(cycle, setting(3), sensors(21))]
        T = len(series)

        # RUL: 마지막이 1로 가정 → t 시점 RUL_t = T - t (1-based)
        # 여기서는 0-based idx i(0..T-1)에 대해 RUL_i = T - i
        # → 마지막 시점 i=T-1일 때 RUL=1
        rul_seq = [T - i for i in range(T, 0, -1)]
        y = [1.0 if r <= args.rul_thr else 0.0 for r in rul_seq]
        main_rows.append(y)

        # 각 aux 채널 채우기
        # 설정 3채널
        settings_all = [st for _, st, _ in series]  # [T,3]
        s1_seq = [st[0] for st in settings_all]
        s2_seq = [st[1] for st in settings_all]
        s3_seq = [st[2] for st in settings_all]

        # 센서 21개
        sensors_all = [sens for _, _, sens in series]  # [T,21]

        # aux 채널 순서: setting1,2,3, 그다음 sensor_keep 20개
        ch_seqs = [
            s1_seq, s2_seq, s3_seq,
            *([[sens[s_name_to_idx[name]] for sens in sensors_all] for name in sensor_keep])
        ]
        assert len(ch_seqs) == 23

        for ch_i in range(23):
            aux_rows_by_ch[ch_i].append(ch_seqs[ch_i])

        meta.append((row_idx, u, T))

    # ----- 패딩 & 저장 -----
    os.makedirs(args.out, exist_ok=True)

    main_mat, _ = pad_right(main_rows, T_max)
    write_csv(os.path.join(args.out, "main.csv"), main_mat)

    for ch_i in range(23):
        mat, _ = pad_right(aux_rows_by_ch[ch_i], T_max)
        fname = f"aux{ch_i+1:02d}.csv"
        write_csv(os.path.join(args.out, fname), mat)

    with open(os.path.join(args.out, "groups.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","unit_id","valid_len"])
        for row_idx, u, L in meta:
            w.writerow([row_idx, u, L])

    print("[OK] saved:")
    print(f" - main.csv      : {len(main_mat)} x {len(main_mat[0])}")
    for ch_i in range(23):
        path = os.path.join(args.out, f"aux{ch_i+1:02d}.csv")
        print(f" - aux{ch_i+1:02d}.csv : {len(aux_rows_by_ch[ch_i])} x {len(main_mat[0])}")
    print(f" - groups.csv    : {len(meta)} rows")

if __name__ == "__main__":
    main()
