#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple C-MAPSS full folders (each with main.csv + aux01..aux23.csv + groups.csv)
into a single combined dataset folder, padding to a global T_max.

Input folder spec (made by cmapss_adapter_full.py):
  <in>/main.csv           [N_i, T_i]
  <in>/aux01.csv..aux23.csv  [N_i, T_i]
  <in>/groups.csv         (row_idx,unit_id,valid_len)

Output:
  <out>/main.csv          [sum N_i, T_max_all]
  <out>/aux01..aux23.csv  [sum N_i, T_max_all]
  <out>/groups.csv        (row_idx,unit_id,valid_len)
  <out>/sources.csv       (row_idx,source,orig_row_idx,orig_unit_id,orig_valid_len)

Notes:
- Each input can have different T_i (max cycle). We right-pad to global T_max_all.
- unit_id collisions across sets are avoided by prefixing with a source-based offset:
    FD001 -> +100000, FD002 -> +200000, FD003 -> +300000, FD004 -> +400000
  (If a source is unknown, offset increments by 100000 in order.)
"""

import argparse, os, sys, csv

NUM_AUX = 23

def read_matrix(path):
    mat = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row: continue
            mat.append([float(x) for x in row])
    return mat  # [N, T]

def pad_rows(mat, T_target):
    out = []
    for row in mat:
        if len(row) < T_target:
            out.append(row + [0.0]*(T_target - len(row)))
        else:
            out.append(row[:T_target])
    return out

def write_matrix(path, mat):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in mat:
            w.writerow([f"{v:.6f}" for v in row])

def read_groups(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for i, row in enumerate(r):
            if not row: continue
            # expected: row_idx,unit_id,valid_len
            try:
                row_idx = int(row[0]); unit_id = int(row[1]); valid_len = int(float(row[2]))
            except Exception:
                # fallback: try parse minimal
                unit_id = int(row[1]); valid_len = int(float(row[2])); row_idx = i
            out.append((row_idx, unit_id, valid_len))
    return out

def write_groups(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","unit_id","valid_len"])
        for r in rows:
            w.writerow(r)

def write_sources(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","source","orig_row_idx","orig_unit_id","orig_valid_len"])
        for r in rows:
            w.writerow(r)

def infer_offset_from_name(name, used_offsets):
    name = name.upper()
    map_fixed = {"FD001":100000, "FD002":200000, "FD003":300000, "FD004":400000}
    if name in map_fixed:
        return map_fixed[name]
    # else assign next free 100000 step
    base = 100000
    k = 1
    while base*k in used_offsets:
        k += 1
    return base*k

def parse_args():
    p = argparse.ArgumentParser(description="Merge C-MAPSS full folders into one")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Input folders, each has main.csv, aux01..aux23.csv, groups.csv")
    p.add_argument("--labels", type=str, default=None,
                   help="Optional source labels same order as --inputs (comma-separated, e.g., FD001,FD002)")
    p.add_argument("--out", required=True, help="Output folder")
    return p.parse_args()

def main():
    args = parse_args()
    inputs = args.inputs
    labels = args.labels.split(",") if args.labels else None
    if labels and len(labels) != len(inputs):
        print("[ERROR] --labels count must match --inputs", file=sys.stderr); sys.exit(1)

    # 1) read shapes & find global T_max
    mats_main = []
    mats_aux  = [[] for _ in range(NUM_AUX)]
    groups_all = []
    sources_rows = []

    T_max_all = 0
    N_total = 0
    used_offsets = set()
    combined_groups = []

    for idx, in_dir in enumerate(inputs):
        label = labels[idx].strip() if labels else os.path.basename(in_dir.rstrip("/"))
        main_p = os.path.join(in_dir, "main.csv")
        grp_p  = os.path.join(in_dir, "groups.csv")
        if not os.path.isfile(main_p) or not os.path.isfile(grp_p):
            print(f"[ERROR] missing files in {in_dir}", file=sys.stderr); sys.exit(1)

        m_main = read_matrix(main_p)
        aux_list = []
        for i in range(NUM_AUX):
            ap = os.path.join(in_dir, f"aux{i+1:02d}.csv")
            if not os.path.isfile(ap):
                print(f"[ERROR] missing {ap}", file=sys.stderr); sys.exit(1)
            aux_list.append(read_matrix(ap))

        # shape check
        N_i = len(m_main)
        T_i = len(m_main[0]) if N_i>0 else 0
        for a in aux_list:
            if len(a) != N_i:
                print(f"[ERROR] N mismatch in {in_dir}", file=sys.stderr); sys.exit(1)
            if len(a)>0 and len(a[0]) != T_i:
                print(f"[ERROR] T mismatch in {in_dir}", file=sys.stderr); sys.exit(1)

        T_max_all = max(T_max_all, T_i)

        # groups & unit offsets
        g = read_groups(grp_p)
        if len(g) != N_i:
            # groups rows should match number of rows
            pass

        offset = infer_offset_from_name(label, used_offsets)
        used_offsets.add(offset)

        # store temporarily with metadata
        mats_main.append((label, offset, m_main))
        for ch in range(NUM_AUX):
            mats_aux[ch].append((label, offset, aux_list[ch]))
        groups_all.append((label, offset, g))

    # 2) pad to global T and concat
    out_main = []
    out_aux  = [[] for _ in range(NUM_AUX)]
    out_groups = []
    out_sources = []
    row_cursor = 0

    for idx, in_dir in enumerate(inputs):
        label = labels[idx].strip() if labels else os.path.basename(in_dir.rstrip("/"))

        # fetch
        offset = None
        m_main = None
        aux_list_by_label = [None]*NUM_AUX
        g = None
        for lab, off, mm in mats_main:
            if lab == label:
                offset = off; m_main = mm; break
        for ch in range(NUM_AUX):
            for lab, off, ma in mats_aux[ch]:
                if lab == label:
                    aux_list_by_label[ch] = ma; break
        for lab, off, gg in groups_all:
            if lab == label:
                g = gg; break

        # pad
        m_main_p = pad_rows(m_main, T_max_all)
        out_main.extend(m_main_p)
        for ch in range(NUM_AUX):
            out_aux[ch].extend(pad_rows(aux_list_by_label[ch], T_max_all))

        # groups & sources
        for local_idx, (orig_row_idx, orig_unit, orig_valid) in enumerate(g):
            new_row_idx = row_cursor + local_idx
            new_unit = offset + orig_unit
            out_groups.append((new_row_idx, new_unit, orig_valid))
            out_sources.append((new_row_idx, label, orig_row_idx, orig_unit, orig_valid))

        row_cursor += len(g)

    # 3) write out
    os.makedirs(args.out, exist_ok=True)
    write_matrix(os.path.join(args.out, "main.csv"), out_main)
    for ch in range(NUM_AUX):
        write_matrix(os.path.join(args.out, f"aux{ch+1:02d}.csv"), out_aux[ch])
    write_groups(os.path.join(args.out, "groups.csv"), out_groups)
    write_sources(os.path.join(args.out, "sources.csv"), out_sources)

    print(f"[OK] merged to {args.out}")
    print(f" - N_total = {len(out_main)}")
    print(f" - T_max_all = {len(out_main[0]) if out_main else 0}")
    print(f" - files: main.csv, aux01..aux23.csv, groups.csv, sources.csv")

if __name__ == "__main__":
    main()
