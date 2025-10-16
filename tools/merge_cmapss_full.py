#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple C-MAPSS *full* folders into a single combined dataset, with
explicit padding alignment to avoid destroying the *future* columns
(critical when labels are at Δ=+1: label = main[:, s+L]).

Input (each folder made by cmapss_adapter_full.py):
  <in>/main.csv            [N_i, T_i]
  <in>/aux01..aux23.csv    [N_i, T_i]
  <in>/groups.csv          row_idx,unit_id,valid_len(,fd optional)

Output:
  <out>/main.csv           [sum N_i, T_max_all]
  <out>/aux01..aux23.csv   [sum N_i, T_max_all]
  <out>/groups.csv         row_idx,unit_id,valid_len
  <out>/sources.csv        row_idx,source,orig_row_idx,orig_unit_id,orig_valid_len

Key options:
  --align right|left
     right (default)  : LEFT padding → future preserved (recommended)
     left             : RIGHT padding → future may be zero-filled (risky for Δ=+1)

  --context L
     Print immediate debug on Δ=+1 label region (main[:, L:]) positive rate.

Usage:
  python merge_cmapss_full.py \
    --inputs ./data/cmapss_fd001_full ./data/cmapss_fd002_full \
    --labels FD001,FD002 \
    --out ./data/cmapss_all_full \
    --align right \
    --context 31
"""

import argparse, os, sys, csv
from typing import List, Tuple, Dict

NUM_AUX = 23

# ---------------------------
# CSV utils
# ---------------------------
def read_matrix(path: str) -> List[List[float]]:
    mat: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            mat.append([float(x) for x in row])
    return mat  # [N, T]

def write_matrix(path: str, mat: List[List[float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in mat:
            w.writerow([f"{v:.6f}" for v in row])

def read_groups(path: str) -> List[Tuple[int,int,int]]:
    """
    Returns list of (row_idx, unit_id, valid_len).
    Accepts optional headers and float-ish valid_len values.
    """
    out: List[Tuple[int,int,int]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        # header tolerant: if first row non-numeric, it was a header
        def _is_int(s):
            try: int(s); return True
            except: return False
        if header and (len(header) < 3 or not _is_int(header[0])):
            pass  # treated as header
        else:
            # header was actually data
            if header:
                try:
                    row_idx = int(header[0]); unit_id = int(header[1]); valid_len = int(float(header[2]))
                    out.append((row_idx, unit_id, valid_len))
                except Exception:
                    pass
        for i, row in enumerate(r):
            if not row:
                continue
            try:
                row_idx = int(row[0]); unit_id = int(row[1]); valid_len = int(float(row[2]))
            except Exception:
                # Fallback: try minimal columns
                try:
                    unit_id = int(row[1]); valid_len = int(float(row[2])); row_idx = i
                except Exception:
                    raise ValueError(f"Invalid groups row: {row}")
            out.append((row_idx, unit_id, valid_len))
    return out

def write_groups(path: str, rows: List[Tuple[int,int,int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","unit_id","valid_len"])
        for r in rows:
            w.writerow(r)

def write_sources(path: str, rows: List[Tuple[int,str,int,int,int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","source","orig_row_idx","orig_unit_id","orig_valid_len"])
        for r in rows:
            w.writerow(r)

# ---------------------------
# Padding
# ---------------------------
def pad_row(row: List[float], T_target: int, align: str) -> List[float]:
    """
    align='right' : LEFT pad with zeros (future preserved)  [RECOMMENDED]
    align='left'  : RIGHT pad with zeros (future risk)
    """
    t = len(row)
    if t == T_target:
        return row[:]
    pad_len = T_target - t
    if pad_len < 0:
        # safety: truncate on the RIGHT only (do not cut the future if align=right)
        return row[:T_target]
    if align == "right":
        return [0.0]*pad_len + row
    else:
        return row + [0.0]*pad_len

def pad_matrix(mat: List[List[float]], T_target: int, align: str) -> List[List[float]]:
    return [pad_row(row, T_target, align) for row in mat]

# ---------------------------
# Label offset offsets
# ---------------------------
def infer_offset_from_name(name: str, used_offsets: set) -> int:
    name = name.upper()
    fixed = {"FD001":100000, "FD002":200000, "FD003":300000, "FD004":400000}
    if name in fixed:
        return fixed[name]
    base = 100000
    k = 1
    while base*k in used_offsets:
        k += 1
    return base * k

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Merge C-MAPSS full folders into a single dataset (future-preserving merge)."
    )
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Input folders: each has main.csv, aux01..aux23.csv, groups.csv")
    p.add_argument("--labels", type=str, default=None,
                   help="Optional labels for inputs (comma-separated, same order). "
                        "e.g., FD001,FD002")
    p.add_argument("--out", required=True, help="Output folder")
    p.add_argument("--align", choices=["right","left"], default="right",
                   help="Padding alignment to reach global T_max: "
                        "'right' = LEFT padding (future preserved, default), "
                        "'left' = RIGHT padding (future may be zero-filled)")
    p.add_argument("--context", type=int, default=None,
                   help="If set, print Δ=+1 label region stats for sanity check.")
    return p.parse_args()

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    inputs = [d.rstrip("/\\") for d in args.inputs]
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    if labels and len(labels) != len(inputs):
        print("[ERROR] --labels count must match --inputs", file=sys.stderr)
        sys.exit(1)

    # Step 1: Read all inputs and collect shapes
    data_by_label: Dict[str, Dict[str, List[List[float]]]] = {}
    groups_by_label: Dict[str, List[Tuple[int,int,int]]] = {}
    T_max_all = 0
    used_offsets = set()
    offset_by_label: Dict[str, int] = {}

    for idx, in_dir in enumerate(inputs):
        label = labels[idx] if labels else os.path.basename(in_dir)
        main_p = os.path.join(in_dir, "main.csv")
        grp_p  = os.path.join(in_dir, "groups.csv")
        if not os.path.isfile(main_p) or not os.path.isfile(grp_p):
            print(f"[ERROR] missing main.csv or groups.csv in {in_dir}", file=sys.stderr)
            sys.exit(1)

        # read main & aux
        mats: Dict[str, List[List[float]]] = {}
        mats["main"] = read_matrix(main_p)
        N_i = len(mats["main"])
        if N_i == 0:
            print(f"[ERROR] empty main matrix in {in_dir}", file=sys.stderr); sys.exit(1)
        T_i = len(mats["main"][0])
        for ch in range(NUM_AUX):
            ap = os.path.join(in_dir, f"aux{ch+1:02d}.csv")
            if not os.path.isfile(ap):
                print(f"[ERROR] missing {ap}", file=sys.stderr); sys.exit(1)
            mat = read_matrix(ap)
            if len(mat) != N_i:
                print(f"[ERROR] N mismatch in {in_dir}: aux{ch+1:02d} N={len(mat)} vs main N={N_i}",
                      file=sys.stderr); sys.exit(1)
            if len(mat[0]) != T_i:
                print(f"[ERROR] T mismatch in {in_dir}: aux{ch+1:02d} T={len(mat[0])} vs main T={T_i}",
                      file=sys.stderr); sys.exit(1)
            mats[f"aux{ch+1:02d}"] = mat

        g = read_groups(grp_p)
        if len(g) != N_i:
            print(f"[ERROR] groups rows({len(g)}) != main rows({N_i}) in {in_dir}",
                  file=sys.stderr); sys.exit(1)

        data_by_label[label] = mats
        groups_by_label[label] = g
        T_max_all = max(T_max_all, T_i)

        # offsets
        off = infer_offset_from_name(label, used_offsets)
        used_offsets.add(off)
        offset_by_label[label] = off

    # Step 2: Pad to global T_max_all WITH alignment
    out_main: List[List[float]] = []
    out_aux : List[List[List[float]]] = [[] for _ in range(NUM_AUX)]
    out_groups: List[Tuple[int,int,int]] = []
    out_sources: List[Tuple[int,str,int,int,int]] = []

    row_cursor = 0
    for label in (labels if labels else inputs):
        lab = label if labels else os.path.basename(label)
        mats = data_by_label[lab]
        g    = groups_by_label[lab]
        off  = offset_by_label[lab]

        main_padded = pad_matrix(mats["main"], T_max_all, args.align)
        out_main.extend(main_padded)
        for ch in range(NUM_AUX):
            mat = mats[f"aux{ch+1:02d}"]
            out_aux[ch].extend(pad_matrix(mat, T_max_all, args.align))

        # groups/sources with new row_idx
        for local_idx, (orig_row_idx, orig_unit, orig_valid) in enumerate(g):
            new_row_idx = row_cursor + local_idx
            new_unit = off + orig_unit
            out_groups.append((new_row_idx, new_unit, orig_valid))
            out_sources.append((new_row_idx, lab, orig_row_idx, orig_unit, orig_valid))

        row_cursor += len(g)

    # Step 3: Write out
    os.makedirs(args.out, exist_ok=True)
    write_matrix(os.path.join(args.out, "main.csv"), out_main)
    for ch in range(NUM_AUX):
        write_matrix(os.path.join(args.out, f"aux{ch+1:02d}.csv"), out_aux[ch])
    write_groups(os.path.join(args.out, "groups.csv"), out_groups)
    write_sources(os.path.join(args.out, "sources.csv"), out_sources)

    N_total = len(out_main)
    T_out   = len(out_main[0]) if out_main else 0
    print(f"[OK] merged to {args.out}")
    print(f" - N_total   = {N_total}")
    print(f" - T_max_all = {T_out}")
    print(f" - files: main.csv, aux01..aux23.csv, groups.csv, sources.csv")
    print(f" - align={args.align} (right=LEFT pad / left=RIGHT pad)")

    # Step 4 (optional): sanity on Δ=+1 label region
    if args.context is not None and N_total > 0 and T_out > 0:
        try:
            import numpy as np
            m = np.array(out_main, dtype=np.float32)  # [N, T]
            L = int(args.context)
            if L >= T_out:
                print(f"[debug] context L={L} ≥ T={T_out} → Δ=+1 labels impossible", file=sys.stderr)
            else:
                tail = m[:, L:]
                total = tail.size
                pos = int((tail != 0).sum())
                rate = (pos / total) if total else 0.0
                print(f"[debug] label@Δ=+1 (context=L={L})  pos={pos} / total={total}  rate={rate:.6f}")
                if pos == 0 and args.align == "left":
                    print("[WARN] Δ=+1 label region is all zeros. "
                          "Try --align right (LEFT pad) to preserve future.", file=sys.stderr)
        except Exception as e:
            print(f"[debug] tail check failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
