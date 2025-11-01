# -*- coding: utf-8 -*-
"""
data.py — 데이터 로딩/출력 경로/시드 (다변량 CSV 지원)
- parse_csv_single : 단일 CSV → [N,1,T]
- parse_csv_list   : 여러 CSV → [N,C,T] (순서 보존, ch0이 라벨 소스)
- parse_csv_auto   : 단일/다중 자동 분기
- read_csv_no_header: BOM/헤더/구분자 자동 처리(, / \t), 빈 줄·주석 무시
"""
import os
import io
import numpy as np
import torch

# ───────────────────────── common ─────────────────────────

def set_seed(seed: int = 777):
    import torch.backends.cudnn as cudnn
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)

def _normalize_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _loadtxt_from_text(text: str, delimiter: str) -> np.ndarray:
    return np.loadtxt(io.StringIO(text), delimiter=delimiter, dtype=np.float32)

# ─────────────────────── CSV reader (robust) ───────────────────────

def read_csv_no_header(path: str) -> np.ndarray:
    """
    - UTF-8 BOM 허용(utf-8-sig)
    - CR/LF 정규화
    - 빈 줄/주석(#...) 제거
    - 구분자 자동 판별: 콤마 우선, 없으면 탭
    - 첫 줄이 헤더(문자열)인 경우 1행 드롭 후 재시도
    반환: np.ndarray [N, T]
    """
    path = _normalize_path(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    # 줄 정규화 + 주석/빈 줄 제거
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for ln in raw.split("\n"):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    if not lines:
        raise SystemExit(f"[data] empty file: {os.path.basename(path)}")

    # 구분자 추정: 콤마가 있으면 콤마, 아니면 탭 시도
    first = lines[0]
    delimiter = "," if ("," in first) or ("\t" not in first) else "\t"

    text = "\n".join(lines)
    try:
        x = _loadtxt_from_text(text, delimiter=delimiter)
    except ValueError:
        # 첫 줄 헤더 가정 후 1행 드롭
        if len(lines) <= 1:
            raise
        try:
            x = _loadtxt_from_text("\n".join(lines[1:]), delimiter=delimiter)
            print(f"[WARN] dropped header line: {os.path.basename(path)}")
        except ValueError as e:
            # 구분자 반대로 최종 재시도
            alt = "\t" if delimiter == "," else ","
            try:
                x = _loadtxt_from_text("\n".join(lines[1:]), delimiter=alt)
                print(f"[WARN] auto-switched delimiter to {repr(alt)} after header drop: {os.path.basename(path)}")
            except Exception:
                raise SystemExit(f"[data] cannot parse CSV: {os.path.basename(path)} ({e})")

    if x.ndim == 1:
        x = x[None, :]  # [N,T]
    if x.ndim != 2:
        raise SystemExit(f"[data] CSV must resolve to 2D [N,T], got {tuple(x.shape)} at {os.path.basename(path)}")
    return x

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

# ─────────────────────── parse (single/list/auto) ───────────────────────

def parse_csv_single(args) -> torch.Tensor:
    """
    단일 CSV: [N,T] → [N,1,T]
    """
    if not args.csv:
        raise SystemExit("CSV 입력 필요: --csv <path>")
    path = args.csv[0] if isinstance(args.csv, list) else args.csv
    path = _normalize_path(path)

    M = to_tensor(read_csv_no_header(path))  # [N,T]
    if M.ndim != 2:
        raise SystemExit(f"--csv는 2차원 [N,T] 형태여야 합니다. got shape={tuple(M.shape)}")
    X = M.unsqueeze(1)  # [N,1,T]

    # 메타 주입
    setattr(args, "_in_channels", 1)
    setattr(args, "_label_csv", path)
    print(f"[INFO] parse_csv_single: {path} -> {tuple(X.shape)} (label from first CSV = ch0)")
    return X

def parse_csv_list(args) -> torch.Tensor:
    """
    여러 CSV: 각각 [N,T] 를 채널 축(C)으로 스택 → [N,C,T]
    모든 CSV의 [N,T]가 일치해야 함. 첫 CSV가 라벨 소스(ch0).
    """
    paths = list(args.csv)
    if len(paths) < 2:
        raise SystemExit("[data] parse_csv_list requires >=2 CSVs")

    paths = [_normalize_path(p) for p in paths]
    mats = [to_tensor(read_csv_no_header(p)) for p in paths]  # 각 [N,T]
    N0, T0 = mats[0].shape
    for p, m in zip(paths, mats):
        if tuple(m.shape) != (N0, T0):
            raise SystemExit(f"[data] shape mismatch at '{p}': expected {(N0,T0)}, got {tuple(m.shape)}")

    X = torch.stack(mats, dim=1)  # [N,C,T]

    # 메타 주입: 채널수/라벨 소스(항상 첫 CSV)
    setattr(args, "_in_channels", len(paths))
    setattr(args, "_label_csv", paths[0])

    print(f"[INFO] parse_csv_list: {len(paths)} files -> {tuple(X.shape)}")
    print(f"[INFO] label source (fixed): ch0 ← {paths[0]}")
    return X

def parse_csv_auto(args) -> torch.Tensor:
    """
    단일/다중 자동 분기
    """
    if not isinstance(args.csv, list):
        args.csv = [args.csv]
    # 경로 정규화(로그/메타 일관성)
    args.csv = [_normalize_path(p) for p in args.csv]
    return parse_csv_single(args) if len(args.csv) == 1 else parse_csv_list(args)

# ─────────────────────── run dir ───────────────────────

def make_run_dir(args, base: str) -> str:
    """
    out_dir을 확정하고 폴더 생성. plots/도 함께 만든다.
    - args.out 지정 시 그대로 사용
    - 그 외에는 규칙 기반 네이밍(원래 규칙 유지)
    """
    if args.out:
        run_name = args.out
    else:
        ch = getattr(args, "_in_channels", 1)
        run_name = (
            f"{args.mode}_{args.task}_ctx{args.context_len}_ch{ch}_"
            f"alpha{args.alpha:.2f}_pw{args.pos_weight}_{args.backbone}_"
            f"seed{args.seed}_split{args.split_mode}"
        )
    base = _normalize_path(base)
    out_dir = os.path.join(base, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    return out_dir
