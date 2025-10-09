# -*- coding: utf-8 -*-
"""
data.py — 데이터 로딩/출력 경로/시드 (다변량 CSV 지원)
- 변경점:
  • parse_csv_single: 단일 CSV → [N,1,T] (기존 유지)
  • parse_csv_list  : 여러 CSV → [N,C,T] (신규)
  • parse_csv_auto  : 단일/다중 자동 분기 (신규)
  • read_csv_no_header: BOM/헤더 1행 자동 처리
"""
import os, io, numpy as np, torch

def set_seed(seed:int=777):
    import torch.backends.cudnn as cudnn
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False

def ensure_outdir(p:str): os.makedirs(p, exist_ok=True)

def _loadtxt_from_text(text: str) -> np.ndarray:
    return np.loadtxt(io.StringIO(text), delimiter=",", dtype=np.float32)

def read_csv_no_header(path:str) -> np.ndarray:
    # UTF-8 BOM 제거를 위해 utf-8-sig 로 읽음
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    try:
        x = _loadtxt_from_text(raw)
    except ValueError:
        # 첫 줄에 문자열 헤더가 있을 수 있으니 1행 삭제 후 재시도
        lines = [ln for ln in raw.split("\n") if ln.strip() != ""]
        if len(lines) <= 1:
            raise
        x = _loadtxt_from_text("\n".join(lines[1:]))
        print(f"[WARN] dropped header line: {os.path.basename(path)}")
    return x[None, :] if x.ndim == 1 else x  # [N,T]

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(dtype=torch.float32).contiguous()

def parse_csv_single(args) -> torch.Tensor:
    """
    단일 CSV: [N,T] → [N,1,T]
    """
    if not args.csv:
        raise SystemExit("CSV 입력 필요: --csv <path>")
    path = args.csv[0] if isinstance(args.csv, list) else args.csv
    M = to_tensor(read_csv_no_header(path))  # [N,T]
    if M.ndim != 2:
        raise SystemExit(f"--csv는 2차원 [N,T] 형태여야 합니다. got shape={tuple(M.shape)}")
    X = M.unsqueeze(1)  # [N,1,T]
    print(f"[INFO] parse_csv_single: {path} -> {tuple(X.shape)}")
    return X

def parse_csv_list(args) -> torch.Tensor:
    """
    여러 CSV: 각각 [N,T] 를 채널 축(C)으로 스택 → [N,C,T]
    모든 CSV의 [N,T]가 일치해야 함
    """
    paths = list(args.csv)
    mats = [to_tensor(read_csv_no_header(p)) for p in paths]  # 각 [N,T]
    N0, T0 = mats[0].shape
    for p, m in zip(paths, mats):
        if tuple(m.shape) != (N0, T0):
            raise SystemExit(f"[data] shape mismatch at '{p}': expected {(N0,T0)}, got {tuple(m.shape)}")
    X = torch.stack(mats, dim=1)  # [N,C,T]
    print(f"[INFO] parse_csv_list: {len(paths)} files -> {tuple(X.shape)}")
    return X

def parse_csv_auto(args) -> torch.Tensor:
    """
    단일/다중 자동 분기
    """
    if not isinstance(args.csv, list):
        args.csv = [args.csv]
    return parse_csv_single(args) if len(args.csv) == 1 else parse_csv_list(args)

def make_run_dir(args, base:str) -> str:
    if args.out:
        run_name = args.out
    else:
        # in_channels는 pipeline에서 세팅한 args._in_channels(없으면 1)
        ch = getattr(args, "_in_channels", 1)
        slug = f"{args.mode}_{args.task}_ctx{args.context_len}_ch{ch}_alpha{args.alpha:.2f}_pw{args.pos_weight}_{args.backbone}_seed{args.seed}_split{args.split_mode}"
        run_name = slug
    out_dir = os.path.join(base, run_name)
    ensure_outdir(out_dir); ensure_outdir(os.path.join(out_dir, "plots"))
    return out_dir
