# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: data.py                                                             ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE                                                                   ║
║   CSV 로딩/텐서 변환/시드 고정/출력 경로 준비. 단일 CSV(헤더/인덱스 없음)    ║
║   [N,T] → [N,1,T] (단일 채널 고정).                                       ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   set_seed(seed:int) -> None                                              ║
║   parse_csv_single(args) -> torch.Tensor [N,1,T]                          ║
║   make_run_dir(args, base:str) -> str                                     ║
╠───────────────────────────────────────────────────────────────────────────╣
║ INPUT/OUTPUT                                                              ║
║   IN : 파일 경로(args.csv)                                                ║
║   OUT: 텐서, 아티팩트 디렉터리 생성                                        ║
╠───────────────────────────────────────────────────────────────────────────╣
║ SIDE EFFECTS  폴더 생성, CUDA deterministic 설정                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ EXCEPTIONS   CSV 미지정/형태 오류 → SystemExit                            ║
╠───────────────────────────────────────────────────────────────────────────╣
║ DEPENDENCY GRAPH  pipeline → data                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

data.py — 데이터 로딩/출력 경로/시드
"""
import os, numpy as np, torch

def set_seed(seed:int=777):
    import torch.backends.cudnn as cudnn
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False

def ensure_outdir(p:str): os.makedirs(p, exist_ok=True)

def read_csv_no_header(path:str) -> np.ndarray:
    with open(path, "r", encoding="utf-8-sig") as f:
        x = np.loadtxt(f, delimiter=",", dtype=np.float32)
    return x[None, :] if x.ndim == 1 else x

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(dtype=torch.float32).contiguous()

def parse_csv_single(args) -> torch.Tensor:
    """
    입력 CSV: [N,T] → 반환 텐서: [N,1,T]
    예외: 경로 누락/차원 불일치 시 SystemExit
    """
    if not args.csv:
        raise SystemExit("CSV 입력 필요: --csv <path_to_single_csv>")
    M = to_tensor(read_csv_no_header(args.csv))  # [N,T]
    if M.ndim != 2:
        raise SystemExit(f"--csv는 2차원 [N,T] 형태여야 합니다. got shape={tuple(M.shape)}")
    return M.unsqueeze(1)  # [N,1,T]

def make_run_dir(args, base:str) -> str:
    if args.out:
        run_name = args.out
    else:
        slug = f"{args.mode}_{args.task}_ctx{args.context_len}_alpha{args.alpha:.2f}_pw{args.pos_weight}_{args.backbone}_seed{args.seed}_split{args.split_mode}"
        run_name = slug
    out_dir = os.path.join(base, run_name)
    ensure_outdir(out_dir); ensure_outdir(os.path.join(out_dir, "plots"))
    return out_dir
