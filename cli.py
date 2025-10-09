# -*- coding: utf-8 -*-
"""
cli.py — 명령행 인터페이스 (다변량 CSV 지원)
- 변경점:
  • --csv 를 단일/다중 경로 모두 수용(nargs='+')
  • 나머지 인자/기본값/흐름은 동일
[의존] pipeline.main_run
[제공] main()
"""
import argparse, torch
from pipeline import main_run

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # 입력
    ap.add_argument(
        "--csv", type=str, nargs="+", required=True,
        help="하나 이상의 CSV 경로 (헤더/인덱스 없음, [N,T]). 여러 개 주면 다변량(C채널)로 처리"
    )

    # 모드/작업/모델/학습/출력
    ap.add_argument("--mode", type=str, required=True, choices=["train","finetune","infer"])
    ap.add_argument("--task", type=str, default="regress", choices=["regress","classify"])
    ap.add_argument("--backbone", type=str, default="llm_ts")
    ap.add_argument("--context-len", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--plot-samples", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=50)

    # 분류 전용
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=str, default="global", choices=["global","batch","none"])
    ap.add_argument("--thresh-default", type=float, default=0.5)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--bin-rule", type=str, default="nonzero", choices=["nonzero","gt","ge"])
    ap.add_argument("--bin-thr", type=float, default=0.0)

    # 리소스/성능
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", type=str, default="", help='torch.compile mode: "", "reduce-overhead", "max-autotune"')
    ap.add_argument("--eval-batch-size", type=int, default=4096)

    # split
    ap.add_argument("--split-mode", type=str, default="group", choices=["group","item","time","window"])

    # 체크포인트
    ap.add_argument("--ckpt", type=str, default=None)
    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    # 항상 list 로 통일
    if isinstance(args.csv, str):
        args.csv = [args.csv]
    main_run(args)
