# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: cli.py                                                              ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE                                                                   ║
║   CLI 파서 및 엔트리 함수. 기존 run_llmts.py의 인자와 기본값을 동일 유지.    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   build_parser() -> argparse.ArgumentParser                               ║
║   main() -> None                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ INPUT/OUTPUT                                                              ║
║   IN : sys.argv                                                           ║
║   OUT: 없음(파이프라인 호출)                                              ║
╠───────────────────────────────────────────────────────────────────────────╣
║ SIDE EFFECTS                                                              ║
║   - 없음(파이프라인에서 파일 생성)                                        ║
╠───────────────────────────────────────────────────────────────────────────╣
║ EXCEPTIONS                                                                ║
║   - argparse 가 잘못된 인자에 대해 SystemExit                            ║
╠───────────────────────────────────────────────────────────────────────────╣
║ DEPENDENCY GRAPH                                                          ║
║   cli → pipeline.main_run                                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

cli.py — 명령행 인터페이스
[의존] pipeline.main_run
[제공] main()
"""
import argparse, torch
from pipeline import main_run

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # 입력
    ap.add_argument("--csv", type=str, required=True, help="단일 CSV 경로 (헤더/인덱스 없음, [N,T])")

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
    main_run(args)
