# -*- coding: utf-8 -*-
"""
cli.py — 명령행 인터페이스 (다변량 CSV 지원)
- --csv 단일/다중 경로 수용(nargs='+')
- 평가/플롯 스플릿 제어 (--eval-split, --plot-split)
- (선택) 라벨 채널 입력 제외 옵션 (--drop-label-from-x, --label-src-ch)
[의존] pipeline.main_run
"""

import argparse
import torch
from pipeline import main_run


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # 입력
    ap.add_argument(
        "--csv", type=str, nargs="+", required=True,
        help="하나 이상의 CSV 경로 ([N,T]). 여러 개면 채널 축으로 스택"
    )

    # 모드/작업/모델/학습/출력
    ap.add_argument("--mode", type=str, required=True, choices=["train", "finetune", "infer"])
    ap.add_argument("--task", type=str, default="regress", choices=["regress", "classify"])
    ap.add_argument("--backbone", type=str, default="llm_ts")
    ap.add_argument("--context-len", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--plot-samples", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=50)

    # 분류 전용
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=str, default="global", choices=["global", "batch", "none"])
    ap.add_argument("--thresh-default", type=float, default=0.5)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--bin-rule", type=str, default="nonzero", choices=["nonzero", "gt", "ge"])
    ap.add_argument("--bin-thr", type=float, default=0.0)

    # 리소스/성능
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", type=str, default="", help='torch.compile mode: "", "reduce-overhead", "max-autotune"')
    ap.add_argument("--eval-batch-size", type=int, default=4096)

    # split
    ap.add_argument("--split-mode", type=str, default="group", choices=["group", "item", "time", "window"])

    # 체크포인트
    ap.add_argument("--ckpt", type=str, default=None)

    # 평가/플롯 스플릿
    ap.add_argument("--eval-split", type=str, default="val", choices=["val", "train", "all"],
                    help="평가(eval_model) 시 사용할 스플릿 (기본: val)")
    ap.add_argument("--plot-split", type=str, default="val", choices=["val", "train", "all"],
                    help="플롯(plot_samples/plot_cls_curves) 시 사용할 스플릿 (기본: val)")
    ap.add_argument("--no-plots", action="store_true", help="플롯 생성 비활성화")

    # (선택) 라벨 채널을 입력에서 제외
    ap.add_argument("--drop-label-from-x", action="store_true",
                    help="켜면 label_src_ch 채널을 입력 X에서 제외하고 라벨로만 사용")
    ap.add_argument("--label-src-ch", type=int, default=0,
                    help="라벨 소스 채널 인덱스 (기본 0; 보통 ch0=rul.csv)")
    ap.add_argument("--outdir", default=None, help="artifacts/<outdir> 폴더명을 직접 지정")


    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if isinstance(args.csv, str):
        args.csv = [args.csv]
    main_run(args)


if __name__ == "__main__":
    main()
