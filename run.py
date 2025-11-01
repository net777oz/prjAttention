#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, logging, os, random, sys, json
from pathlib import Path
from datetime import datetime

def set_all_seeds(seed: int = 42, deterministic: bool = True):
    import numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _mk_artifact_dir(root: str, auto_name: str | None, outdir: str | None, exist_ok: bool) -> Path:
    root_p = Path(root); root_p.mkdir(parents=True, exist_ok=True)
    if outdir:
        p = root_p / outdir
        if not exist_ok and p.exists():
            base = outdir; k = 1
            while (root_p / f"{base}_{k}").exists():
                k += 1
            p = root_p / f"{base}_{k}"
        p.mkdir(parents=True, exist_ok=True)
        return p
    assert auto_name is not None
    p = root_p / auto_name
    p.mkdir(parents=True, exist_ok=True)
    return p

def _auto_run_name(task: str, dataset_tag: str, desc: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_desc = (desc or "").strip().replace(" ", "_")
    return f"{task}_{dataset_tag}_{ts}" + (f"__{safe_desc}" if safe_desc else "")

def _setup_logging(save_dir: Path, level_name: str = "INFO"):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(save_dir / "logs" / "run.log", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

# ---- 백본 레지스트리 ---------------------------------------------------------
# ap/ 패키지로 분리 (파일은 아래에 제공). 여기선 사용만 합니다.
def _build_backbone(name: str, **kwargs):
    from ap.backbones import build_backbone
    return build_backbone(name, **kwargs)

def _register_default_backbones():
    from ap.backbones import register
    # 기본 LLM-TS 어댑터와 예시 LSTM 등록
    register("llm_ts", "backbones.llm_ts_adapter", "build_model")
    register("lstm",   "backbones.lstm",          "build_model")

# ---- CLI ---------------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(description="AttentionProject — unified CLI")
    # README의 옵션 집합을 그대로 유지
    p.add_argument("--mode", required=True, choices=["train","finetune","infer","eval"])
    p.add_argument("--task", required=True, choices=["classify","regress"])
    p.add_argument("--csv", nargs="+")  # README는 여러 파일 나열/브레이스 확장 사용
    p.add_argument("--label-src-ch", type=int, default=0)
    p.add_argument("--drop-label-from-x", action="store_true")
    p.add_argument("--context-len", type=int, default=31)
    p.add_argument("--label-offset", type=int, default=1)
    p.add_argument("--bin-rule", default="nonzero", choices=["nonzero","gt","lt","quantile"])
    p.add_argument("--bin-thr", type=float, default=0.0)
    p.add_argument("--thresh-default", type=float, default=0.5)
    p.add_argument("--split-mode", default="group", choices=["group","rolling","holdout"])
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--eval-split", default="val", choices=["train","val","test"])
    p.add_argument("--plot-split", default="val", choices=["train","val","test"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--pos-weight", default="global")  # "global" 또는 float 문자열
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", default="off", choices=["off","reduce-overhead"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--backbone", default="llm_ts")  # 기본은 llm_ts (README와 동일)
    p.add_argument("--desc", default="")
    # 새 옵션 ①: 아티팩트 outdir 직접 지정
    p.add_argument("--outdir", default=None, help="artifacts/<outdir>로 저장 (자동 네이밍 무시)")
    p.add_argument("--exist-ok", action="store_true", help="--outdir가 존재해도 사용")
    # 자동 네이밍에 쓰일 태그
    p.add_argument("--dataset-tag", default="run")
    # 아티팩트 루트
    p.add_argument("--artifacts-root", default="artifacts")
    return p

def main():
    args = _build_parser().parse_args()
    _register_default_backbones()
    set_all_seeds(args.seed, deterministic=True)

    auto_name = None
    if not args.outdir:
        auto_name = _auto_run_name(args.task, args.dataset_tag, args.desc)
    save_dir = _mk_artifact_dir(args.artifacts_root, auto_name, args.outdir, args.exist_ok)
    log = _setup_logging(save_dir, "INFO")
    log.info(f"[AP] Artifacts → {save_dir}")
    log.info(f"[AP] Mode={args.mode} Task={args.task} Backbone={args.backbone}")

    # 내부 파이프라인 호출 (기존 trainer/pipeline/evaler API 유지)
    if args.mode in ("train","finetune"):
        from trainer import train_main as _train_main
        _train_main(args, save_dir, log, backbone_builder=_build_backbone, finetune=(args.mode=="finetune"))
    elif args.mode == "infer":
        from pipeline import infer_main as _infer_main
        _infer_main(args, save_dir, log, backbone_builder=_build_backbone)
    elif args.mode == "eval":
        from evaler import eval_main as _eval_main
        _eval_main(args, save_dir, log)
    else:
        raise ValueError(args.mode)

    # 실행 메타 기록 (유지보수 편의)
    try:
        with open(save_dir / "run_args.json", "w", encoding="utf-8") as f:
            json.dump({"cli": " ".join(sys.argv), "args": vars(args)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Failed to write run_args.json: {e}")

if __name__ == "__main__":
    main()
