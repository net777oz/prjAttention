# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: pipeline.py                                                         ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE                                                                   ║
║   Train / Finetune / Infer 오케스트레이션.                                ║
║   데이터 로드 → 모델 준비 → (finetune/infer) ckpt 로드 → (옵션)컴파일     ║
║   → split/학습/평가 → 리포트/플롯 → 저장.                                 ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PUBLIC INTERFACE                                                          ║
║   main_run(args: argparse.Namespace) -> None                              ║
╠───────────────────────────────────────────────────────────────────────────╣
║ KEY BEHAVIORS                                                             ║
║   • 체크포인트 로드 순서 고정:  “로드 먼저 → 컴파일 나중”                 ║
║   • ckpt/모델 키 네임스페이스 표준화: 접두사(strip) + 교집합(shape 일치)  ║
║     - 지원 접두사: "_orig_mod.", "module.", "model.", "backbone.",        ║
║                    "net.", "network."                                     ║
║   • torch.load(2.6+) 안전 로더(weights_only) → 실패 시 trusted 폴백       ║
║   • 저장은 항상 언래핑(DDP/compile) 후 ‘클린 키’ + 메타 포함              ║
║   • AMP/compile 표시와 실제 가용성 분리                                   ║
╠───────────────────────────────────────────────────────────────────────────╣
║ INPUT                                                                     ║
║   args: argparse Namespace (cli.py에서 구성)                              ║
║     - mode: {"train","finetune","infer"}                                  ║
║     - task: {"regress","classify"}                                        ║
║     - csv, context_len, epochs, batch_size, lr, weight_decay, ...         ║
║     - amp(bool), compile(str|"")                                          ║
║     - ckpt(path for finetune/infer)                                       ║
╠───────────────────────────────────────────────────────────────────────────╣
║ OUTPUT                                                                    ║
║   artifacts/<run>/                                                        ║
║     - model.pt (표준 저장 포맷)                                           ║
║     - train_report.txt / infer_report.txt                                 ║
║     - plots/*                                                             ║
╠───────────────────────────────────────────────────────────────────────────╣
║ MODULE RELATION                                                           ║
║   cli.py → main_run(args)                                                 ║
║   data.py    : set_seed, parse_csv_single, make_run_dir                   ║
║   windows.py : build_windows_dataset                                      ║
║   splits.py  : make_splits                                                ║
║   trainer.py : train_all_epochs                                           ║
║   evaler.py  : eval_model                                                 ║
║   metrics.py : metrics_from_probs, find_best_threshold_for_f1,            ║
║                compute_pos_weight_from_labels                             ║
║   viz.py     : plot_samples, plot_training_curve, plot_cls_curves         ║
║   ttm_flow.model : load_model_and_cfg(backbone, in_channels=1)            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os, time
import torch
import numpy as np
from typing import Optional, List, Tuple

from ttm_flow.model import load_model_and_cfg  # 외부 의존(원본 유지)

from config import DEFAULT_BASE_OUT
from data import set_seed, parse_csv_single, make_run_dir
from windows import build_windows_dataset
from splits import make_splits
from trainer import train_all_epochs
from evaler import eval_model
from metrics import metrics_from_probs, find_best_threshold_for_f1, compute_pos_weight_from_labels
from viz import plot_samples, plot_training_curve, plot_cls_curves


# ===== 공통: 래퍼 언래핑 / 접두사 스트립 / 저장/로드 정책 =====

_STRIP_PREFIXES = ("_orig_mod.", "module.", "model.", "backbone.", "net.", "network.")

def _get_base_module(model):
    """DDP/compile 래퍼를 벗겨 실제 원본 nn.Module 반환"""
    m = getattr(model, "module", model)    # DDP unwrap
    m = getattr(m, "_orig_mod", m)         # torch.compile unwrap
    return m

def _strip_prefix(k: str) -> str:
    for p in _STRIP_PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k

def _save_ckpt(model, args, out_path: str):
    """항상 언래핑된 원본 모듈에서 state_dict를 뽑아 ‘클린 키’+메타로 저장"""
    base = _get_base_module(model)
    sd = base.state_dict()
    sd_clean = { _strip_prefix(k): v for k, v in sd.items() }
    meta = {
        "backbone": args.backbone,
        "task": args.task,
        "in_channels": 1,
        "context_len": args.context_len,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save({"state_dict": sd_clean, "meta": meta}, out_path)
    print(f"[DONE] saved checkpoint to {out_path}", flush=True)

def _load_ckpt_if_needed(model, args):
    """1) 안전 로더(weights_only=True) 시도(+allowlist), 실패 시 trusted 폴백
       2) 접두사 스트립
       3) **원본 모듈(base)** state_dict와 교집합(shape 일치)만 로드
       4) 교집합 과소시 명시적 실패(백본/버전/헤드/채널 차이 가시화)
    """
    import torch
    from collections import Counter

    if not args.ckpt:
        raise SystemExit("--ckpt 가 필요합니다.")

    # 1) 안전 로더
    raw = None
    try:
        try:
            from torch.serialization import add_safe_globals
            import torch.torch_version
            add_safe_globals([torch.torch_version.TorchVersion])
        except Exception:
            pass
        raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except Exception as e_safe:
        print(f"[WARN] safe load failed ({e_safe.__class__.__name__}): {e_safe}", flush=True)
        print("[WARN] falling back to weights_only=False (trusted ckpt assumed).", flush=True)
        raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    sd = raw.get("state_dict", raw)

    # 2) ckpt 키 접두사 스트립
    sd = { _strip_prefix(k): v for k, v in sd.items() }

    # 3) 원본 모듈 기준 교집합 로드
    base = _get_base_module(model)
    msd = base.state_dict()

    intersect = {k: v for k, v in sd.items() if (k in msd) and (msd[k].shape == v.shape)}
    miss = [k for k in msd if k not in sd]
    unexp = [k for k in sd if k not in msd]

    print(f"[INFO] ckpt intersect={len(intersect)} / model={len(msd)} "
          f"missing={len(miss)} unexpected={len(unexp)}", flush=True)

    if len(intersect) < max(10, len(msd)//2):
        def tops(keys): return Counter(k.split('.',1)[0] for k in keys).most_common(5)
        print("[ERROR] Large mismatch between ckpt and current model.")
        print("  - ckpt top prefixes:", tops(sd.keys()))
        print("  - model top prefixes:", tops(msd.keys()))
        raise SystemExit("Backbone/version/head mismatch: aborting safe load.")

    msd.update(intersect)
    missing, unexpected = base.load_state_dict(msd, strict=False)
    print(f"[INFO] partial load done (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)


# ===== compile 적용은 “로드 후”에만 =====

def _maybe_compile(model, args):
    """체크포인트 로드가 끝난 후에만 compile 적용"""
    compiled_mode = None
    if args.compile:
        # reduce-overhead 등 비-맥스오토튠 모드에서 noisy 경고 억제
        if args.compile != "max-autotune":
            try:
                from torch._inductor import config as inductor_config
                inductor_config.max_autotune_gemm = False
            except Exception:
                pass
        try:
            model = torch.compile(model, mode=args.compile)
            compiled_mode = args.compile
            print(f"[INFO] torch.compile enabled: mode={args.compile}", flush=True)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}", flush=True)
    return model, compiled_mode


# ===== 기타 보조 =====

def _maybe_fix_context_len(args, T: int):
    if args.context_len >= T:
        print(f"ATN [WARN] context_len({args.context_len}) >= T({T}) → auto-set to T-1", flush=True)
        args.context_len = T - 1
    if args.context_len < 1:
        print(f"ATN [WARN] context_len({args.context_len}) < 1 → auto-set to 1", flush=True)
        args.context_len = 1


# ===== 메인 =====

def main_run(args):
    set_seed(args.seed)
    out_dir = make_run_dir(args, DEFAULT_BASE_OUT)

    # 데이터
    X = parse_csv_single(args)          # [N,1,T]
    N, C, T = X.shape                   # C=1 고정
    _maybe_fix_context_len(args, T)

    # AMP 실제 가용성 (플래그 + CUDA 장치)
    amp_available = bool(args.amp and torch.cuda.is_available() and str(args.device).startswith("cuda"))

    # 1) 모델 생성 (여기선 compile 하지 않음)
    model, _ = load_model_and_cfg(backbone=args.backbone, in_channels=1)
    model = model.to(device=args.device, dtype=torch.float32)
    compiled_mode = None

    print(
        f"[INFO] torch={torch.__version__} device={args.device} "
        f"amp={'ON' if amp_available else 'OFF'} "
        f"compile={'OFF' if compiled_mode is None else compiled_mode}",
        flush=True,
    )
    print(
        f"[INFO] start | mode={args.mode}, task={args.task}, split={args.split_mode}, "
        f"backbone={args.backbone}, "
        f"N={N},C={C},T={T},context={args.context_len},params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    # 2) finetune/infer이면 먼저 ckpt 로드(원본 모듈 기준)
    if args.mode in ("finetune", "infer"):
        _load_ckpt_if_needed(model, args)

    # 3) 그 다음에 compile 적용 (필요 시)
    model, compiled_mode = _maybe_compile(model, args)

    # ---- INFER ----
    if args.mode == "infer":
        if args.task == "regress":
            bmse, bmae, _, _ = eval_model(model, X, args.context_len, desc="infer", task="regress")
            print(f"[INFER] MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)
            plot_samples(model, X, args.context_len, out_dir, k=args.plot_samples, prefix="infer")
            with open(os.path.join(out_dir, "infer_report.txt"), "w", encoding="utf-8") as f:
                f.write(f"infer MSE {bmse:.6f} MAE {bmae:.6f}\n")
        else:
            _, _, ytrue, yprob = eval_model(
                model, X, args.context_len, desc="infer",
                task="classify", bin_rule=args.bin_rule, bin_thr=args.bin_thr,
                tau_for_cls=args.thresh_default
            )
            if yprob is not None and yprob.numel():
                rep = metrics_from_probs(ytrue, yprob, threshold=args.thresh_default)
                print(
                    f"[INFER-CLS] τ={rep['threshold']:.3f} | "
                    f"F1={rep['F1']:.6f} Acc={rep['Accuracy']:.6f} "
                    f"P={rep['Precision']:.6f} R={rep['Recall']:.6f} "
                    f"(MSE={rep['MSE']:.6f}, MAE={rep['MAE']:.6f})",
                    flush=True,
                )
                with open(os.path.join(out_dir, "infer_report.txt"), "w", encoding="utf-8") as f:
                    f.write(
                        f"τ {rep['threshold']:.3f} F1 {rep['F1']:.6f} Acc {rep['Accuracy']:.6f} "
                        f"P {rep['Precision']:.6f} R {rep['Recall']:.6f} "
                        f"MSE {rep['MSE']:.6f} MAE {rep['MAE']:.6f}\n"
                    )
                plot_cls_curves(ytrue, yprob, args.thresh_default, out_dir, prefix="infer")
        _save_ckpt(model, args, os.path.join(out_dir, "model.pt"))
        print(f"[DONE] infer saved to {out_dir}", flush=True)
        return

    # ---- TRAIN/FINETUNE ----

    # BEFORE eval
    print("[INFO] starting BEFORE eval...", flush=True)
    if args.task == "regress":
        bmse, bmae, _, _ = eval_model(model, X, args.context_len, desc="before", task="regress")
        print(f"[BEFORE] MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)
    else:
        _, _, ytrue_b, yprob_b = eval_model(
            model, X, args.context_len, desc="before",
            task="classify", bin_rule=args.bin_rule, bin_thr=args.bin_thr,
            tau_for_cls=args.thresh_default
        )
        if yprob_b is not None and yprob_b.numel():
            rep_b = metrics_from_probs(ytrue_b, yprob_b, threshold=args.thresh_default)
            print(
                f"[BEFORE-CLS] τ={rep_b['threshold']:.3f} | "
                f"F1={rep_b['F1']:.6f} Acc={rep_b['Accuracy']:.6f} "
                f"P={rep_b['Precision']:.6f} R={rep_b['Recall']:.6f} "
                f"(MSE={rep_b['MSE']:.6f}, MAE={rep_b['MAE']:.6f})",
                flush=True,
            )
            plot_cls_curves(ytrue_b, yprob_b, args.thresh_default, out_dir, prefix="before")

    # 윈도우 & (분류) 타깃 변환
    Xw_all, Yw_next, groups, W = build_windows_dataset(X, args.context_len)
    if args.task == "regress":
        Yw_all = Yw_next
    else:
        if args.bin_rule == "nonzero":
            Yw_all = (Yw_next != 0).float()
        elif args.bin_rule == "gt":
            Yw_all = (Yw_next > args.bin_thr).float()
        else:
            Yw_all = (Yw_next >= args.bin_thr).float()

    # split
    trn_idx, val_idx = make_splits(
        Xw_all, Yw_all, groups, N, T, args.context_len, W,
        args.split_mode, args.val_ratio, args.seed
    )
    print(
        f"[INFO] split={args.split_mode} train_windows={len(trn_idx)} val_windows={len(val_idx)}",
        flush=True,
    )

    # DataLoader
    from torch.utils.data import TensorDataset, DataLoader, Subset
    ds = TensorDataset(Xw_all, Yw_all)
    dl = DataLoader(
        Subset(ds, trn_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=True if args.num_workers > 0 else False
    )
    vl = DataLoader(
        Subset(ds, val_idx), batch_size=max(1, args.eval_batch_size), shuffle=False,
        num_workers=0, pin_memory=True
    )

    # 옵티마이저
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # GradScaler (권장 API, 실제 가용성 반영)
    from torch import amp as torch_amp
    scaler = torch_amp.GradScaler("cuda", enabled=amp_available)

    # 전역 pos_weight
    global_pw = None
    if args.task == "classify" and args.pos_weight != "none":
        y_trn = Yw_all[trn_idx]
        global_pw = compute_pos_weight_from_labels(y_trn)
        print(f"[INFO] pos_weight(global) = {global_pw:.6f} (mode={args.pos_weight})", flush=True)

    # 학습 (AMP 실제 가용성 기준 전달)
    loss_hist = train_all_epochs(
        model, dl, opt, scaler,
        epochs=args.epochs,
        amp_enabled=amp_available,
        log_every=args.log_every,
        compiled=compiled_mode,
        task=args.task, alpha=args.alpha,
        pos_weight_mode=args.pos_weight, global_pos_weight=global_pw
    )
    plot_training_curve(loss_hist, out_dir, title="Training Loss")

    # AFTER eval + (분류) τ 선택
    print("[INFO] starting AFTER eval...", flush=True)
    if args.task == "regress":
        amse, amae, _, _ = eval_model(model, X, args.context_len, desc="after", task="regress")
        print(f"[AFTER ] MSE={amse:.6f} MAE={amae:.6f}", flush=True)
        with open(os.path.join(out_dir, "train_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"before MSE {bmse:.6f} MAE {bmae:.6f}\n")
            f.write(f"after  MSE {amse:.6f} MAE {amae:.6f}\n")
        plot_samples(model, X, args.context_len, out_dir, k=args.plot_samples, prefix="after")
    else:
        # 검증 윈도우 기반 τ 선택 (배치 추론, OOM-안전)
        probs_chunks, yv_chunks = [], []
        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(args.device, non_blocking=True)
                if amp_available:
                    from torch import amp
                    with amp.autocast(device_type='cuda', enabled=True):
                        logits_b, _ = model(xb)
                else:
                    logits_b, _ = model(xb)
                probs_chunks.append(torch.sigmoid(logits_b).detach().cpu())
                yv_chunks.append(yb.detach().cpu())
        if probs_chunks:
            probs_val = torch.cat(probs_chunks, dim=0)
            y_val = torch.cat(yv_chunks, dim=0)
            tau, f1_at_tau = find_best_threshold_for_f1(y_val, probs_val, step=0.001)
        else:
            tau, f1_at_tau = args.thresh_default, float("nan")
            print("[WARN] 검증 분할이 없어 τ 기본값을 사용합니다.", flush=True)

        # 전체 데이터 리포트 + 플롯
        _, _, ytrue_all, yprob_all = eval_model(
            model, X, args.context_len, desc="after",
            task="classify", bin_rule=args.bin_rule, bin_thr=args.bin_thr,
            tau_for_cls=tau
        )
        rep = metrics_from_probs(ytrue_all, yprob_all, threshold=tau)
        print(
            f"[AFTER-CLS] Selected τ={rep['threshold']:.3f} | "
            f"F1={rep['F1']:.6f} Acc={rep['Accuracy']:.6f} "
            f"P={rep['Precision']:.6f} R={rep['Recall']:.6f} "
            f"(MSE={rep['MSE']:.6f}, MAE={rep['MAE']:.6f})",
            flush=True,
        )
        with open(os.path.join(out_dir, "train_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"tau {rep['threshold']:.3f}\n")
            f.write(f"F1 {rep['F1']:.6f} Acc {rep['Accuracy']:.6f} P {rep['Precision']:.6f} R {rep['Recall']:.6f}\n")
            f.write(f"(ref) MSE {rep['MSE']:.6f} MAE {rep['MAE']:.6f}\n")
        plot_cls_curves(ytrue_all, yprob_all, rep["threshold"], out_dir, prefix="after")

    # 표준 저장
    _save_ckpt(model, args, os.path.join(out_dir, "model.pt"))
    print(f"[DONE] model/report/plots saved to {out_dir}", flush=True)
