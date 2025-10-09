# -*- coding: utf-8 -*-
"""
pipeline.py — Train / Finetune / Infer 오케스트레이션 (다변량 입력 지원)
변경 핵심:
  • data.parse_csv_auto 로 [N,1,T] 또는 [N,C,T] 입력 허용
  • in_channels=C 자동 검출 후 load_model_and_cfg(backbone, in_channels=C)
  • 체크포인트 메타에 in_channels 반영(기존 형식 유지)
  • 나머지 플로우/모듈 경계는 원본 유지
"""
import os, time
import torch
import numpy as np
from typing import Optional, List, Tuple

from ttm_flow.model import load_model_and_cfg  # 외부 의존(원본 유지)

from config import DEFAULT_BASE_OUT
from data import set_seed, parse_csv_auto, make_run_dir  # ← 변경: auto 사용
from windows import build_windows_dataset
from splits import make_splits
from trainer import train_all_epochs
from evaler import eval_model
from metrics import metrics_from_probs, find_best_threshold_for_f1, compute_pos_weight_from_labels
from viz import plot_samples, plot_training_curve, plot_cls_curves

# ===== 공통: 래퍼 언래핑 / 접두사 스트립 / 저장/로드 정책 =====
_STRIP_PREFIXES = ("_orig_mod.", "module.", "model.", "backbone.", "net.", "network.")

def _get_base_module(model):
    m = getattr(model, "module", model)    # DDP unwrap
    m = getattr(m, "_orig_mod", m)         # torch.compile unwrap
    return m

def _strip_prefix(k: str) -> str:
    for p in _STRIP_PREFIXES:
        if k.startswith(p): return k[len(p):]
    return k

def _save_ckpt(model, args, out_path: str):
    """항상 언래핑된 원본 모듈에서 state_dict를 뽑아 ‘클린 키’+메타로 저장"""
    base = _get_base_module(model)
    sd = base.state_dict()
    sd_clean = { _strip_prefix(k): v for k, v in sd.items() }
    meta = {
        "backbone": args.backbone,
        "task": args.task,
        "in_channels": int(getattr(args, "_in_channels", 1)),  # ← 채널 수 반영
        "context_len": args.context_len,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save({"state_dict": sd_clean, "meta": meta}, out_path)
    print(f"[DONE] saved checkpoint to {out_path}", flush=True)

def _load_ckpt_if_needed(model, args):
    import torch
    from collections import Counter
    if not args.ckpt:
        raise SystemExit("--ckpt 가 필요합니다.")
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
    sd = { _strip_prefix(k): v for k, v in sd.items() }
    base = _get_base_module(model)
    msd = base.state_dict()

    intersect = {k: v for k, v in sd.items() if (k in msd) and (msd[k].shape == v.shape)}
    miss = [k for k in msd if k not in sd]
    unexp = [k for k in sd if k not in msd]

    print(f"[INFO] ckpt intersect={len(intersect)} / model={len(msd)} missing={len(miss)} unexpected={len(unexp)}", flush=True)

    if len(intersect) < max(10, len(msd)//2):
        def tops(keys): 
            from collections import Counter
            return Counter(k.split('.',1)[0] for k in keys).most_common(5)
        print("[ERROR] Large mismatch between ckpt and current model.")
        print("  - ckpt top prefixes:", tops(sd.keys()))
        print("  - model top prefixes:", tops(msd.keys()))
        raise SystemExit("Backbone/version/head mismatch: aborting safe load.")

    msd.update(intersect)
    missing, unexpected = base.load_state_dict(msd, strict=False)
    print(f"[INFO] partial load done (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

def _maybe_compile(model, args):
    compiled_mode = None
    if args.compile:
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

    # 데이터: 단일/다중 자동
    X = parse_csv_auto(args)            # [N,1,T] 또는 [N,C,T]
    N, C, T = X.shape
    args._in_channels = int(C)          # ← 채널 수를 args에 기록(메타/출력 이름 반영)
    _maybe_fix_context_len(args, T)

    out_dir = make_run_dir(args, DEFAULT_BASE_OUT)

    # AMP 실제 가용성
    amp_available = bool(args.amp and torch.cuda.is_available() and str(args.device).startswith("cuda"))

    # 1) 모델 생성 (compile 전)
    model, _ = load_model_and_cfg(backbone=args.backbone, in_channels=int(C))  # ← C 반영
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

    # 2) finetune/infer이면 먼저 ckpt 로드
    if args.mode in ("finetune", "infer"):
        _load_ckpt_if_needed(model, args)

    # 3) compile 적용(필요 시)
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

    # GradScaler
    from torch import amp as torch_amp
    scaler = torch_amp.GradScaler("cuda", enabled=amp_available)

    # 전역 pos_weight
    global_pw = None
    if args.task == "classify" and args.pos_weight != "none":
        y_trn = Yw_all[trn_idx]
        global_pw = compute_pos_weight_from_labels(y_trn)
        print(f"[INFO] pos_weight(global) = {global_pw:.6f} (mode={args.pos_weight})", flush=True)

    # 학습
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

    # 표준 저장 (메타에 in_channels 포함)
    _save_ckpt(model, args, os.path.join(out_dir, "model.pt"))
    print(f"[DONE] model/report/plots saved to {out_dir}", flush=True)
