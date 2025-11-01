# -*- coding: utf-8 -*-
"""
pipeline.py — Train / Finetune / Infer 오케스트레이션 (다변량 입력 지원)
변경 핵심:
  • data.parse_csv_auto 로 [N,1,T] 또는 [N,C,T] 입력 허용
  • in_channels=C 자동 검출 후 load_model_and_cfg(backbone, in_channels=C)
  • 체크포인트 메타에 in_channels 반영(기존 형식 유지)
  • (중요) 평가/플롯은 검증 행(row)만 사용하도록 고정
  • (신규) out_dir 확정 직후 AP_OUT_DIR를 내부적으로 고정 → viz/evaler가 동일 런 폴더만 사용
  • (신규) plot_samples 호출을 viz.py 현재 시그니처로 교체
  • (신규) 리포트용 산출물 저장: manifest.json / (classify) metrics.json+preds.csv / (regress) summary.json
  • (신규) 옵션/ENV로 라벨 소스 채널을 입력 특징에서 제외(drop)하고 라벨로만 사용
"""
import os, time, json, csv
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
        # 신규 기록(재현성): 라벨 소스/드롭 정보
        "label_src_ch": int(getattr(args, "_label_src_ch", 0)),
        "dropped_channels": list(getattr(args, "_dropped_channels", [])),
    }
    torch.save({"state_dict": sd_clean, "meta": meta}, out_path)
    print(f"[DONE] saved checkpoint to {out_path}", flush=True)

def _load_ckpt_if_needed(model, args):
    import torch
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

def _rows_from_indices(groups: torch.Tensor, idx_list: List[int]) -> List[int]:
    """윈도우 인덱스 → 원본 행(row) 인덱스 집합"""
    if not isinstance(groups, torch.Tensor):
        groups = torch.as_tensor(groups)
    if len(idx_list) == 0:
        return []
    sel = groups[torch.as_tensor(idx_list, dtype=torch.long)]
    rows = torch.unique(sel).cpu().tolist()
    return sorted(int(r) for r in rows)

# ──────────────────────── report helpers ─────────────────────────

import json as _json_for_helper  # shadow guard

def _write_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] write json failed {path}: {e}", flush=True)

def _write_preds_csv(path, y_true_np, y_score_np, tau):
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true", "y_score", f"y_pred@tau={tau:.3f}"])
            yp = (y_score_np >= float(tau)).astype(int)
            for yt, ys, yp1 in zip(y_true_np, y_score_np, yp):
                w.writerow([int(yt), float(ys), int(yp1)])
    except Exception as e:
        print(f"[WARN] write preds.csv failed: {e}", flush=True)

def _try_auroc_auprc(y_true_t, y_prob_t):
    """metrics.roc_auc_pr_auc()이 있으면 사용, 없으면 (None, None)"""
    try:
        from metrics import roc_auc_pr_auc  # optional
        auroc, auprc = roc_auc_pr_auc(y_true_t, y_prob_t)
        return float(auroc), float(auprc)
    except Exception:
        return None, None

def _save_manifest(out_dir, args, N, C, T, train_w, val_w, device, amp_available, compiled_mode, model):
    base = _get_base_module(model)
    n_params = sum(p.numel() for p in base.parameters())
    man = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": args.backbone,
        "task": args.task,
        "context_len": int(args.context_len),
        "in_channels": int(getattr(args, "_in_channels", C)),
        "split_mode": args.split_mode,
        "seed": int(args.seed),
        "device": str(args.device),
        "amp": bool(amp_available),
        "compile": compiled_mode if compiled_mode else "OFF",
        "model_size": {"params": int(n_params)},
        "data": {"N": int(N), "C_original": int(C), "T": int(T), "train_windows": int(train_w), "val_windows": int(val_w)},
        "label_src_ch": int(getattr(args, "_label_src_ch", 0)),
        "dropped_channels": list(getattr(args, "_dropped_channels", [])),
    }
    _write_json(os.path.join(out_dir, "manifest.json"), man)

# ===== 메인 =====
def main_run(args):
    set_seed(args.seed)

    # 데이터: 단일/다중 자동
    X = parse_csv_auto(args)            # [N,1,T] 또는 [N,C,T]
    N, C, T = X.shape

    # ---------- 신규: 라벨소스/드롭 플래그 수집 (CLI > ENV > default) ----------
    drop_label_from_x = bool(getattr(args, "drop_label_from_x", False) or (os.environ.get("AP_DROP_LABEL_FROM_X", "0") == "1"))
    label_src_ch = int(getattr(args, "label_src_ch", os.environ.get("AP_LABEL_SRC_CH", "0")))
    if label_src_ch < 0 or label_src_ch >= C:
        print(f"[WARN] label_src_ch({label_src_ch}) out of range for C={C} → clamp to 0", flush=True)
        label_src_ch = 0
    dropped = set([label_src_ch]) if drop_label_from_x else set()

    # 모델 입력 채널 수 계산(라벨 채널 제외 여부 반영)
    C_eff = C - (1 if drop_label_from_x else 0)
    if C_eff <= 0:
        raise SystemExit(f"[data] Effective input channels becomes 0 (C={C}, drop_label_from_x={drop_label_from_x}). Need at least one non-label feature.")

    # 메타/ENV 주입(윈도우/평가 경로 공통 적용)
    args._in_channels = int(C_eff)
    args._label_src_ch = int(label_src_ch)
    args._dropped_channels = sorted(list(dropped))
    os.environ.setdefault("AP_OUT_DIR", "")  # 아래에서 재설정
    os.environ["AP_LABEL_SRC_CH"] = str(label_src_ch)
    os.environ["AP_DROP_LABEL_FROM_X"] = "1" if drop_label_from_x else "0"

    _maybe_fix_context_len(args, T)

    # --- out_dir 결정: ① CLI --outdir > ② ENV AP_OUTDIR > ③ 기존 make_run_dir ---
    _cli_outdir = getattr(args, "outdir", None)
    _env_outdir = os.environ.get("AP_OUTDIR", "").strip()

    if _cli_outdir:
        out_dir = os.path.join(DEFAULT_BASE_OUT, _cli_outdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] outdir(CLI) → {out_dir}", flush=True)
    elif _env_outdir:
        out_dir = os.path.join(DEFAULT_BASE_OUT, _env_outdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] outdir(ENV:AP_OUTDIR) → {out_dir}", flush=True)
    else:
        out_dir = make_run_dir(args, DEFAULT_BASE_OUT)  # 기존 자동 네이밍 유지
        print(f"[INFO] outdir(auto) → {out_dir}", flush=True)

    # ↓ 아래는 기존 로직 그대로
    os.environ["AP_OUT_DIR"] = str(out_dir)

    # AMP 실제 가용성
    amp_available = bool(args.amp and torch.cuda.is_available() and str(args.device).startswith("cuda"))

    # 1) 모델 생성 (compile 전) — 입력 채널은 C_eff 사용
    model, _ = load_model_and_cfg(backbone=args.backbone, in_channels=int(C_eff))
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
        f"N={N},C={C}(eff={C_eff}),T={T},context={args.context_len},params={sum(p.numel() for p in model.parameters())}, "
        f"label_src_ch={label_src_ch}, drop_label_from_x={drop_label_from_x}",
        flush=True,
    )

    # 2) finetune/infer이면 ckpt 로드
    if args.mode in ("finetune", "infer"):
        _load_ckpt_if_needed(model, args)

    # 3) compile 적용(필요 시)
    model, compiled_mode = _maybe_compile(model, args)

    # ---- 공통 전처리: 윈도우/스플릿 준비
    #   windows.build_windows_dataset는 ENV(AP_DROP_LABEL_FROM_X/AP_LABEL_SRC_CH)를 읽어
    #   라벨(ch=label_src_ch)만 유지하고, 입력 채널에서는 필요 시 제외 처리합니다.
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

    trn_idx, val_idx = make_splits(
        Xw_all, Yw_all, groups, N, T, args.context_len, W,
        args.split_mode, args.val_ratio, args.seed
    )
    print(f"[INFO] split={args.split_mode} train_windows={len(trn_idx)} val_windows={len(val_idx)}", flush=True)

    # ★ 행(row) 기준 검증/학습 집합 복원
    val_rows = _rows_from_indices(groups, val_idx)
    trn_rows = _rows_from_indices(groups, trn_idx)
    X_val_rows = X[val_rows] if len(val_rows) else X[:0]
    X_trn_rows = X[trn_rows] if len(trn_rows) else X[:0]
    print(f"[CHECK] rows.train={len(trn_rows)} rows.val={len(val_rows)}", flush=True)

    # ---- INFER ----
    if args.mode == "infer":
        # 추론은 전체 X 대상(요구사항 그대로 유지)
        if args.task == "regress":
            bmse, bmae, _, _ = eval_model(model, X, args.context_len, desc="infer", task="regress")
            print(f"[INFER] MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)
            try:
                plot_samples(X, split="infer", max_samples=args.plot_samples, ch=0, title="Infer Samples")
            except Exception as e:
                print(f"[WARN] plot_samples failed: {e}", flush=True)
            # 회귀 요약 저장 (전체 데이터 기준)
            summ = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "avg_mse": float(bmse), "avg_mae": float(bmae),
                "n_windows": int(W)
            }
            _write_json(os.path.join(out_dir, "summary.json"), summ)
        else:
            # ── (분류 infer) 순서/원본행 추적을 위한 최소 침습 로더 구성 ──
            from torch.utils.data import TensorDataset, DataLoader

            # 윈도우 인덱스 포함 데이터셋
            idx_all = torch.arange(len(Yw_all), dtype=torch.long)
            ds = TensorDataset(Xw_all, Yw_all, idx_all)

            dl = DataLoader(
                ds,
                batch_size=max(1, getattr(args, "eval_batch_size", 8192)),
                shuffle=True,  # 섞여도 order로 복원 가능
                num_workers=getattr(args, "num_workers", 0),
                pin_memory=True,
                drop_last=False,
                persistent_workers=True if getattr(args, "num_workers", 0) > 0 else False
            )

            probs_chunks, y_chunks, order_chunks = [], [], []

            model.eval()
            amp_available = bool(args.amp and torch.cuda.is_available() and str(args.device).startswith("cuda"))
            with torch.no_grad():
                for xb, yb, wi in dl:
                    xb = xb.to(args.device, non_blocking=True)
                    if amp_available:
                        from torch import amp
                        with amp.autocast(device_type='cuda', enabled=True):
                            logits_b, _ = model(xb)
                    else:
                        logits_b, _ = model(xb)

                    if logits_b.dim() == 2:
                        logits_b = logits_b[:, 0]     # [B]

                    probs_b = torch.sigmoid(logits_b)   # [B]
                    probs_chunks.append(probs_b.detach().cpu())
                    y_chunks.append(yb.detach().cpu())  # 1D
                    order_chunks.append(wi.detach().cpu())

            if probs_chunks:
                yprob = torch.cat(probs_chunks, dim=0).view(-1).float()
                ytrue = torch.cat(y_chunks, dim=0).view(-1).float()
                order = torch.cat(order_chunks, dim=0).view(-1).long().numpy()
            else:
                yprob = torch.empty(0)
                ytrue = torch.empty(0)
                order = None

            # ── 기존 메트릭/곡선 출력 유지 ─────────────────────────────────────────────
            if yprob is not None and yprob.numel():
                rep = metrics_from_probs(ytrue, yprob, threshold=args.thresh_default)
                print(
                    f"[INFER-CLS] τ={rep['threshold']:.3f} | "
                    f"F1={rep['F1']:.6f} Acc={rep['Accuracy']:.6f} "
                    f"P={rep['Precision']:.6f} R={rep['Recall']:.6f} "
                    f"(MSE={rep['MSE']:.6f}, MAE={rep['MAE']:.6f})",
                    flush=True,
                )
                plot_cls_curves(ytrue, yprob, split="infer", tau=args.thresh_default)
                # 분류 요약 저장 (전체 데이터 기준)
                auroc, auprc = _try_auroc_auprc(ytrue, yprob)
                metrics_json = {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "counts": {"n": int(ytrue.numel()), "pos": int((ytrue==1).sum().item()), "neg": int((ytrue==0).sum().item())},
                    "prevalence": float((ytrue==1).float().mean().item()) if ytrue.numel() else None,
                    "threshold_default": float(args.thresh_default),
                    "metrics_at_default": {
                        "threshold": float(rep["threshold"]),
                        "precision": float(rep["Precision"]), "recall": float(rep["Recall"]),
                        "f1": float(rep["F1"]), "accuracy": float(rep["Accuracy"]),
                        "tp": int(rep["TP"]), "tn": int(rep["TN"]), "fp": int(rep["FP"]), "fn": int(rep["FN"]),
                    },
                    "threshold_best": None, "metrics_at_best": {},
                    "roc_auc": auroc, "pr_auc": auprc
                }
                _write_json(os.path.join(out_dir, "metrics.json"), metrics_json)

            # ── preds.csv: seq/uid/row 포함 저장 (리스트 csv 대응) ────────────────────
            import numpy as _np, csv as _csv, os as _os
            csv_arg = getattr(args, "csv", "X.csv")
            if isinstance(csv_arg, (list, tuple)):
                csv_base = _os.path.basename(str(csv_arg[0])) if len(csv_arg) else "X.csv"
            else:
                csv_base = _os.path.basename(str(csv_arg))

            ctx = int(args.context_len)

            if order is None:
                order = _np.arange(yprob.numel(), dtype=_np.int64)

            seq = order.astype(_np.int64)                     # 배출 순서(복원용)
            rows = _np.asarray(groups, dtype=_np.int64)       # 원본 행 매핑(groups[win_idx])
            uid  = [f"{csv_base}#row={int(rows[i])}#ctx={ctx}#off=1" for i in order]

            tau = float(args.thresh_default)
            ypred = (yprob.numpy() >= tau).astype(int)

            with open(os.path.join(out_dir, "preds.csv"), "w", encoding="utf-8", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["seq","uid","row","y_true","y_score", f"y_pred@tau={tau:.3f}"])
                for j, i in enumerate(order):
                    w.writerow([
                        int(seq[j]),
                        uid[j],
                        int(rows[i]),
                        int(ytrue[j].item()),
                        float(yprob[j].item()),
                        int(ypred[j]),
                    ])

        _save_manifest(out_dir, args, N, C, T, len(trn_idx), len(val_idx), args.device, amp_available, compiled_mode, model)
        _save_ckpt(model, args, os.path.join(out_dir, "model.pt"))
        print(f"[DONE] infer saved to {out_dir}", flush=True)
        return

    # ---- TRAIN/FINETUNE ----

    # BEFORE eval (★ 이제 '검증 행' 기준으로만)
    print("[INFO] starting BEFORE eval (on VAL rows)...", flush=True)
    if args.task == "regress":
        bmse, bmae, _, _ = eval_model(model, X_val_rows, args.context_len, desc="before", task="regress")
        print(f"[BEFORE] (val) MSE={bmse:.6f} MAE={bmae:.6f}", flush=True)
    else:
        _, _, ytrue_b, yprob_b = eval_model(
            model, X_val_rows, args.context_len, desc="before",
            task="classify", bin_rule=args.bin_rule, bin_thr=args.bin_thr,
            tau_for_cls=args.thresh_default
        )
        if yprob_b is not None and yprob_b.numel():
            rep_b = metrics_from_probs(ytrue_b, yprob_b, threshold=args.thresh_default)
            print(
                f"[BEFORE-CLS] (val) τ={rep_b['threshold']:.3f} | "
                f"F1={rep_b['F1']:.6f} Acc={rep_b['Accuracy']:.6f} "
                f"P={rep_b['Precision']:.6f} R={rep_b['Recall']:.6f} "
                f"(MSE={rep_b['MSE']:.6f}, MAE={rep_b['MAE']:.6f})",
                flush=True,
            )
            plot_cls_curves(ytrue_b, yprob_b, split="val", tau=args.thresh_default)

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
    plot_training_curve(loss_hist, split="val", title="Training Loss")

    # AFTER eval + (분류) τ 선택  — ★ 전부 '검증 행' 기준
    print("[INFO] starting AFTER eval (on VAL rows)...", flush=True)
    if args.task == "regress":
        amse, amae, _, _ = eval_model(model, X_val_rows, args.context_len, desc="after", task="regress")
        print(f"[AFTER ] (val) MSE={amse:.6f} MAE={amae:.6f}", flush=True)
        with open(os.path.join(out_dir, "train_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"before(val) MSE {bmse:.6f} MAE {bmae:.6f}\n")
            f.write(f"after (val) MSE {amse:.6f} MAE {amae:.6f}\n")
        try:
            plot_samples(X_val_rows, split="val", max_samples=args.plot_samples, ch=0, title="After (val) Samples")
        except Exception as e:
            print(f"[WARN] plot_samples failed: {e}", flush=True)
        # 회귀 summary.json (VAL 기준)
        summ = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avg_mse": float(amse), "avg_mae": float(amae),
            "n_windows": int(len(val_idx))
        }
        _write_json(os.path.join(out_dir, "summary.json"), summ)
    else:
        # τ 선택은 'val windows(vl)'로 수행
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
                if logits_b.dim() == 2:
                    logits_b = logits_b[:, 0]          # [B]
                probs_b = torch.sigmoid(logits_b)       # [B]
                probs_chunks.append(probs_b.detach().cpu())
                yv_chunks.append(yb.detach().cpu())     # Yw_all은 1D

        if probs_chunks:
            probs_val = torch.cat(probs_chunks, dim=0).view(-1).float()
            y_val = torch.cat(yv_chunks, dim=0).view(-1).float()
            assert probs_val.numel() == y_val.numel(), \
                f"probs({probs_val.numel()}) vs labels({y_val.numel()}) length mismatch"
            tau, f1_at_tau = find_best_threshold_for_f1(y_val, probs_val, step=0.001)
        else:
            tau, f1_at_tau = args.thresh_default, float("nan")
            print("[WARN] 검증 분할이 없어 τ 기본값을 사용합니다.", flush=True)

        # AFTER (VAL rows) 정식 평가
        _, _, ytrue_val, yprob_val = eval_model(
            model, X_val_rows, args.context_len, desc="after",
            task="classify", bin_rule=args.bin_rule, bin_thr=args.bin_thr,
            tau_for_cls=tau
        )
        rep = metrics_from_probs(ytrue_val, yprob_val, threshold=tau)
        print(
            f"[AFTER-CLS] (val) τ={rep['threshold']:.3f} | "
            f"F1={rep['F1']:.6f} Acc={rep['Accuracy']:.6f} "
            f"P={rep['Precision']:.6f} R={rep['Recall']:.6f} "
            f"(MSE={rep['MSE']:.6f}, MAE={rep['MAE']:.6f})",
            flush=True,
        )
        with open(os.path.join(out_dir, "train_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"tau {rep['threshold']:.3f}\n")
            f.write(f"(val) F1 {rep['F1']:.6f} Acc {rep['Accuracy']:.6f} "
                    f"P {rep['Precision']:.6f} R {rep['Recall']:.6f}\n")
            f.write(f"(ref) MSE {rep['MSE']:.6f} MAE {rep['MAE']:.6f}\n")
        plot_cls_curves(ytrue_val, yprob_val, split="val", tau=rep["threshold"])

        # 분류 metrics.json / preds.csv 저장 (VAL 기준)
        auroc, auprc = _try_auroc_auprc(ytrue_val, yprob_val)
        metrics_json = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "counts": {"n": int(ytrue_val.numel()), "pos": int((ytrue_val==1).sum().item()), "neg": int((ytrue_val==0).sum().item())},
            "prevalence": float((ytrue_val==1).float().mean().item()) if ytrue_val.numel() else None,
            "threshold_default": float(args.thresh_default),
            "metrics_at_default": {
                "threshold": float(args.thresh_default),
                "precision": float(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["Precision"]),
                "recall": float(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["Recall"]),
                "f1": float(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["F1"]),
                "accuracy": float(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["Accuracy"]),
                "tp": int(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["TP"]),
                "tn": int(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["TN"]),
                "fp": int(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["FP"]),
                "fn": int(metrics_from_probs(ytrue_val, yprob_val, threshold=args.thresh_default)["FN"]),
            },
            "threshold_best": float(rep["threshold"]),
            "metrics_at_best": {
                "threshold": float(rep["threshold"]),
                "precision": float(rep["Precision"]), "recall": float(rep["Recall"]),
                "f1": float(rep["F1"]), "accuracy": float(rep["Accuracy"]),
                "tp": int(rep["TP"]), "tn": int(rep["TN"]), "fp": int(rep["FP"]), "fn": int(rep["FN"]),
            },
            "roc_auc": auroc, "pr_auc": auprc
        }
        _write_json(os.path.join(out_dir, "metrics.json"), metrics_json)
        _write_preds_csv(
            os.path.join(out_dir, "preds.csv"),
            y_true_np=ytrue_val.detach().cpu().numpy().astype(int),
            y_score_np=yprob_val.detach().cpu().numpy().astype(float),
            tau=float(rep["threshold"])
        )

    # 공통 manifest.json 저장 (러닝 메타)
    _save_manifest(out_dir, args, N, C, T, len(trn_idx), len(val_idx), args.device, amp_available, compiled_mode, model)

    # 표준 저장 (메타에 in_channels 포함)
    _save_ckpt(model, args, os.path.join(out_dir, "model.pt"))
    print(f"[DONE] model/report/plots saved to {out_dir}", flush=True)
