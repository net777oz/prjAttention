# -*- coding: utf-8 -*-
"""
evaler.py — 회귀/분류 공용 OOM-안전 평가 (이미지 비생성)
- 저장: "런 폴더 루트(AP_OUT_DIR)" 바로 아래에만 JSON/CSV 저장
  * 분류:  <run>/metrics.json, <run>/preds.csv
  * 회귀:  <run>/summary.json, <run>/after_pred.csv   ← 추가
- 그림은 만들지 않음 (viz.py가 plots/after_<split>_*.png 한 군데만 생성)

저장 시점 정책:
  * 기본: AFTER / INFER 일 때만 저장 (BEFORE 등에서는 저장 안 함)
  * 환경변수 AP_EVAL_SAVE_SCOPE:
      - "auto" 또는 "after_only" → 위와 동일(기본)
      - "never" → 어떤 단계에서도 파일 저장 안 함

반환:
  회귀: (avg_mse, avg_mae, None, None)
  분류: (None, None, y_true_all, y_prob_all)
"""
import os, json, time, numpy as np, torch, torch.nn.functional as F
from typing import Optional, List
from pathlib import Path

from windows import build_windows_dataset
from utils import USE_RICH, console, build_eval_table

# ─────────────────────────── 경로/정책 유틸 ───────────────────────────

def _run_root() -> Path:
    base = os.environ.get("AP_OUT_DIR")
    root = Path(base).expanduser().resolve() if base else Path("./artifacts").resolve() / "default_run"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _should_save(desc: str, split_name: Optional[str]) -> bool:
    """AFTER/INFER에서만 저장. AP_EVAL_SAVE_SCOPE로 오버라이드 가능."""
    scope = (os.environ.get("AP_EVAL_SAVE_SCOPE") or "auto").lower()
    if scope == "never":
        return False
    # auto / after_only
    tag = (split_name or desc or "").lower()
    # "after", "infer"가 들어가면 저장
    return ("after" in tag) or ("infer" in tag)

def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

# ─────────────────────────── 지표 유틸 ───────────────────────────

def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score); yt = y_true[order]; ys = y_score[order]
    P = float((yt==1).sum()); N = float((yt==0).sum())
    if P==0 or N==0: return np.array([0.0,1.0]), np.array([0.0,1.0])
    tps = np.cumsum(yt==1).astype(float); fps = np.cumsum(yt==0).astype(float)
    distinct = np.where(np.diff(ys))[0]; idx = np.r_[distinct, yt.size-1]
    tpr = tps[idx]/P; fpr = fps[idx]/N
    return np.r_[0.0,fpr,1.0], np.r_[0.0,tpr,1.0]

def _pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score); yt = y_true[order]; P = float((yt==1).sum())
    tp = np.cumsum(yt==1).astype(float); fp = np.cumsum(yt==0).astype(float); fn = P - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
        rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)>0)
    return np.r_[0.0,rec], np.r_[1.0,prec]

def _trapz(x,y): return float(np.trapz(y,x)) if x.size>1 else 0.0

def _bin_metrics(y, yhat):
    tp=float(np.sum((y==1)&(yhat==1))); tn=float(np.sum((y==0)&(yhat==0)))
    fp=float(np.sum((y==0)&(yhat==1))); fn=float(np.sum((y==1)&(yhat==0)))
    prec=tp/(tp+fp) if (tp+fp)>0 else 0.0; rec=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=(2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0; acc=(tp+tn)/max(1.0,tp+tn+fp+fn)
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,precision=prec,recall=rec,f1=f1,accuracy=acc)

def _best_threshold(y, p, strategy="f1"):
    ths=np.linspace(0,1,1001); best=(-1,0.5,{})
    for t in ths:
        m=_bin_metrics(y,(p>=t).astype(int)); v=m["f1"] if strategy=="f1" else (m["recall"]+m["accuracy"])/2
        if v>best[0]: best=(v,float(t),m)
    return best[1], best[2]

# ─────────────────────────── 저장 루틴 (이미지 없음) ───────────────────────────

def _save_preds_csv(run_root: Path, y_true: np.ndarray, y_score: np.ndarray, tau_default: float):
    try:
        import csv
        path=run_root/"preds.csv"; y_pred=(y_score>=tau_default).astype(int)
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["y_true","y_score",f"y_pred@tau={tau_default:.3f}"])
            for yt,ys,yp in zip(y_true,y_score,y_pred): w.writerow([int(yt),float(ys),int(yp)])
        print(f"[INFO] preds.csv 저장: {path}")
    except Exception as e:
        print(f"[WARN] preds.csv 저장 실패: {e}")

def _write_metrics_json_cls(run_root: Path, y_true: np.ndarray, y_score: np.ndarray, tau_default: float):
    fpr,tpr=_roc_curve(y_true,y_score); auroc=_trapz(fpr,tpr)
    rec,prec=_pr_curve(y_true,y_score); order=np.argsort(rec); auprc=_trapz(rec[order],prec[order])
    m_def=_bin_metrics(y_true,(y_score>=tau_default).astype(int)); tau_best,m_best=_best_threshold(y_true,y_score,"f1")
    metrics={"counts":{"n":int(y_true.size),"pos":int((y_true==1).sum()),"neg":int((y_true==0).sum())},
             "prevalence":float(np.mean(y_true==1)),"roc_auc":auroc,"pr_auc":auprc,
             "threshold_default":float(tau_default),"metrics_at_default":m_def,
             "threshold_best":float(tau_best),"metrics_at_best":m_best,
             "notes":{"best_strategy":"f1"},"generated_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    (run_root/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[INFO] metrics.json 저장: {run_root/'metrics.json'}")

def _write_summary_json_reg(run_root: Path, mse_list: List[float], mae_list: List[float]):
    summary={"avg_mse":float(np.mean(mse_list)) if mse_list else float("nan"),
             "avg_mae":float(np.mean(mae_list)) if mae_list else float("nan"),
             "n_windows":int(len(mse_list)),"generated_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    (run_root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[INFO] summary.json 저장: {run_root/'summary.json'}")

def _write_preds_csv_reg(run_root: Path,
                         units_list: List[int],
                         cycles_list: List[np.ndarray],
                         y_true_tensors: List[torch.Tensor],
                         y_pred_tensors: List[torch.Tensor]):
    """회귀용 per-window 예측 CSV 저장: after_pred.csv
       cols: unit, cycle, y_true, y_pred
    """
    try:
        import csv, torch
        # 펼치기
        rows = []
        for u, cyc_arr, yt_t, yp_t in zip(units_list, cycles_list, y_true_tensors, y_pred_tensors):
            yt = yt_t.detach().float().cpu().numpy().ravel()
            yp = yp_t.detach().float().cpu().numpy().ravel()
            assert len(yt) == len(yp) == len(cyc_arr), "shape mismatch while writing after_pred.csv"
            for c, yt1, yp1 in zip(cyc_arr.tolist(), yt.tolist(), yp.tolist()):
                rows.append((int(u), int(c), float(yt1), float(yp1)))

        path = run_root / "after_pred.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["unit","cycle","y_true","y_pred"])
            w.writerows(rows)
        print(f"[INFO] after_pred.csv 저장: {path} (rows={len(rows)})")
    except Exception as e:
        print(f"[WARN] after_pred.csv 저장 실패: {e}")

# ─────────────────────────── 메인 API ───────────────────────────

@torch.no_grad()
def eval_model(model,
               X: torch.Tensor,
               L: int,
               desc: str = "eval",
               task: str = "regress",
               bin_rule: str = "nonzero",
               bin_thr: float = 0.0,
               tau_for_cls: float = 0.5,
               heartbeat_sec: int = 5,
               indices: Optional[List[int]] = None,
               split_name: Optional[str] = None):
    """
    반환:
      회귀: (avg_mse, avg_mae, None, None)
      분류: (None, None, y_true_all, y_prob_all)
    """
    model.eval()
    dev = next(model.parameters()).device
    N, C, T = X.shape

    rows = indices if (indices and len(indices)>0) else list(range(N))
    last = time.time(); start = time.time()
    mse_list, mae_list = [], []
    prob_buf, true_buf = [], []

    # 회귀 CSV 저장용 버퍼(유닛/사이클/정답/예측)
    reg_units: List[int] = []
    reg_cycles: List[np.ndarray] = []
    reg_ytrue: List[torch.Tensor] = []
    reg_ypred: List[torch.Tensor] = []

    # 콘솔 표시에만 사용(저장 경로엔 영향 없음)
    tag = str(split_name if split_name else desc)

    def _sel_ch0(t: torch.Tensor) -> torch.Tensor:
        return t[:,0] if (hasattr(t,"dim") and t.dim()==2) else t

    def binarize(y_next: torch.Tensor)->torch.Tensor:
        if bin_rule=="nonzero": return (y_next!=0).float()
        if bin_rule=="gt": return (y_next>bin_thr).float()
        if bin_rule=="ge": return (y_next>=bin_thr).float()
        return (y_next!=0).float()

    metric_name = "avg MSE" if task=="regress" else f"F1@τ={tau_for_cls:.3f}"
    metric_val = float("nan")

    if USE_RICH:
        from rich.live import Live
        with Live(build_eval_table(f"{desc} ({tag})",0,len(rows),metric_name,float("nan"),None),
                  refresh_per_second=8, console=console) as live:
            for i,ridx in enumerate(rows):
                row=X[ridx].unsqueeze(0)
                Xw, Yw_reg, _, _ = build_windows_dataset(row, L)
                if Xw.numel()==0: continue
                Xw = Xw.to(dev, non_blocking=True)
                if task=="regress":
                    Yw=_sel_ch0(Yw_reg).to(dev,non_blocking=True)
                    pred,_=model(Xw); pred=_sel_ch0(pred)
                    assert pred.shape==Yw.shape
                    mse_list.append(F.mse_loss(pred,Yw).item()); mae_list.append(F.l1_loss(pred,Yw).item())
                    # 회귀 CSV 버퍼
                    reg_units.append(int(ridx))
                    # 타깃 cycle = [L, L+1, ..., L+W_row-1]
                    cycles = np.arange(L, L + Yw.shape[0], dtype=int)
                    reg_cycles.append(cycles)
                    reg_ytrue.append(Yw.detach().cpu())
                    reg_ypred.append(pred.detach().cpu())

                    metric_val=float(np.mean(mse_list)) if mse_list else float("nan")
                else:
                    Yw_sel=_sel_ch0(Yw_reg).to(dev,non_blocking=True); Yb=binarize(Yw_sel)
                    logits,_=model(Xw); logits=_sel_ch0(logits); probs=torch.sigmoid(logits)
                    assert Yb.dim()==1 and probs.dim()==1
                    prob_buf.append(probs.detach().cpu()); true_buf.append(Yb.detach().cpu())
                    yprob=torch.cat(prob_buf,0); ytrue=torch.cat(true_buf,0)
                    tp=((yprob>=tau_for_cls).float()*ytrue).sum().item()
                    fp=((yprob>=tau_for_cls).float()*(1-ytrue)).sum().item()
                    fn=(((yprob<tau_for_cls).float())*ytrue).sum().item()
                    prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); metric_val=2*prec*rec/(prec+rec+1e-8)
                now=time.time()
                if now-last>=heartbeat_sec:
                    done=i+1; rate=done/max(1,now-start); remain=(len(rows)-done)/rate if rate>0 else None
                    live.update(build_eval_table(f"{desc} ({tag})",done,len(rows),metric_name,metric_val,remain)); last=now
            live.update(build_eval_table(f"{desc} ({tag})",len(rows),len(rows),metric_name,metric_val,0.0))
    else:
        for ridx in rows:
            row=X[ridx].unsqueeze(0)
            Xw, Yw_reg, _, _ = build_windows_dataset(row, L)
            if Xw.numel()==0: continue
            Xw=Xw.to(dev,non_blocking=True)
            if task=="regress":
                Yw=_sel_ch0(Yw_reg).to(dev,non_blocking=True); pred,_=model(Xw); pred=_sel_ch0(pred)
                assert pred.shape==Yw.shape
                mse_list.append(F.mse_loss(pred,Yw).item()); mae_list.append(F.l1_loss(pred,Yw).item())
                # 회귀 CSV 버퍼
                reg_units.append(int(ridx))
                cycles = np.arange(L, L + Yw.shape[0], dtype=int)
                reg_cycles.append(cycles)
                reg_ytrue.append(Yw.detach().cpu())
                reg_ypred.append(pred.detach().cpu())
            else:
                Yw_sel=_sel_ch0(Yw_reg).to(dev,non_blocking=True); Yb=binarize(Yw_sel)
                logits,_=model(Xw); logits=_sel_ch0(logits); probs=torch.sigmoid(logits)
                assert Yb.dim()==1 and probs.dim()==1
                prob_buf.append(probs.detach().cpu()); true_buf.append(Yb.detach().cpu())

    # ───────────────── 저장 & 리턴 ─────────────────
    run_root = _run_root()
    save_allowed = _should_save(desc=str(desc), split_name=split_name)

    if task=="regress":
        avg_mse=float(np.mean(mse_list)) if mse_list else float("nan")
        avg_mae=float(np.mean(mae_list)) if mae_list else float("nan")
        if save_allowed:
            try:
                _write_summary_json_reg(run_root, mse_list, mae_list)
                # 추가: 회귀 예측 CSV 저장
                if reg_ytrue and reg_ypred:
                    _write_preds_csv_reg(run_root, reg_units, reg_cycles, reg_ytrue, reg_ypred)
            except Exception as e:
                print(f"[WARN] 회귀 요약/CSV 저장 실패: {e}")
        else:
            print("[INFO] (evaler) SAVE SKIPPED (regress) — BEFORE/검증 중간단계이거나 정책상 저장 안 함")
        return (avg_mse, avg_mae, None, None)
    else:
        yprob=torch.cat(prob_buf,0) if prob_buf else torch.empty(0)
        ytrue=torch.cat(true_buf,0) if true_buf else torch.empty(0)
        if ytrue.numel()>0 and yprob.numel()>0:
            if save_allowed:
                try:
                    y_true_np=_as_numpy(ytrue).astype(int); y_score_np=_as_numpy(yprob).astype(float)
                    _save_preds_csv(run_root, y_true_np, y_score_np, tau_default=tau_for_cls)
                    _write_metrics_json_cls(run_root, y_true_np, y_score_np, tau_default=tau_for_cls)
                except Exception as e:
                    print(f"[WARN] 분류 메트릭 저장 실패: {e}")
            else:
                print("[INFO] (evaler) SAVE SKIPPED (classify) — BEFORE/검증 중간단계이거나 정책상 저장 안 함")
        else:
            print("[WARN] 라벨/확률 비어 있음 → 지표/CSV 저장 생략")
        return None, None, ytrue, yprob
