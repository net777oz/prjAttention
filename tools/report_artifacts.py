# -*- coding: utf-8 -*-
"""
report_artifacts.py — artifacts/ 이하 실험 폴더를 스캔해 단일 HTML 리포트 생성.
- AttentionProject evaler.py 산출물(./artifacts/llm_ts_eval/<tag>/...)에 최적화
- 분류/회귀 자동 감지 (metrics.json ↔ summary.json)
- JSON이 없어도 이미지만으로 카드 표시
- 상대 경로 자동 계산(리포트 위치와 무관)
- 검색/필터/정렬 UI + 간단 규칙 기반 분석 문구

사용 예:
  python tools/report_artifacts.py \
    --artifacts ./artifacts \
    --out ./artifacts/report/index.html \
    --max-images 24 \
    --sort-by score \
    --verbose
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".svg")

# 우선 탐색 파일 (분류/회귀)
CLS_METRICS = ["metrics.json", "eval/metrics.json"]
REG_SUMMARY = ["summary.json", "eval/summary.json"]

# 분류 스코어 선호도(높을수록 좋은 지표)
CLS_SCORE_KEYS = ["f1", "macro.f1", "metrics_at_best.f1", "metrics_at_default.f1",
                  "pr_auc", "roc_auc", "accuracy", "precision", "recall"]

# 회귀 스코어는 "낮을수록 좋은" MSE/MAE → 내부적으로 -값으로 변환해 정렬
REG_SCORE_KEYS = ["avg_mse", "avg_mae"]


def human_dt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def safe_read_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def flatten(d, prefix=""):
    flat = {}
    if isinstance(d, dict):
        for k, v in d.items():
            nk = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten(v, nk))
            else:
                flat[nk] = v
    return flat


def detect_task(metrics_json: dict, summary_json: dict):
    """분류/회귀/unknown"""
    if metrics_json and any(k in metrics_json for k in ["roc_auc", "pr_auc", "metrics_at_best", "metrics_at_default"]):
        return "classify"
    if summary_json and any(k in summary_json for k in ["avg_mse", "avg_mae"]):
        return "regress"
    return "unknown"


def pick_score(task: str, metrics_json: dict, summary_json: dict):
    """(score_key, score_value, display_value) 반환. 정렬은 value 기준(내림차순)."""
    if task == "classify" and metrics_json:
        flat = flatten(metrics_json)
        for key in CLS_SCORE_KEYS:
            if key in flat and isinstance(flat[key], (int, float)):
                val = float(flat[key])
                return key, val, f"{val:.4f}"
        # 마지막으로 숫자 아무거나
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                val = float(v); return k, val, f"{val:.4f}"
        return None, None, "-"
    if task == "regress" and summary_json:
        flat = flatten(summary_json)
        for key in REG_SCORE_KEYS:
            if key in flat and isinstance(flat[key], (int, float)):
                val = float(flat[key])
                # 낮을수록 좋은 지표 → 내부 정렬은 -val
                return key, -val, f"{val:.6f}"  # 표시값은 원래 값
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                val = float(v); return k, -val, f"{val:.6f}"
        return None, None, "-"
    return None, None, "-"


def collect_images(run_dir: Path, max_images: int):
    files = []
    for ext in IMG_EXT:
        files.extend(run_dir.glob(f"*{ext}"))
    # 이름 가중치(우선순위) + 최신순
    def weight(p: Path):
        name = p.name.lower()
        pri = ["roc", "pr", "precision_recall", "hist", "confusion", "resid", "pred_vs_true", "loss", "metric"]
        w = 0
        for i, kw in enumerate(pri):
            if kw in name:
                w += (len(pri) - i) * 10
        return w
    files.sort(key=lambda x: (-weight(x), -x.stat().st_mtime, x.name.lower()))
    return files[:max_images]


def first_existing(run_dir: Path, names):
    for n in names:
        p = run_dir / n
        if p.exists() and p.is_file():
            return p
    return None


def is_candidate_dir(p: Path):
    if not p.is_dir():
        return False
    if p.name in {"report", "logs", "cache", "__pycache__", ".git", "tmp", "temp"}:
        return False
    # evaler 기본 산출이 있는지 러프 체크
    if any((p / n).exists() for n in CLS_METRICS + REG_SUMMARY):
        return True
    if any(p.glob("*.png")):
        return True
    return False


def rel_from_out(path: Path, out_dir: Path):
    """out_dir 기준 상대 경로(실패 시 원본 경로)"""
    try:
        return os.path.relpath(path, start=out_dir)
    except Exception:
        return path.as_posix()


def collect_runs(artifacts_dir: Path, out_dir: Path, max_images: int, verbose: bool):
    runs = []
    # 우선 llm_ts_eval 중심으로 스캔
    primary = artifacts_dir / "llm_ts_eval"
    scan_roots = [primary] if primary.exists() else [artifacts_dir]
    for root in scan_roots:
        for p in root.rglob("*"):
            if not is_candidate_dir(p):
                continue
            # 파일 로딩
            mfile = first_existing(p, CLS_METRICS)
            sfile = first_existing(p, REG_SUMMARY)
            metrics = safe_read_json(mfile) if mfile else None
            summary = safe_read_json(sfile) if sfile else None
            task = detect_task(metrics, summary)
            imgs = collect_images(p, max_images) if any(p.glob("*")) else []
            # 점수
            score_key, score_val, score_disp = pick_score(task, metrics, summary)
            # 카드 타이틀은 artifacts 기준 상대경로
            try:
                title_rel = p.relative_to(artifacts_dir).as_posix()
            except Exception:
                title_rel = p.as_posix()
            # 이미지 상대 경로(out 기준)
            rel_imgs = [rel_from_out(img, out_dir) for img in imgs]
            # 파일 상대 경로(out 기준)
            metrics_rel = rel_from_out(mfile, out_dir) if mfile else None
            summary_rel = rel_from_out(sfile, out_dir) if sfile else None

            runs.append({
                "dir": p, "rel": title_rel, "mtime": p.stat().st_mtime,
                "task": task,
                "metrics_file": metrics_rel, "summary_file": summary_rel,
                "metrics": metrics, "summary": summary,
                "score_key": score_key, "score_val": score_val, "score_disp": score_disp,
                "images": rel_imgs,
            })

    # 기본 정렬: 최신순. score 정렬은 호출부에서 처리
    runs.sort(key=lambda r: (-r["mtime"], r["rel"]))
    if verbose:
        print(f"[INFO] collected {len(runs)} runs under {artifacts_dir}")
    return runs


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_assets(out_html: Path):
    css = r"""
:root { --bg:#0b0f14; --card:#111827; --muted:#94a3b8; --text:#e5e7eb; --accent:#60a5fa; --good:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Ubuntu,Apple Color Emoji}
a{color:var(--accent);text-decoration:none}
.container{max-width:1280px;margin:24px auto;padding:0 16px 48px}
.header{display:flex;gap:16px;align-items:center;justify-content:space-between;margin:24px 0}
.h-title{font-weight:700;font-size:24px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
input,select{background:#0f172a;color:var(--text);border:1px solid #1f2937;border-radius:10px;padding:8px 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
.card{background:var(--card);border:1px solid #1f2937;border-radius:16px;overflow:hidden;display:flex;flex-direction:column}
.card .head{padding:14px 14px 8px;border-bottom:1px solid #1f2937}
.card .title{font-weight:700;font-size:16px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.badge{font-size:12px;padding:2px 8px;border-radius:999px;background:#0b1220;border:1px solid #1f2937;color:var(--muted)}
.meta{padding:0 14px 10px;color:var(--muted);font-size:12px;display:flex;gap:10px;flex-wrap:wrap}
.metrics{padding:0 14px 10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#cbd5e1}
.analysis{padding:0 14px 12px;color:#dbeafe}
.media{display:grid;grid-template-columns:1fr;gap:4px;padding:0 0 10px}
.media img{width:100%;height:auto;display:block;background:#0b1220}
.table{width:100%;border-collapse:collapse;margin-top:8px}
.table th,.table td{border-bottom:1px solid #1f2937;padding:6px 8px;text-align:left}
.footer{margin-top:24px;color:var(--muted);font-size:12px}
.score-pill{font-weight:700}
.score-good{color:var(--good)}
.score-warn{color:var(--warn)}
.score-bad{color:var(--bad)}
.search-empty{padding:24px;text-align:center;color:var(--muted)}
"""
    js = r"""
function $(q, el=document){return el.querySelector(q)}
function $all(q, el=document){return [...el.querySelectorAll(q)]}
function normalize(s){return (s||"").toLowerCase()}

function applyFilters(){
  const q = normalize($("#filter-q").value)
  const min = parseFloat($("#min-score").value||"")
  const key = $("#score-key").value
  const mode = $("#sort-mode").value

  const cards = $all(".card")
  cards.forEach(card=>{
    const name = normalize(card.dataset.name)
    const scoreKey = card.dataset.scoreKey
    const scoreVal = parseFloat(card.dataset.scoreVal || "NaN")
    const hitQ = !q || name.includes(q)
    const hitMin = isNaN(min) || (!isNaN(scoreVal) && scoreVal >= min)
    const hitKey = !key || !scoreKey || scoreKey.endsWith(key) || scoreKey===key
    card.style.display = (hitQ && hitMin && hitKey) ? "" : "none"
  })

  // 정렬
  const grid = $(".grid")
  const shown = $all(".card").filter(c=>c.style.display!=="none")
  shown.sort((a,b)=>{
    const am = parseFloat(a.dataset.mtime), bm = parseFloat(b.dataset.mtime)
    const as = parseFloat(a.dataset.scoreVal || "NaN"), bs = parseFloat(b.dataset.scoreVal || "NaN")
    if(mode==="mtime"){ return (bm - am) }
    if(mode==="score"){
      if(!isNaN(bs) && !isNaN(as)) return (bs - as)
      if(!isNaN(bs)) return -1
      if(!isNaN(as)) return 1
      return (bm - am)
    }
    return 0
  })
  shown.forEach(c=>grid.appendChild(c))

  const any = shown.length>0
  $("#empty").style.display = any ? "none" : ""
}

function init(){
  $all("#filter-q,#min-score,#score-key,#sort-mode").forEach(el=>el.addEventListener("input", applyFilters))
  applyFilters()
}
document.addEventListener("DOMContentLoaded", init)
"""
    ensure_parent(out_html)
    css_path = out_html.parent / "style.css"
    js_path = out_html.parent / "app.js"
    if not css_path.exists():
        css_path.write_text(css, encoding="utf-8")
    if not js_path.exists():
        js_path.write_text(js, encoding="utf-8")


def metric_cell(obj: dict, keys):
    if not obj:
        return "-"
    flat = flatten(obj)
    for k in keys:
        if k in flat and isinstance(flat[k], (int, float)):
            v = float(flat[k])
            return f"{v:.6f}" if "mse" in k or "mae" in k else f"{v:.4f}"
    return "-"


def analyze(task: str, metrics: dict, summary: dict):
    if task == "classify" and metrics:
        f = flatten(metrics)
        f1 = f.get("metrics_at_best.f1") or f.get("metrics_at_default.f1") or f.get("f1")
        auprc = f.get("pr_auc"); auroc = f.get("roc_auc")
        prec = (f.get("metrics_at_best.precision") or f.get("precision"))
        rec  = (f.get("metrics_at_best.recall") or f.get("recall"))
        notes = []
        if f1 is not None:
            if f1 >= 0.8: notes.append(f"F1={f1:.3f} (우수).")
            elif f1 >= 0.6: notes.append(f"F1={f1:.3f} (보통) — 임계값/가중치 개선 여지.")
            else: notes.append(f"F1={f1:.3f} (낮음) — 불균형/특징 확장 검토.")
        if prec is not None and rec is not None and abs(prec-rec) > 0.2:
            tilt = "정밀도>재현율" if prec > rec else "재현율>정밀도"
            notes.append(f"정밀-재현 불균형({tilt}, Δ≈{abs(prec-rec):.2f}). 임계값/가중치 재조정 권장.")
        if auprc is not None: notes.append(f"AUPRC={auprc:.3f}.")
        if auroc is not None: notes.append(f"ROC-AUC={auroc:.3f}.")
        return " ".join(notes) if notes else "핵심 지표 부족 — 기본 분석 생략."
    if task == "regress" and summary:
        avg_mse = summary.get("avg_mse"); avg_mae = summary.get("avg_mae")
        notes = []
        if avg_mse is not None: notes.append(f"MSE={avg_mse:.6f}.")
        if avg_mae is not None: notes.append(f"MAE={avg_mae:.6f}.")
        return " ".join(notes) if notes else "요약 지표 부족 — 기본 분석 생략."
    return "지표 파일이 없어 기본 분석을 생략합니다. (이미지/폴더만 표시)"


def render_html(out_html: Path, artifacts_dir: Path, runs: list):
    write_assets(out_html)
    # 헤더
    head = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Artifacts Report</title>
<link rel="stylesheet" href="style.css" />
</head>
<body>
<div class="container">
  <div class="header">
    <div class="h-title">Artifacts Report</div>
    <div class="controls">
      <input id="filter-q" placeholder="실험명 필터..." />
      <input id="min-score" placeholder="최소 스코어 (예: 0.7)" />
      <select id="score-key">
        <option value="">score key 필터</option>
        <option>f1</option><option>pr_auc</option><option>roc_auc</option><option>accuracy</option>
        <option>precision</option><option>recall</option>
        <option>avg_mse</option><option>avg_mae</option>
      </select>
      <select id="sort-mode">
        <option value="mtime">정렬: 최신순</option>
        <option value="score">정렬: 스코어순</option>
      </select>
    </div>
  </div>
"""
    # 요약 테이블
    cols = ["run", "task", "mtime", "score_key", "score", "f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall", "avg_mse", "avg_mae"]
    table = [
        '<div class="card"><div class="head"><div class="title">요약 테이블 <span class="badge">상위 지표 미리보기</span></div></div>',
        '<div style="overflow:auto"><table class="table"><thead><tr>',
        *[f"<th>{c}</th>" for c in cols],
        "</tr></thead><tbody>"
    ]
    body_rows = []
    for r in runs:
        score_html = "-"
        if r["score_disp"] != "-" and r["score_val"] is not None:
            # 분류(높을수록 좋음) / 회귀(낮을수록 좋음) 시각 힌트
            sign = 1 if r["task"] == "classify" else -1
            raw = float(r["score_disp"]) if r["task"] == "classify" else -float(r["score_val"])
            cls = "score-good" if (sign*float(r["score_val"]) >= 0.8) else ("score-warn" if (sign*float(r["score_val"]) >= 0.6) else "score-bad") if r["task"]=="classify" else ""
            score_html = f'<span class="score-pill {cls}">{r["score_disp"]}</span>'
        f_f1 = metric_cell(r["metrics"], ["metrics_at_best.f1","metrics_at_default.f1","f1","macro.f1"])
        f_pr = metric_cell(r["metrics"], ["pr_auc"])
        f_roc= metric_cell(r["metrics"], ["roc_auc"])
        f_acc= metric_cell(r["metrics"], ["accuracy"])
        f_prec=metric_cell(r["metrics"], ["metrics_at_best.precision","precision"])
        f_rec= metric_cell(r["metrics"], ["metrics_at_best.recall","recall"])
        f_mse= metric_cell(r["summary"], ["avg_mse"])
        f_mae= metric_cell(r["summary"], ["avg_mae"])
        cells = [r["rel"], r["task"], human_dt(r["mtime"]), r["score_key"] or "-", score_html,
                 f_f1, f_pr, f_roc, f_acc, f_prec, f_rec, f_mse, f_mae]
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    table.extend(body_rows)
    table.append("</tbody></table></div></div>")

    # 카드 그리드
    grid_head = '<div id="empty" class="search-empty" style="display:none">조건에 맞는 실험이 없습니다.</div><div class="grid">'
    cards = []
    for r in runs:
        # 점수 표시
        score_html = r["score_disp"] if r["score_disp"] != "-" else "-"
        if r["task"] == "classify" and r["score_val"] is not None:
            val = float(r["score_val"])
            cls = "score-good" if val >= 0.8 else ("score-warn" if val >= 0.6 else "score-bad")
            score_html = f'<span class="score-pill {cls}">{r["score_disp"]}</span>'
        # 메타
        meta_bits = [f"<span>폴더: <code>{r['rel']}</code></span>",
                     f"<span>수정: {human_dt(r['mtime'])}</span>",
                     f"<span>task: <b>{r['task']}</b></span>"]
        if r["metrics_file"]:
            meta_bits.append(f"<span>metrics: <code>{r['metrics_file']}</code></span>")
        if r["summary_file"]:
            meta_bits.append(f"<span>summary: <code>{r['summary_file']}</code></span>")
        if r["score_key"]:
            meta_bits.append(f"<span>score_key: <code>{r['score_key']}</code> {score_html}</span>")

        # metrics preview(상위 10개)
        metrics_preview = ""
        preview_items = []
        if r["metrics"]:
            fl = flatten(r["metrics"])
            for k in sorted(fl.keys())[:10]:
                v = fl[k]; preview_items.append((k, v))
        elif r["summary"]:
            fl = flatten(r["summary"])
            for k in sorted(fl.keys())[:10]:
                v = fl[k]; preview_items.append((k, v))
        if preview_items:
            metrics_preview = "<div class='metrics'><ul style='padding-left:18px;margin:6px 0'>"
            for k, v in preview_items:
                if isinstance(v, float):
                    metrics_preview += f"<li><b>{k}</b>: {v:.6f}</li>"
                else:
                    metrics_preview += f"<li><b>{k}</b>: {v}</li>"
            metrics_preview += "</ul></div>"

        # 분석
        analysis = analyze(r["task"], r["metrics"], r["summary"])
        # 이미지
        imgs_html = ""
        if r["images"]:
            imgs_html = "<div class='media'>" + "\n".join(
                f"<img loading='lazy' src='{img}' alt='{img}' />" for img in r["images"]
            ) + "</div>"

        cards.append(f"""
<div class="card" data-name="{r['rel']}" data-mtime="{r['mtime']}" data-score-key="{r['score_key'] or ''}" data-score-val="{r['score_val'] if r['score_val'] is not None else ''}">
  <div class="head"><div class="title"><span>{r['rel']}</span><span class="badge">{human_dt(r['mtime'])}</span></div></div>
  <div class="meta">{' · '.join(meta_bits)}</div>
  {metrics_preview}
  <div class="analysis">▶ 분석: {analysis}</div>
  {imgs_html}
</div>
""")

    tail = f"""</div>
  <div class="footer">생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · 자동 생성 리포트</div>
</div>
<script src="app.js"></script>
</body>
</html>"""

    html = head + "\n".join(table) + grid_head + "\n".join(cards) + tail
    ensure_parent(out_html)
    out_html.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="./artifacts", help="실험 아티팩트 루트")
    ap.add_argument("--out", type=str, default="./artifacts/report/index.html", help="생성될 HTML 파일 경로")
    ap.add_argument("--max-images", type=int, default=24, help="실험별 최대 이미지 수")
    ap.add_argument("--sort-by", type=str, choices=["mtime", "score"], default="mtime", help="초기 정렬 기준")
    ap.add_argument("--verbose", action="store_true", help="수집 로그 출력")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts).resolve()
    out_html = Path(args.out).resolve()
    out_dir = out_html.parent

    if not artifacts_dir.exists():
        raise SystemExit(f"[ERR] artifacts 폴더가 없습니다: {artifacts_dir}")

    runs = collect_runs(artifacts_dir, out_dir, max_images=args.max_images, verbose=args.verbose)

    # 초기 정렬(옵션 반영)
    if args.sort_by == "score":
        runs.sort(key=lambda r: (-(r["score_val"] if isinstance(r["score_val"], (int, float)) else float('-inf')),
                                 -r["mtime"], r["rel"]))

    render_html(out_html, artifacts_dir, runs)
    print(f"[OK] Report written → {out_html}")
    print(f"     Open file://{out_html}")


if __name__ == "__main__":
    main()
