# -*- coding: utf-8 -*-
"""
report_artifacts.py — Static HTML Report (images embedded via base64 data URIs)
+ Hover Zoom Lens (cursor-centered magnifier, no external deps)
- artifacts/**/ 에서 model.pt 들어있는 폴더만 런으로 간주
- 각 런의 핵심 지표를 카드 내 표로 바로 표시 (링크 없음)
- 각 런의 plots/after_*.png 이미지를 data URI로 HTML에 직접 포함 (외부 경로/링크 없음)

Usage:
  python tools/report_artifacts.py \
    --artifacts ./artifacts \
    --out ./artifacts/report_static.html \
    --title "AttentionProject Report" \
    --max-images 36
"""
import argparse
import base64
import csv
import html
import json
import math
from pathlib import Path
from datetime import datetime
import mimetypes

DEF_TITLE = "AttentionProject Report"
DEF_OUT   = "artifacts/report_static.html"

# ----------------------------- utils -----------------------------

def find_runs(artifacts_dir: Path):
    """Find run directories that contain model.pt (exclude report subtrees)."""
    runs = []
    for p in artifacts_dir.rglob("model.pt"):
        run_dir = p.parent
        if run_dir == artifacts_dir:
            continue
        parts = set(map(str.lower, run_dir.parts))
        if "report" in parts:
            continue
        runs.append(run_dir)
    runs = sorted(set(runs), key=lambda x: x.stat().st_mtime, reverse=True)
    return runs

def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def try_parse_metrics_from_preds(preds_csv: Path):
    """Fallback metrics from preds.csv (y_true, y_score?, y_pred@tau=...)."""
    if not preds_csv.exists():
        return None
    tp=tn=fp=fn=0
    has_score = False
    n=0; pos=0; neg=0
    try:
        with preds_csv.open("r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header: return None
            yi = header.index("y_true") if "y_true" in header else None
            si = header.index("y_score") if "y_score" in header else None
            pi = None
            for j, h in enumerate(header):
                if h.startswith("y_pred@tau="):
                    pi = j; break
            if yi is None or pi is None:
                return None
            for row in r:
                y = int(float(row[yi]))
                p = int(float(row[pi]))
                if si is not None:
                    has_score = True
                if y == 1:
                    pos += 1
                    if p == 1: tp += 1
                    else: fn += 1
                else:
                    neg += 1
                    if p == 1: fp += 1
                    else: tn += 1
                n += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        acc  = (tp + tn) / max(1, n)
        prev = pos / max(1, n)
        return {
            "counts": {"n": n, "pos": pos, "neg": neg},
            "prevalence": prev,
            "metrics_at_default": {
                "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn
            }
        }
    except Exception:
        return None

def collect_run(run_dir: Path, max_images: int):
    """Collect metrics & after_* images (list of Paths)."""
    plots = sorted((run_dir / "plots").glob("after_*.png"))
    if max_images > 0:
        plots = plots[:max_images]

    metrics_json = run_dir / "metrics.json"
    summary_json = run_dir / "summary.json"
    preds_csv    = run_dir / "preds.csv"

    kind = "unknown"
    metrics = None

    if metrics_json.exists():  # classification (preferred)
        raw = safe_read_json(metrics_json)
        if isinstance(raw, dict):
            kind = "classify"
            metrics = {
                "n": (raw.get("counts") or {}).get("n"),
                "pos": (raw.get("counts") or {}).get("pos"),
                "neg": (raw.get("counts") or {}).get("neg"),
                "prev": raw.get("prevalence"),
                "auroc": raw.get("roc_auc"),
                "auprc": raw.get("pr_auc"),
                "tau_def": raw.get("threshold_default"),
                "m_def": raw.get("metrics_at_default") or {},
                "tau_best": raw.get("threshold_best"),
                "m_best": raw.get("metrics_at_best") or {},
                "generated_at": raw.get("generated_at"),
            }
    elif summary_json.exists():  # regression
        raw = safe_read_json(summary_json)
        if isinstance(raw, dict):
            kind = "regress"
            metrics = {
                "mse": raw.get("avg_mse"),
                "mae": raw.get("avg_mae"),
                "nwin": raw.get("n_windows"),
                "generated_at": raw.get("generated_at"),
            }
    elif preds_csv.exists():  # classification minimal (fallback)
        raw = try_parse_metrics_from_preds(preds_csv)
        if isinstance(raw, dict):
            kind = "classify"
            metrics = {
                "n": (raw.get("counts") or {}).get("n"),
                "pos": (raw.get("counts") or {}).get("pos"),
                "neg": (raw.get("counts") or {}).get("neg"),
                "prev": raw.get("prevalence"),
                "tau_def": None,
                "m_def": raw.get("metrics_at_default") or {},
                "auroc": None, "auprc": None,
                "tau_best": None, "m_best": {},
                "generated_at": None,
            }

    return kind, metrics, plots

def fmt(v, nd=4, na="-"):
    try:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return na
        return f"{float(v):.{nd}f}"
    except Exception:
        return na

def image_to_data_uri(path: Path) -> str:
    """Read image and return a data URI string."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{b64}"

# --------------------------- HTML build ---------------------------

def build_html(runs: list[Path], title: str, max_images: int) -> str:
    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;background:#0b0d10;color:#e7ecf3;margin:0}
    header{padding:20px 24px;border-bottom:1px solid #1d232f;background:#0f1116;position:sticky;top:0}
    h1{font-size:20px;margin:0}
    main{padding:18px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
    .card{background:#121621;border:1px solid #1e293b;border-radius:14px;padding:14px}
    .card h2{font-size:16px;margin:0 0 8px 0;word-break:break-all}
    .meta{font-size:12px;color:#9fb1c5;margin-bottom:8px}
    table{width:100%;border-collapse:collapse;margin:8px 0}
    th,td{border-bottom:1px solid #233044;padding:6px 4px;font-size:13px;text-align:right}
    th:first-child,td:first-child{text-align:left;color:#9fb1c5}
    .thumbs{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
    .thumbs img{max-width:100%;height:auto;border-radius:8px;border:1px solid #233044}
    .empty{opacity:0.8}
    footer{padding:14px;color:#8193a8;font-size:12px;text-align:center}
    /* Zoom lens */
    .zoom-lens{position:fixed;left:0;top:0;width:220px;height:220px;border-radius:50%;
      border:1px solid #334155; box-shadow:0 4px 16px rgba(0,0,0,.35); pointer-events:none;
      background-repeat:no-repeat; background-position:center; background-size:contain;
      display:none; z-index:9999}
    """

    # 간단한 마우스 루페 JS (cursor-centered zoom, wheel로 배율 조절)
    js = """
    <script>
    (function(){
      const lens = document.createElement('div');
      lens.className = 'zoom-lens';
      document.body.appendChild(lens);

      let active = null;
      let zoom = 2.5;               // 기본 배율
      const Z_MIN = 1.5, Z_MAX = 6; // 배율 범위

      function onEnter(e){
        const img = e.currentTarget;
        active = img;
        lens.style.display = 'block';
        lens.style.backgroundImage = `url(${img.src})`;
      }

      function onLeave(){
        active = null;
        lens.style.display = 'none';
      }

      function onMove(e){
        if(!active) return;
        const rect = active.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        // 원본 대비 배율
        const rx = active.naturalWidth / active.clientWidth;
        const ry = active.naturalHeight / active.clientHeight;

        // 렌즈 위치 (커서 중심)
        const lw = lens.offsetWidth, lh = lens.offsetHeight;
        lens.style.left = (e.clientX - lw/2) + 'px';
        lens.style.top  = (e.clientY - lh/2) + 'px';

        // 배경 이미지 스케일
        lens.style.backgroundSize = (active.naturalWidth*zoom)+'px ' + (active.naturalHeight*zoom)+'px';

        // 배경 위치: 커서 지점이 렌즈 중앙에 오도록
        const bgX = -(cx*rx*zoom - lw/2);
        const bgY = -(cy*ry*zoom - lh/2);
        lens.style.backgroundPosition = bgX+'px '+bgY+'px';
      }

      function onWheel(e){
        if(!active) return;
        e.preventDefault();
        const delta = Math.sign(e.deltaY);
        if (delta < 0) zoom *= 1.1;    // up → 확대
        else zoom /= 1.1;              // down → 축소
        zoom = Math.max(Z_MIN, Math.min(Z_MAX, zoom));
        onMove(e);
      }

      // 모든 이미지에 바인딩
      function bindZoom(img){
        img.addEventListener('mouseenter', onEnter);
        img.addEventListener('mouseleave', onLeave);
        img.addEventListener('mousemove', onMove);
        img.addEventListener('wheel', onWheel, {passive:false});
      }

      document.querySelectorAll('img.zoomable').forEach(bindZoom);
    })();
    </script>
    """

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cards = []

    for run_dir in runs:
        run_name = run_dir.name
        when = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        kind, md, imgs = collect_run(run_dir, max_images)

        # Header
        card = [f'<div class="card">', f"<h2>{html.escape(run_name)}</h2>", f'<div class="meta">{html.escape(when)}</div>']

        # Metrics table (embedded, no links)
        if kind == "classify" and isinstance(md, dict):
            n = md.get("n"); pos = md.get("pos"); neg = md.get("neg")
            prev = md.get("prev")
            auroc = md.get("auroc"); auprc = md.get("auprc")
            tdef = md.get("tau_def"); mdef = md.get("m_def") or {}
            tbest = md.get("tau_best"); mbest = md.get("m_best") or {}
            card.append("<table>")
            card.append("<tr><th>Type</th><td>Classification</td></tr>")
            if n is not None:
                card.append(f"<tr><th>Samples (N / + / -)</th><td>{int(n)} / {int(pos or 0)} / {int(neg or 0)}</td></tr>")
            if prev is not None:
                card.append(f"<tr><th>Prevalence (+)</th><td>{fmt(100*prev,2)}%</td></tr>")
            if auroc is not None or auprc is not None:
                card.append(f"<tr><th>AUROC / AUPRC</th><td>{fmt(auroc,4)} / {fmt(auprc,4)}</td></tr>")
            if mdef:
                row = f"F1 {fmt(mdef.get('f1'))} · Acc {fmt(mdef.get('accuracy'))} · P {fmt(mdef.get('precision'))} · R {fmt(mdef.get('recall'))}"
                if tdef is not None: row = f"τ={fmt(tdef,3)} · " + row
                card.append(f"<tr><th>At default</th><td>{row}</td></tr>")
            if mbest:
                row = f"F1 {fmt(mbest.get('f1'))} · Acc {fmt(mbest.get('accuracy'))} · P {fmt(mbest.get('precision'))} · R {fmt(mbest.get('recall'))}"
                if tbest is not None: row = f"τ={fmt(tbest,3)} · " + row
                card.append(f"<tr><th>At best F1</th><td>{row}</td></tr>")
            card.append("</table>")

        elif kind == "regress" and isinstance(md, dict):
            card.append("<table>")
            card.append("<tr><th>Type</th><td>Regression</td></tr>")
            card.append(f"<tr><th>avg MSE</th><td>{fmt(md.get('mse'),6)}</td></tr>")
            card.append(f"<tr><th>avg MAE</th><td>{fmt(md.get('mae'),6)}</td></tr>")
            if md.get("nwin") is not None:
                card.append(f"<tr><th>#Windows</th><td>{int(md.get('nwin'))}</td></tr>")
            card.append("</table>")
        else:
            card.append('<div class="meta empty">No metrics found (metrics.json/summary.json/preds.csv)</div>')

        # Thumbnails (embedded data URIs) — zoomable 클래스를 부여
        if imgs:
            card.append('<div class="thumbs">')
            for im in imgs:
                try:
                    data_uri = image_to_data_uri(im)
                    card.append(f'<img class="zoomable" src="{data_uri}" alt="{html.escape(im.name)}">')
                except Exception:
                    card.append(f'<div class="meta empty">[image read failed] {html.escape(im.name)}</div>')
            card.append("</div>")
        else:
            card.append('<div class="meta empty">No images found under plots/after_*.png</div>')

        card.append("</div>")
        cards.append("".join(card))

    if not cards:
        cards.append('<div class="card empty"><div class="meta">No runs found (folders with model.pt)</div></div>')

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta http-equiv="x-ua-compatible" content="ie=edge">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1></header>
<main>
  <div class="grid">
    {''.join(cards)}
  </div>
</main>
<footer>Generated at {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</footer>
{js}
</body>
</html>
"""
    return html_doc

# ----------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts root to scan")
    ap.add_argument("--out", default=DEF_OUT, help="Output HTML file path")
    ap.add_argument("--title", default=DEF_TITLE, help="Report title")
    ap.add_argument("--max-images", type=int, default=36, help="Max images per run")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts).resolve()
    out_html = Path(args.out).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    runs = find_runs(artifacts_dir)
    html_text = build_html(runs, args.title, args.max_images)
    out_html.write_text(html_text, encoding="utf-8")
    print(f"[OK] Wrote static (embedded) report: {out_html}")

if __name__ == "__main__":
    main()
