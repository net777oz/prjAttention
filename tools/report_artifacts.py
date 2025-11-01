# -*- coding: utf-8 -*-
"""
report_artifacts.py — Static HTML Report (images embedded via base64 data URIs)
+ Hover Zoom Lens (cursor-centered magnifier, no external deps)
+ NEW: Precision/Recall vs Threshold(τ) mini chart (inline SVG, uses preds.csv if available)
- artifacts/**/ 에서 model.pt 들어있는 폴더만 런으로 간주
- 각 런의 핵심 지표를 카드 내 표로 바로 표시 (링크 없음)
- 각 런의 plots/after_*.png 이미지를 data URI로 HTML에 직접 포함

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
import numpy as np

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
                y = int(float(row[yi])); p = int(float(row[pi]))
                if si is not None: has_score = True
                if y == 1:
                    pos += 1; tp += 1 if p == 1 else 0; fn += 1 if p == 0 else 0
                else:
                    neg += 1; fp += 1 if p == 1 else 0; tn += 1 if p == 0 else 0
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

def load_preds_arrays(preds_csv: Path):
    """
    Read y_true and y_score from preds.csv (if exists). Returns (yt, ys) or (None, None).
    """
    if not preds_csv.exists():
        return None, None
    try:
        with preds_csv.open("r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header or ("y_true" not in header) or ("y_score" not in header):
                return None, None
            yi = header.index("y_true")
            si = header.index("y_score")
            ys = []
            yt = []
            for row in r:
                try:
                    yt.append(int(float(row[yi])))
                    ys.append(float(row[si]))
                except Exception:
                    continue
        if not ys or not yt:
            return None, None
        yt = np.asarray(yt, dtype=np.int32)
        ys = np.asarray(ys, dtype=np.float32)
        return yt, ys
    except Exception:
        return None, None

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

# ---------- P/R vs τ (no matplotlib; inline SVG) ----------

def pr_recall_f1_vs_tau(yt: np.ndarray, ys: np.ndarray, n_points: int = 201):
    """
    Return dict with arrays: taus, precision, recall, f1, best_tau, best_f1
    """
    if yt.size == 0 or ys.size == 0:
        return None
    taus = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    P = float((yt == 1).sum())
    N = float((yt == 0).sum())
    prec = np.zeros_like(taus)
    rec  = np.zeros_like(taus)
    f1   = np.zeros_like(taus)

    for i, t in enumerate(taus):
        yhat = (ys >= t).astype(np.int32)
        tp = float(((yhat == 1) & (yt == 1)).sum())
        fp = float(((yhat == 1) & (yt == 0)).sum())
        fn = float(((yhat == 0) & (yt == 1)).sum())
        prec[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[i]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i]   = (2 * prec[i] * rec[i]) / (prec[i] + rec[i]) if (prec[i] + rec[i]) > 0 else 0.0

    j = int(np.nanargmax(f1)) if np.any(np.isfinite(f1)) else 0
    return {
        "taus": taus.tolist(),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "best_tau": float(taus[j]),
        "best_f1": float(f1[j]),
    }

def series_to_svg(prc: list[float], rec: list[float], taus: list[float], best_tau: float,
                  width: int = 360, height: int = 180, pad: int = 28) -> str:
    """
    Draw two lines (Precision, Recall) vs τ into an inline SVG string.
    y-range [0,1]. x-range [0,1].
    """
    def path_from_series(vals, stroke):
        pts = []
        for i, v in enumerate(vals):
            x = pad + (width - 2*pad) * (taus[i])
            y = height - pad - (height - 2*pad) * max(0.0, min(1.0, v))
            pts.append(f"{x:.1f},{y:.1f}")
        return f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="{" ".join(pts)}"/>'

    # axes
    ax = []
    ax.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#334155" stroke-width="1"/>')
    ax.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#334155" stroke-width="1"/>')
    # ticks (0,0.5,1)
    for tx in (0.0, 0.5, 1.0):
        x = pad + (width - 2*pad) * tx
        ax.append(f'<line x1="{x:.1f}" y1="{height-pad}" x2="{x:.1f}" y2="{height-pad+4}" stroke="#475569" />')
        ax.append(f'<text x="{x:.1f}" y="{height-pad+16}" font-size="10" fill="#94a3b8" text-anchor="middle">{tx:.1f}</text>')
    for ty in (0.0, 0.5, 1.0):
        y = height - pad - (height - 2*pad) * ty
        ax.append(f'<line x1="{pad-4}" y1="{y:.1f}" x2="{pad}" y2="{y:.1f}" stroke="#475569" />')
        ax.append(f'<text x="6" y="{y+3:.1f}" font-size="10" fill="#94a3b8">{ty:.1f}</text>')

    # series
    p_path = path_from_series(prc, "#60a5fa")   # Precision (blue-ish)
    r_path = path_from_series(rec, "#f97316")   # Recall (orange)

    # best τ marker
    bx = pad + (width - 2*pad) * best_tau
    marker = f'<line x1="{bx:.1f}" y1="{pad}" x2="{bx:.1f}" y2="{height-pad}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>'

    title = '<text x="50%" y="12" text-anchor="middle" font-size="12" fill="#e2e8f0">Precision / Recall vs τ</text>'
    legend = (
        '<rect x="{x}" y="{y}" width="8" height="2" fill="{c}"/>'
    )
    leg = (
        '<g transform="translate({},{})">'
        '<rect x="0" y="0" width="8" height="2" fill="#60a5fa"/><text x="12" y="3" font-size="10" fill="#cbd5e1">Precision</text>'
        '<rect x="80" y="0" width="8" height="2" fill="#f97316"/><text x="92" y="3" font-size="10" fill="#cbd5e1">Recall</text>'
        '<text x="180" y="3" font-size="10" fill="#cbd5e1">τ*</text>'
        '</g>'
    ).format(pad, 6)

    svg = f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Precision/Recall vs threshold">'
    svg += "".join(ax) + p_path + r_path + marker + title + leg + "</svg>"
    return svg

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
    .mini{margin-top:8px;border-top:1px solid #1f2937;padding-top:8px}
    footer{padding:14px;color:#8193a8;font-size:12px;text-align:center}
    /* Zoom lens */
    .zoom-lens{position:fixed;left:0;top:0;width:220px;height:220px;border-radius:50%;
      border:1px solid #334155; box-shadow:0 4px 16px rgba(0,0,0,.35); pointer-events:none;
      background-repeat:no-repeat; background-position:center; background-size:contain;
      display:none; z-index:9999}
    """

    # Zoom lens JS
    js = """
    <script>
    (function(){
      const lens = document.createElement('div');
      lens.className = 'zoom-lens';
      document.body.appendChild(lens);
      let active = null, zoom = 2.5; const Z_MIN=1.5, Z_MAX=6;
      function onEnter(e){ active=e.currentTarget; lens.style.display='block'; lens.style.backgroundImage='url('+active.src+')'; }
      function onLeave(){ active=null; lens.style.display='none'; }
      function onMove(e){
        if(!active) return;
        const r=active.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
        const rx=active.naturalWidth/active.clientWidth, ry=active.naturalHeight/active.clientHeight;
        const lw=lens.offsetWidth, lh=lens.offsetHeight;
        lens.style.left=(e.clientX-lw/2)+'px'; lens.style.top=(e.clientY-lh/2)+'px';
        lens.style.backgroundSize=(active.naturalWidth*zoom)+'px '+(active.naturalHeight*zoom)+'px';
        const bgX=-(cx*rx*zoom - lw/2), bgY=-(cy*ry*zoom - lh/2);
        lens.style.backgroundPosition=bgX+'px '+bgY+'px';
      }
      function onWheel(e){
        if(!active) return; e.preventDefault();
        const d=Math.sign(e.deltaY); zoom = d<0 ? zoom*1.1 : zoom/1.1; zoom=Math.max(Z_MIN, Math.min(Z_MAX, zoom)); onMove(e);
      }
      document.querySelectorAll('img.zoomable').forEach(img=>{
        img.addEventListener('mouseenter', onEnter);
        img.addEventListener('mouseleave', onLeave);
        img.addEventListener('mousemove', onMove);
        img.addEventListener('wheel', onWheel, {passive:false});
      });
    })();
    </script>
    """

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

            # --- NEW: mini chart (P/R vs τ), only if preds.csv exists ---
            yt, ys = load_preds_arrays(run_dir / "preds.csv")
            if yt is not None and ys is not None and yt.size > 0:
                sweep = pr_recall_f1_vs_tau(yt, ys, n_points=201)
                if sweep:
                    svg = series_to_svg(sweep["precision"], sweep["recall"], sweep["taus"], sweep["best_tau"])
                    card.append('<div class="mini">')
                    card.append(svg)
                    card.append(f'<div class="meta">τ* (F1 max) = {fmt(sweep["best_tau"],3)} · F1* = {fmt(sweep["best_f1"],4)}</div>')
                    card.append("</div>")
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

        # Thumbnails (embedded data URIs) — zoomable
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
