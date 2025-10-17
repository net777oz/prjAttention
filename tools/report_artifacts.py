# -*- coding: utf-8 -*-
"""
report_artifacts.py — Static HTML Report (images embedded via base64 data URIs)
+ Hover Zoom Lens (cursor-centered magnifier, no external deps)
+ Leaderboard (static) + Manifest/Perf summaries (optional if files exist)

- artifacts/**/ 에서 model.pt 들어있는 폴더만 런으로 간주
- 각 런의 핵심 지표를 카드 내 표로 바로 표시 (링크 없음)
- plots/after_*.png 이미지를 data URI로 HTML에 직접 포함 (외부 경로/링크 없음)

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
    n=0; pos=0; neg=0
    try:
        with preds_csv.open("r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header: return None
            yi = header.index("y_true") if "y_true" in header else None
            pi = None
            for j, h in enumerate(header):
                if h.startswith("y_pred@tau="):
                    pi = j; break
            if yi is None or pi is None:
                return None
            for row in r:
                y = int(float(row[yi]))
                p = int(float(row[pi]))
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
    """Collect metrics & after_* images (list of Paths). Also manifest/perf if present."""
    plots = sorted((run_dir / "plots").glob("after_*.png"))
    if max_images > 0:
        plots = plots[:max_images]

    metrics_json = run_dir / "metrics.json"
    summary_json = run_dir / "summary.json"
    preds_csv    = run_dir / "preds.csv"
    manifest_json = run_dir / "manifest.json"
    perf_json     = run_dir / "perf.json"

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
                "tau_def": (raw.get("metrics_at_default") or {}).get("threshold", raw.get("threshold_default")),
                "m_def": raw.get("metrics_at_default") or {},
                "tau_best": (raw.get("metrics_at_best") or {}).get("threshold", raw.get("threshold_best")),
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
                "rmse": raw.get("rmse"),
                "r2": raw.get("r2"),
                "mape": raw.get("mape"),
                "smape": raw.get("smape"),
                "mase": raw.get("mase"),
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

    manifest = safe_read_json(manifest_json) if manifest_json.exists() else None
    perf     = safe_read_json(perf_json) if perf_json.exists() else None

    return kind, metrics, plots, manifest, perf

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

# --------------------------- Leaderboard build ---------------------------

def build_leaderboard_rows(runs_info):
    """
    Build a static leaderboard rows list of dicts.
    For classification: name, F1_best, F1_def, AUROC, AUPRC, N
    For regression:     name, MAE, RMSE, R2, #Win
    """
    rows = []
    for info in runs_info:
        run = info["dir"].name
        kind = info["kind"]
        md = info["metrics"]
        if kind == "classify" and isinstance(md, dict):
            n = md.get("n")
            f1_def = (md.get("m_def") or {}).get("f1")
            f1_best = (md.get("m_best") or {}).get("f1")
            rows.append({
                "name": run, "type": "CLS",
                "c1": fmt(f1_best, 4), "c2": fmt(f1_def, 4),
                "c3": fmt(md.get("auroc"), 4), "c4": fmt(md.get("auprc"), 4),
                "c5": str(int(n)) if isinstance(n, (int, float)) else "-"
            })
        elif kind == "regress" and isinstance(md, dict):
            rows.append({
                "name": run, "type": "REG",
                "c1": fmt(md.get("mae"), 6), "c2": fmt(md.get("rmse"), 6),
                "c3": fmt(md.get("r2"), 4),  "c4": fmt(md.get("smape"), 2),
                "c5": str(int(md.get("nwin"))) if isinstance(md.get("nwin"), (int, float)) else "-"
            })
    return rows

def render_leaderboard_html(rows):
    if not rows:
        return ""
    # Determine header by detecting presence of CLS or REG
    has_cls = any(r["type"] == "CLS" for r in rows)
    has_reg = any(r["type"] == "REG" for r in rows)

    head = []
    if has_cls and not has_reg:
        head = ["Run", "Type", "F1@best", "F1@def", "AUROC", "AUPRC", "N"]
    elif has_reg and not has_cls:
        head = ["Run", "Type", "MAE", "RMSE", "R²", "sMAPE", "#Win"]
    else:
        # mixed
        head = ["Run", "Type", "Col1", "Col2", "Col3", "Col4", "Col5"]

    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    trs = []
    for r in rows:
        tds = [
            f"<td style='text-align:left'>{html.escape(r['name'])}</td>",
            f"<td>{r['type']}</td>",
            f"<td>{r['c1']}</td>",
            f"<td>{r['c2']}</td>",
            f"<td>{r['c3']}</td>",
            f"<td>{r['c4']}</td>",
            f"<td>{r['c5']}</td>",
        ]
        trs.append("<tr>" + "".join(tds) + "</tr>")
    table = f"""
    <section class="leader">
      <h2>Leaderboard</h2>
      <table class="leader-table">
        <thead><tr>{th}</tr></thead>
        <tbody>
          {''.join(trs)}
        </tbody>
      </table>
    </section>
    """
    return table

# --------------------------- HTML build ---------------------------

def build_html(run_dirs: list, title: str, max_images: int) -> str:
    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;background:#0b0d10;color:#e7ecf3;margin:0}
    header.top{padding:20px 24px;border-bottom:1px solid #1d232f;background:#0f1116;position:sticky;top:0;z-index:2}
    header.top h1{font-size:20px;margin:0}
    main{padding:18px}
    .leader{margin:0 0 16px 0}
    .leader h2{font-size:16px;margin:0 0 8px 0}
    .leader-table{width:100%;border-collapse:collapse;background:#121621;border:1px solid #1e293b;border-radius:14px;overflow:hidden}
    .leader-table th,.leader-table td{border-bottom:1px solid #233044;padding:8px 6px;font-size:13px;text-align:right}
    .leader-table th:first-child,.leader-table td:first-child{text-align:left;color:#b6c6d8}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
    .card{background:#121621;border:1px solid #1e293b;border-radius:14px;padding:14px}
    .card h2{font-size:16px;margin:0 0 8px 0;word-break:break-all;display:flex;align-items:center;gap:8px}
    .badge{display:inline-block;background:#1e293b;border:1px solid #2a3a52;border-radius:10px;padding:2px 8px;font-size:11px;color:#dbe7f3}
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

    # 루페 JS
    js = """
    <script>
    (function(){
      const lens = document.createElement('div');
      lens.className = 'zoom-lens';
      document.body.appendChild(lens);

      let active = null;
      let zoom = 2.5;
      const Z_MIN = 1.5, Z_MAX = 6;

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
        const rx = active.naturalWidth / active.clientWidth;
        const ry = active.naturalHeight / active.clientHeight;
        const lw = lens.offsetWidth, lh = lens.offsetHeight;
        lens.style.left = (e.clientX - lw/2) + 'px';
        lens.style.top  = (e.clientY - lh/2) + 'px';
        lens.style.backgroundSize = (active.naturalWidth*zoom)+'px ' + (active.naturalHeight*zoom)+'px';
        const bgX = -(cx*rx*zoom - lw/2);
        const bgY = -(cy*ry*zoom - lh/2);
        lens.style.backgroundPosition = bgX+'px '+bgY+'px';
      }
      function onWheel(e){
        if(!active) return;
        e.preventDefault();
        const delta = Math.sign(e.deltaY);
        zoom = delta < 0 ? zoom*1.1 : zoom/1.1;
        zoom = Math.max(Z_MIN, Math.min(Z_MAX, zoom));
        onMove(e);
      }
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

    # 먼저 런 정보를 수집하여 리더보드 생성
    runs_info = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        kind, md, imgs, manifest, perf = collect_run(run_dir, max_images)
        runs_info.append({
            "dir": run_dir,
            "name": run_name,
            "kind": kind,
            "metrics": md or {},
            "images": imgs,
            "manifest": manifest,
            "perf": perf
        })

    # Leaderboard 섹션
    lb_rows = build_leaderboard_rows(runs_info)
    leader_html = render_leaderboard_html(lb_rows)

    # 카드들 생성
    cards = []
    for info in runs_info:
        run_dir = info["dir"]
        run_name = run_dir.name
        when = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        kind = info["kind"]; md = info["metrics"]; imgs = info["images"]
        manifest = info["manifest"]; perf = info["perf"]

        # 제목 배지
        badge = ""
        if kind == "classify":
            f1_best = (md.get("m_best") or {}).get("f1")
            if f1_best is None:
                f1_best = (md.get("m_def") or {}).get("f1")
            if f1_best is not None:
                badge = f"<span class='badge'>F1 {fmt(f1_best,4)}</span>"
        elif kind == "regress":
            if md.get("mae") is not None:
                badge = f"<span class='badge'>MAE {fmt(md.get('mae'),6)}</span>"

        # Header
        card = [f'<div class="card">', f"<h2>{html.escape(run_name)} {badge}</h2>", f'<div class="meta">{html.escape(when)}</div>']

        # Manifest/Perf 요약 (있을 때만)
        if isinstance(manifest, dict):
            mdev = manifest.get("device")
            amp = manifest.get("amp")
            comp = manifest.get("compile")
            back = manifest.get("backbone")
            ctx = manifest.get("context_len")
            ch = manifest.get("in_channels")
            params = (manifest.get("model_size") or {}).get("params")
            ds = manifest.get("data") or {}
            n = ds.get("N"); c = ds.get("C"); t = ds.get("T")
            trw = ds.get("train_windows"); vaw = ds.get("val_windows")
            card.append("<table>")
            card.append("<tr><th>Run meta</th><td></td></tr>")
            if back is not None:
                card.append(f"<tr><th>Backbone</th><td>{html.escape(str(back))}</td></tr>")
            if params is not None:
                card.append(f"<tr><th>Params</th><td>{int(params):,}</td></tr>")
            if ctx is not None or ch is not None:
                card.append(f"<tr><th>Input</th><td>C={html.escape(str(ch))} · L={html.escape(str(ctx))}</td></tr>")
            if (n is not None) or (c is not None) or (t is not None):
                card.append(f"<tr><th>Data (N,C,T)</th><td>{n}/{c}/{t}</td></tr>")
            if (trw is not None) or (vaw is not None):
                card.append(f"<tr><th>Windows</th><td>train {trw} · val {vaw}</td></tr>")
            if mdev is not None or amp is not None or comp is not None:
                card.append(f"<tr><th>Device</th><td>{html.escape(str(mdev))} · amp {amp} · compile {html.escape(str(comp))}</td></tr>")
            card.append("</table>")

        if isinstance(perf, dict):
            thr = perf.get("throughput_sps")
            lat = perf.get("latency_ms") or {}
            p95 = (perf.get("latency") or {}).get("p95_ms")
            maxmem = perf.get("max_mem_mb")
            card.append("<table>")
            card.append("<tr><th>Performance</th><td></td></tr>")
            if thr is not None:
                card.append(f"<tr><th>Throughput</th><td>{fmt(thr,2)} samples/s</td></tr>")
            if lat:
                # show few keys if present
                keys = sorted(lat.keys())
                show = ", ".join([f"{k}:{fmt(lat[k],2)}ms" for k in keys[:4]])
                card.append(f"<tr><th>Latency</th><td>{show}</td></tr>")
            if p95 is not None:
                card.append(f"<tr><th>Latency p95</th><td>{fmt(p95,2)} ms</td></tr>")
            if maxmem is not None:
                card.append(f"<tr><th>Max Mem</th><td>{fmt(maxmem,0)} MB</td></tr>")
            card.append("</table>")

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
            if md.get("mse") is not None:
                card.append(f"<tr><th>avg MSE</th><td>{fmt(md.get('mse'),6)}</td></tr>")
            if md.get("mae") is not None:
                card.append(f"<tr><th>avg MAE</th><td>{fmt(md.get('mae'),6)}</td></tr>")
            if md.get("rmse") is not None or md.get("r2") is not None:
                card.append(f"<tr><th>RMSE / R²</th><td>{fmt(md.get('rmse'),6)} / {fmt(md.get('r2'),4)}</td></tr>")
            if md.get("mape") is not None or md.get("smape") is not None:
                card.append(f"<tr><th>MAPE / sMAPE</th><td>{fmt(md.get('mape'),2)}% / {fmt(md.get('smape'),2)}%</td></tr>")
            if md.get("mase") is not None:
                card.append(f"<tr><th>MASE</th><td>{fmt(md.get('mase'),3)}</td></tr>")
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
<header class="top"><h1>{html.escape(title)}</h1></header>
<main>
  {leader_html}
  <div class="grid">
    {''.join(cards)}
  </div>
</main>
<footer>Generated at {html.escape(now)}</footer>
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

    run_dirs = find_runs(artifacts_dir)
    html_text = build_html(run_dirs, args.title, args.max_images)
    out_html.write_text(html_text, encoding="utf-8")
    print(f"[OK] Wrote static (embedded) report: {out_html}")

if __name__ == "__main__":
    main()
