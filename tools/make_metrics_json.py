# -*- coding: utf-8 -*-
"""
make_metrics_json.py — 과거 실험 폴더에서 예측값 파일을 읽어 metrics.json 생성.

지원 입력:
  1) preds.csv (열 이름 추정: ["y_true","label","target"] & ["y_score","prob","pred_proba","score"])
  2) preds.npy (dict 또는 (y_true, y_score) 저장한 npy)
  3) 기타 *.csv에서 위와 유사한 열을 자동 탐색

사용:
  python tools/make_metrics_json.py --in ./artifacts/run_xxx/eval/preds.csv --outdir ./artifacts/run_xxx/eval
"""
import argparse, os, json
from pathlib import Path
import numpy as np

from metrics_writer import write_metrics_json  # 같은 tools 폴더에 있어야 합니다.

CAND_YTRUE = ["y_true","label","labels","target","y","gt","truth"]
CAND_YSCORE= ["y_score","prob","proba","pred_proba","score","p","y_prob"]

def _load_from_csv(path: Path):
    import csv
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise SystemExit("[ERR] CSV가 비어 있습니다.")
    cols = rows[0].keys()
    def find(names):
        for n in names:
            if n in cols:
                return n
        return None
    c_y = find(CAND_YTRUE)
    c_p = find(CAND_YSCORE)
    if not c_y or not c_p:
        raise SystemExit(f"[ERR] 열 이름을 찾지 못했습니다. y_true 후보={CAND_YTRUE}, y_score 후보={CAND_YSCORE}, 실제={list(cols)}")
    y = np.array([float(x[c_y]) for x in rows], dtype=np.float32).round().astype(int)
    p = np.array([float(x[c_p]) for x in rows], dtype=np.float32)
    return y, p

def _load_from_npy(path: Path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):  # np.savez? (예외적으로 처리)
        keys = list(obj.keys())
        if "y_true" in keys and "y_score" in keys:
            return obj["y_true"].astype(int).flatten(), obj["y_score"].astype(float).flatten()
        raise SystemExit(f"[ERR] npz 파일에 y_true/y_score 키가 없습니다. keys={keys}")
    # 단일 npy: dict 또는 tuple/list
    if isinstance(obj.item(), dict):
        d = obj.item()
        return np.asarray(d["y_true"]).astype(int).flatten(), np.asarray(d["y_score"]).astype(float).flatten()
    elif isinstance(obj.item(), (tuple, list)):
        y, p = obj.item()
        return np.asarray(y).astype(int).flatten(), np.asarray(p).astype(float).flatten()
    else:
        raise SystemExit("[ERR] npy 구조를 인식하지 못했습니다. dict 또는 (y_true,y_score) 형태여야 합니다.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="입력 파일 경로 (preds.csv / preds.npy 등)")
    ap.add_argument("--outdir", required=True, help="metrics.json을 기록할 폴더")
    ap.add_argument("--thr", type=float, default=0.5, help="기본 임계값")
    ap.add_argument("--best", type=str, default="f1", help="best threshold 기준 (f1)")
    ap.add_argument("--save-preds", action="store_true", help="preds.csv도 복사 생성")
    args = ap.parseArgs() if hasattr(ap, "parseArgs") else ap.parse_args()

    inp = Path(args.inp); outdir = Path(args.outdir)
    if not inp.exists():
        raise SystemExit(f"[ERR] 입력 파일이 없습니다: {inp}")
    outdir.mkdir(parents=True, exist_ok=True)

    if inp.suffix.lower() == ".csv":
        y, p = _load_from_csv(inp)
    elif inp.suffix.lower() == ".npy" or inp.suffix.lower() == ".npz":
        y, p = _load_from_npy(inp)
    else:
        raise SystemExit("[ERR] 지원하지 않는 입력 형식입니다. (.csv, .npy, .npz)")

    metrics = write_metrics_json(
        y_true=y, y_score=p, out_dir=str(outdir),
        default_threshold=args.thr, strategy_for_best=args.best,
        save_preds=args.save_preds, preds_name="preds.csv"
    )
    print("[OK] metrics.json 생성:", outdir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
