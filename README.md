🧠 AttentionProject

LLM-style Time-Series Forecasting & Classification Framework

데이터 로드 → 윈도우 구성 → 학습/추론 → 아티팩트 자동 저장 → HTML 리포트까지 한 번에.

🔔 이번 업데이트 핵심

evaler.py 드롭-인 교체

분류: preds.csv, metrics.json, ROC/PR/히스토그램 자동 저장

회귀: summary.json 및 잔차/산점도 자동 저장

저장 경로: ./artifacts/llm_ts_eval/<tag>/ (<tag> = split_name 또는 desc)

리포트 생성기 tools/report_artifacts.py

artifacts/ 하위 실험 폴더를 훑어 단일 HTML 리포트(검색/정렬/필터) 생성

지표 생성 도구

tools/metrics_writer.py: 코드 내에서 즉시 metrics.json 저장

tools/make_metrics_json.py: 과거 preds.csv/.npy/.npz로 사후 metrics.json 생성

🚀 빠른 시작 (명령만 보면 되는 섹션)
1) 학습(분류)
python run_llmts.py --mode train --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 --epochs 30 --batch-size 4096 \
  --alpha 0.5 --pos-weight global \
  --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5 \
  --split-mode group --val-ratio 0.2 \
  --eval-split val --plot-split val \
  --amp --compile reduce-overhead

2) 파인튜닝
python run_llmts.py --mode finetune --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 --epochs 10 --batch-size 4096 \
  --alpha 0.5 --pos-weight global \
  --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5 \
  --split-mode group --val-ratio 0.2 \
  --ckpt ./artifacts/train_classify_ctx31_ch3_alpha0.50_pwglobal_llm_ts_seed777_splitgroup/model.pt \
  --eval-split val --plot-split val \
  --amp --compile reduce-overhead

3) 추론(infer)
python run_llmts.py --mode infer --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 \
  --ckpt ./artifacts/train_classify_ctx31_ch3_alpha0.50_pwglobal_llm_ts_seed777_splitgroup/model.pt \
  --eval-split val --plot-split val \
  --amp --compile reduce-overhead


라벨 없는 추론이라면 지표/곡선은 생성되지 않습니다(확률/예측만 파일로 남을 수 있음).

🧩 주요 기능 요약
데이터 입력

단변량: CSV 1개 → 자동으로 [N, 1, T]

다변량(Multi-CSV): CSV 여러 개 → 자동으로 [N, C, T] (C=파일 수)
타깃은 항상 채널 0(첫 번째 CSV) 다음 스텝입니다.

스플릿 제어(안전 기본값)

--eval-split val, --plot-split val이 기본 → 학습 중 평가/플롯은 검증셋 기준
필요 시 train/all로 변경하세요.

성능/자원

--amp(가능한 디바이스에서만 자동 활성), --compile reduce-overhead|max-autotune 지원

대규모 시계열에 맞춘 OOM-안전 롤링 윈도우 평가

🗂️ 아티팩트(자동 저장)
저장 구조(기본)
artifacts/
├── llm_ts_eval/
│   └── <tag>/                   # <tag> = split_name or desc
│       ├── (분류) metrics.json, preds.csv, roc.png, pr.png, hist_*.png
│       └── (회귀) summary.json, resid_hist.png, pred_vs_true.png
└── report/
    ├── index.html               # 리포트
    ├── style.css
    └── app.js

evaler.py가 언제 실행됨?

train/finetune 중 검증 단계 및 학습 종료 후 최종 평가

infer 모드:

라벨이 있으면 → 분류/회귀 지표 및 플롯, metrics.json/summary.json 생성

라벨이 없으면 → 순수 추론(지표/곡선 생략)

📊 HTML 리포트 생성 (tools/report_artifacts.py)
실행
python tools/report_artifacts.py \
  --artifacts ./artifacts \
  --out ./artifacts/report/index.html \
  --max-images 24 \
  --sort-key f1 \
  --sort-by-score \
  --verbose

특징

자동 스캔: artifacts/ 하위의 지표(JSON)와 그래프 이미지 모아 단일 HTML

간단 분석 포함: F1/Precision/Recall/AUPRC/AUROC/Accuracy/Threshold 패턴으로 한 줄 코멘트

검색/정렬/필터: 실험명 검색, 최소 스코어, score-key 필터, 최신순/스코어순 정렬

가벼움: 이미지 Base64 미사용(상대경로 링크) → 용량 최소화

JSON이 없어도 이미지만 있는 폴더는 카드가 표시됩니다.
경로 계산은 자동(리포트 위치와 무관)이며, 문제 시 --verbose로 원인을 확인하세요.

🧾 지표 생성 도구 (선택)
1) 실시간 저장 모듈 (tools/metrics_writer.py)

평가 루틴에서 바로 호출하여 metrics.json(+옵션 preds.csv) 저장

ROC-AUC/PR-AUC, default/best τ, confusion, prevalence 등을 포함

사용처: 커스텀 평가코드/외부 러너에서 즉시 지표 파일을 만들고 싶을 때

2) 사후 소급 CLI (tools/make_metrics_json.py)

과거 실험의 preds.csv/.npy/.npz로 나중에 metrics.json 생성

컬럼 자동 탐색:

y_true: y_true,label,labels,target,y,gt,truth

y_score: y_score,prob,proba,pred_proba,score,p,y_prob

실행 예시
# CSV → metrics.json
python tools/make_metrics_json.py \
  --in ./artifacts/llm_ts_eval/val/preds.csv \
  --outdir ./artifacts/llm_ts_eval/val \
  --save-preds

# NPY/NPZ → metrics.json (dict or (y_true, y_score))
python tools/make_metrics_json.py \
  --in ./artifacts/old_run/preds.npy \
  --outdir ./artifacts/old_run

⚙️ 자주 쓰는 주요 인자 (요약)

공통

--mode {train,finetune,infer}

--task {regress,classify}

--context-len <int> (필수)

--batch-size, --epochs, --lr, --weight-decay, --device, --seed

--split-mode {group,item,time,window}, --val-ratio <float>

--eval-split {val,train,all}, --plot-split {val,train,all}, --no-plots

--amp, --compile {,reduce-overhead,max-autotune}

분류 전용

--alpha (BCE vs SoftF1 혼합 가중치)

--pos-weight {global,batch,none}

--bin-rule {nonzero,gt,ge}, --bin-thr <float>

--thresh-default <float> (기본 임계값 τ)

🆘 트러블슈팅

리포트에 카드가 안 보임

폴더에 이미지나 JSON이 없는 경우일 수 있습니다.

평가가 실제로 돌았는지(evaler.py가 artifacts/llm_ts_eval/<tag>/… 파일 생성했는지) 확인하세요.

과거 실험이면 tools/make_metrics_json.py로 metrics.json을 소급 생성.

이미지가 깨짐

--out / --artifacts 경로가 달라도 상대경로를 자동 계산합니다.

권한/경로 문제는 --verbose로 콘솔 로그 확인.

라벨 없는 추론(Infer)

y_true가 없으면 지표/곡선은 생성되지 않습니다(의도된 동작).

확률/예측만 필요하면 OK, 지표가 필요하면 라벨이 있는 평가 루틴으로 돌리세요.

📌 팁

Ampere+ GPU면 시작부에 아래 설정을 권장합니다(가속 기대):

import torch
torch.set_float32_matmul_precision("high")


--amp는 CUDA/MPS 등에서만 활성, CPU 환경은 자동 비활성.