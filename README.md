# AttentionProject (prjAttention)

시계열 CSV를 입력으로 받아 **분류(classify) / 회귀(regress)** 작업을 수행하는 파이프라인입니다.  
여러 개의 CSV를 **채널(C)** 로 스택해 **[N, C, T]** 텐서(배치 N, 채널 C, 시간길이 T)로 학습/평가하며,  
아티팩트(모델 가중치, 지표, 그래프)는 표준 규칙에 따라 `artifacts/` 하위에 저장됩니다.

- Python: 3.12
- PyTorch: 2.8.0+cu129 (CUDA 환경 권장)
- 기본 백본: `llm_ts`

> **체크포인트 규칙**: `--ckpt`(혹은 `--model`)에 **디렉터리 경로**를 전달하면 `model.pt.best`를 우선 사용하고, 없으면 `model.pt`를 사용합니다.

---

## 설치

```bash
# 1) 가상환경 (예시)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) 필수 패키지
pip install -U pip wheel
pip install -r requirements.txt  # (프로젝트 루트 기준)
```

---

## 데이터 규칙

- **시간축은 열(컬럼)** 방향입니다.  
- **단변량**: CSV 1개 = 채널 1개 → 입력 형태 [N, 1, T]  
- **다변량**: CSV 여러 개를 나열 → 채널로 스택(C 증가) → [N, C, T]  
- 컨텍스트 길이 `--context-len = L`이면, **윈도 길이 `T = L + 1`** 입니다.  
  (예: `--context-len 31` → 입력 31 스텝 + 다음 1 스텝 라벨)

> **타깃(라벨) 채널**: 기본은 **채널 0(첫 번째 CSV)** 의 다음 스텝입니다. 필요 시 `--label-src-ch`로 변경할 수 있습니다.

---

## 디렉터리 구조

```
.
├── run_llmts.py         # 엔트리포인트 (mode/task/옵션 파싱 및 파이프라인 호출)
├── cli.py               # 공용 argparse 정의
├── pipeline.py          # End-to-End (로드→윈도→분할→모델→학습/평가→저장)
├── trainer.py           # 학습 루프(에폭/AMP/compile/스케줄러/체크포인트)
├── evaler.py            # 평가 및 산출물 기록(분류/회귀 공용)
├── data.py              # CSV 로더(N개의 CSV → 채널 스택 C)
├── windows.py           # 슬라이딩 윈도우(T = context_len + 1)
├── splits.py            # 분할 정책(group/rolling/holdout)
├── losses.py            # 손실함수(BCE, pos_weight 등)
├── metrics.py           # 지표 계산(F1/ROC_AUC/PR_AUC/MAE/R² 등)
├── viz.py               # ROC/PR/히스토그램/잔차/산점도 그리기
├── ttm_flow/            # TinyTimeMixer 계열 백본 래퍼
├── tools/               # 메트릭 집계/리포트 보조 스크립트
├── data/                # CSV 데이터
└── artifacts/           # 실행 결과(체크포인트/지표/그래프/예측)
```

---

## 아티팩트 저장 규칙

### 분류(classify)
- `model.pt`, `model.pt.best`(있을 때)
- `preds.csv` (y_true, y_score, y_pred@tau)
- `metrics.json` (precision, recall, F1, accuracy, roc_auc, pr_auc, best_tau 등)
- `roc_curve.png`, `pr_curve.png`, `hist_score.png` 등

### 회귀(regress)
- `model.pt`, `model.pt.best`(있을 때)
- `summary.json` (MAE, MSE, RMSE, R² 등)
- `residuals.png`, `scatter.png` 등

---

## CLI 옵션

### 필수/핵심
- `--mode {train,finetune,infer}` : 동작 모드
- `--task {classify,regress}` : 작업 유형

### 데이터 입력
- `--csv <파일들...>` : 입력 CSV 경로(1개 이상). **나열 순서 = 채널 순서(C)**  
  - 브레이스 확장 예:  
    - `./data/LP{1..31}a.csv` → `LP1a.csv` … `LP31a.csv`  
    - `./data/LP{001..031}a.csv` → `LP001a.csv` … `LP031a.csv`
- `--label-src-ch <int>` (기본: 0) : 라벨을 생성할 소스 채널 인덱스
- `--drop-label-from-x` : 라벨 소스 채널을 입력 X에서 제외(기본: 제외하지 않음)

### 윈도우/라벨링
- `--context-len <int>` : 컨텍스트 길이 L (**T = L + 1**)
- `--label-offset <int>` (기본: 1) : 예측 지연 스텝(1-step ahead 등)

### 이진 분류 라벨 생성
- `--bin-rule {nonzero,gt,lt,quantile}` : 라벨 이진화 규칙
  - `nonzero` : 0/비0 여부
  - `gt` : `value > --bin-thr`
  - `lt` : `value < --bin-thr`
  - `quantile` : 분위수 기준
- `--bin-thr <float>` : 임계/분위값
- `--thresh-default <float>` (기본: 0.5) : 확률→클래스 변환 임계값 τ  
  (리포트에는 best_tau도 함께 기록)

### 분할/평가
- `--split-mode {group,rolling,holdout}` : 데이터 분할 정책
- `--val-ratio <0~1>` : 검증셋 비율(학습 시)
- `--eval-split {train,val,test}` : 지표 산출 대상 스플릿
- `--plot-split {train,val,test}` : 그래프 생성 대상 스플릿

### 학습 하이퍼파라미터
- `--epochs <int>`
- `--batch-size <int>`
- `--lr <float>` : 학습률
- `--alpha <float>` : 복합 손실 가중(예: BCE+F1 혼합 비율)
- `--pos-weight {global,<float>}` : 클래스 불균형 보정
  - `global` : 전체 빈도 기반 양성 가중 자동 계산
  - `<float>` : 사용자가 직접 지정
- `--seed <int>` : 시드 고정

### 가속/컴파일
- `--amp` : 자동 혼합정밀 사용
- `--compile {off,reduce-overhead}` : `torch.compile` 전략

### 체크포인트/백본/태깅
- `--ckpt <path>` : 파일 또는 디렉터리 경로  
  - 디렉터리: `model.pt.best` → `model.pt` 순으로 로드
  - 파일: 해당 파일을 직접 로드
- `--backbone {llm_ts}` : 백본 선택(기본: `llm_ts`)
- `--desc <str>` : 실행 폴더명에 부가 태그 추가

---

## 실행 예시

### A. Train — 분류 / 단변량
```bash
python run_llmts.py --mode train --task classify   --csv ./data/LP0a.csv   --context-len 31 --epochs 30 --batch-size 4096   --alpha 0.5 --pos-weight global   --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5   --split-mode group --val-ratio 0.2   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

### B. Train — 분류 / 다변량 (명시 나열)
```bash
python run_llmts.py --mode train --task classify   --csv ./data/LP0a.csv ./data/LP1a.csv ./data/LP2a.csv   --context-len 31 --epochs 30 --batch-size 4096   --alpha 0.5 --pos-weight global   --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5   --split-mode group --val-ratio 0.2   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

### C. Train — 분류 / 다변량 (브레이스 확장 1..31)
```bash
python run_llmts.py --mode train --task classify   --csv ./data/LP{1..31}a.csv   --context-len 31 --epochs 30 --batch-size 4096   --alpha 0.5 --pos-weight global   --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5   --split-mode group --val-ratio 0.2   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

### D. Train — 회귀 / 단변량
```bash
python run_llmts.py --mode train --task regress   --csv ./data/x_main.csv   --context-len 31 --epochs 20 --batch-size 2048   --split-mode group --val-ratio 0.2   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

### E. Finetune — 분류 / 다변량 (기존 체크포인트에서 이어서)
```bash
python run_llmts.py --mode finetune --task classify   --csv ./data/LP{001..031}a.csv   --context-len 31 --epochs 10 --batch-size 4096   --alpha 0.5 --pos-weight global   --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5   --split-mode group --val-ratio 0.2   --ckpt ./artifacts/train_classify_ctx31_ch31_alpha0.50_pwglobal_llm_ts_seed777_splitgroup   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

### F. Infer — 분류 / 단변량
```bash
python run_llmts.py --mode infer --task classify   --csv ./data/LP0a.csv   --context-len 31   --ckpt ./artifacts/train_classify_ctx31_ch1_alpha0.50_pwglobal_llm_ts_seed777_splitgroup   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

### G. Infer — 분류 / 다변량(브레이스 확장)
```bash
python run_llmts.py --mode infer --task classify   --csv ./data/LP{001..031}a.csv   --context-len 31   --ckpt ./artifacts/train_classify_ctx31_ch31_alpha0.50_pwglobal_llm_ts_seed777_splitgroup   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

### H. 고급 — 라벨 채널 변경 + 입력에서 제외
```bash
python run_llmts.py --mode train --task classify   --csv ./data/A.csv ./data/B.csv   --label-src-ch 1 --drop-label-from-x   --context-len 31 --epochs 20 --batch-size 2048   --bin-rule gt --bin-thr 0.0 --thresh-default 0.5   --split-mode rolling --val-ratio 0.2   --eval-split val --plot-split val   --amp --compile reduce-overhead
```

---

## FAQ

- **시간축은 어디인가요?** → **열(컬럼)** 입니다. `--context-len = L`이면, 윈도 길이 `T = L + 1` 입니다.  
- **다변량 입력은 어떻게 구성하나요?** → `--csv` 뒤에 여러 파일을 나열하면 채널로 스택됩니다. 첫 CSV(채널 0)가 기본 타깃 소스입니다.  
- **분류 임계값 τ는 어디서 설정하나요?** → `--thresh-default` (기본 0.5). 리포트에는 best_tau도 함께 기록됩니다.  
- **클래스 불균형 보정은 어떻게 하나요?** → `--pos-weight global` 권장. 필요 시 실수값으로 고정 가중 지정 가능합니다.  
- **가속 설정은 어떻게 하나요?** → `--amp` + `--compile reduce-overhead` 조합을 권장합니다.

---

## 라이선스
TBD
