# 🧠 AttentionProject

**LLM-style Time-Series Forecasting & Classification Framework**

---

## 📘 개요

AttentionProject는 시계열 데이터를 Transformer/LLM 기반으로 처리하기 위한  
**모듈형 파이프라인**입니다.  
데이터 로드 → 윈도우 생성 → 모델 로드 → 학습/추론 → 리포트 및 플롯을  
모두 자동화하며, 단일 CSV뿐 아니라 **다변량(Multi-CSV)** 입력을 지원합니다.

---

## 🧩 주요 특징

| 구성요소 | 설명 |
|-----------|------|
| `cli.py` | 명령행 인자 파서 및 entry point (`run_llmts.py` 호출) |
| `pipeline.py` | 전체 오케스트레이션 (데이터 로드 → 학습/추론 → 리포트 저장) |
| `data.py` | CSV 파싱, 텐서 변환, 시드 고정 및 출력 경로 생성 |
| `windows.py` | 슬라이딩 윈도우 데이터셋 생성 |
| `splits.py` | 학습/검증 분할 로직 |
| `trainer.py` | 모델 학습 루프 |
| `evaler.py` | 모델 평가 (회귀/분류) |
| `metrics.py` | F1, Accuracy, MAE, MSE 등 메트릭 계산 |
| `viz.py` | 학습 곡선, 분류 ROC/PR 플롯 생성 |
| `ttm_flow/model.py` | Granite TinyTimeMixer 기반 LLM-TS 백본 로더 |

---

## 🧮 입력 구조

### 단변량 (기존)
- 입력 CSV 1개 (`[N,T]`)
- 자동 변환: `[N,1,T]`

```bash
--csv ./data/SPa.csv
다변량 (New)
입력 CSV 여러 개 ([N,T] 동일 형태)

자동 변환: [N,C,T] (C = CSV 개수)

첫 번째 CSV(x_main.csv)가 타겟 채널

bash
코드 복사
--csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv
⚙️ 공통 인자 요약
인자	설명	기본값
--mode	train / finetune / infer	(필수)
--task	regress / classify	regress
--context-len	윈도우 길이	(필수)
--epochs	학습 반복 횟수	1
--batch-size	배치 크기	4096
--alpha	분류용 F1 조합 가중치	0.5
--pos-weight	양성 가중치 계산방식	global
--bin-rule	분류 타겟 이진화 규칙	nonzero
--bin-thr	이진화 기준값	0.0
--split-mode	group / item / time / window	group
--val-ratio	검증 데이터 비율	0.2
--ckpt	checkpoint 경로	None
--amp	자동 혼합정밀 (CUDA)	False
--compile	torch.compile 모드	""
--device	cuda / cpu 자동선택	cuda

🚀 실행 예제
🧠 Training
새 모델 학습

bash
코드 복사
python run_llmts.py --mode train --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 --epochs 30 --batch-size 4096 \
  --alpha 0.5 --pos-weight global \
  --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5 \
  --split-mode group --val-ratio 0.2 \
  --amp --compile reduce-overhead
다변량 입력: [N,3,T]

첫 번째 CSV(x_main.csv)가 타겟

저장: artifacts/train_classify_ctx31_ch3_* 폴더에

model.pt

train_report.txt

plots/

🔧 Finetune
기존 모델 이어 학습

bash
코드 복사
python run_llmts.py --mode finetune --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 --epochs 10 --batch-size 4096 \
  --alpha 0.5 --pos-weight global \
  --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5 \
  --split-mode group --val-ratio 0.2 \
  --ckpt ./artifacts/train_classify_ctx31_ch3_alpha0.50_pwglobal_llm_ts_seed777_splitgroup/model.pt \
  --amp --compile reduce-overhead
🔍 Inference
학습된 모델로 추론

bash
코드 복사
python run_llmts.py --mode infer --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 \
  --ckpt ./artifacts/train_classify_ctx31_ch3_alpha0.50_pwglobal_llm_ts_seed777_splitgroup/model.pt \
  --amp --compile reduce-overhead
출력:

infer_report.txt

F1 / Accuracy / Precision / Recall / MSE / MAE

plots/ 시각화

model.pt (메타 정보 포함 저장)

🧾 출력 구조
파일	설명
model.pt	원본 nn.Module의 state_dict + meta 정보
train_report.txt	Before/After 성능 지표
infer_report.txt	추론 결과 리포트
plots/	손실 곡선, ROC/PR 플롯 등

⚡ 성능 최적화
Tensor Core GPU (RTX/Ampere+)에서 더 빠른 연산을 위해
pipeline.py 또는 run_llmts.py 시작 부분에 아래 한 줄을 추가하세요.

python
코드 복사
import torch
torch.set_float32_matmul_precision('high')
이 설정은 torch.matmul, nn.Linear, Attention 등에서
FP32 → TF32 텐서코어 연산을 활성화하여 속도를 20~40% 향상시킵니다.

🧠 Reference
Granite TinyTimeMixer (TTM-R1)

PyTorch torch.compile

Automatic Mixed Precision (AMP)

📁 프로젝트 구조
markdown
코드 복사
prjAttention/
├── run_llmts.py
├── cli.py
├── pipeline.py
├── data.py
├── windows.py
├── splits.py
├── trainer.py
├── evaler.py
├── metrics.py
├── viz.py
├── ttm_flow/
│   ├── __init__.py
│   ├── model.py
│   └── ...
└── artifacts/
    └── (자동생성)
🧩 요약
모드	설명	출력
train	새 모델 학습	model.pt, train_report.txt, plots/
finetune	기존 모델 이어 학습	동일 폴더 내 재저장
infer	추론 및 리포트 생성	infer_report.txt, plots/