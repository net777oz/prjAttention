# 🧠 AttentionProject
LLM-style Time-Series Forecasting & Classification Framework

## 📘 개요
AttentionProject는 시계열 데이터를 Transformer/LLM 방식으로 처리하는 모듈형 파이프라인입니다.  
데이터 로드 → 윈도우 생성 → 모델 로드 → 학습/추론 → 리포트 & 플롯까지 자동화하며, **단일 CSV**는 물론 **다변량(Multi-CSV)** 입력을 지원합니다.

> **중요 변경점(업데이트)**
> - `viz.py`, `evaler.py`가 넘겨받은 텐서만 사용하도록 명확화. 기본 동작은 변경 없음.
> - **CLI에 스플릿 제어 플래그 추가**: `--eval-split`, `--plot-split` (기본값 `val`).  
>   학습 중 평가/플롯이 **검증셋만** 사용되도록 기본을 안전하게 고정했습니다.
> - `trainer.py` AMP 자동화(디바이스 인식), 분류 pos_weight 글로벌 설정 안정화.

---

## 🧩 주요 구성요소
| 파일 | 설명 |
|---|---|
| `run_llmts.py` | 런처(엔트리). 내부에서 `cli.main()` 호출 |
| `cli.py` | 명령행 인자 파서(스플릿 제어 플래그 포함) |
| `pipeline.py` | 전체 오케스트레이션(데이터 로드 → 학습/추론 → 리포트 저장) |
| `data.py` | CSV 파싱, 텐서 변환, 시드 고정, 출력 경로 생성 |
| `windows.py` | 슬라이딩 윈도우 데이터셋 생성 |
| `splits.py` | 학습/검증 분할 로직 |
| `trainer.py` | 학습 루프(AMP/torch.compile 대응, 분류 하이브리드 손실) |
| `evaler.py` | 평가(회귀/분류 공용, OOM-안전, 선택 인덱스 지원) |
| `metrics.py` | F1, Accuracy, MAE, MSE 등 |
| `viz.py` | 손실 곡선, ROC/PR, F1-τ, Confusion, Prob-Hist, 샘플 플롯 |

---

## 🧮 입력 구조
### 단변량 (기존)
- 입력 CSV 1개: `shape [N, T]` → 자동으로 ` [N, 1, T]`
```bash
--csv ./data/SPa.csv
다변량 (Multi-CSV)
입력 CSV 여러 개: 각 파일 shape [N, T] 동일 → 자동으로 [N, C, T] (C=파일 수)

타깃은 항상 채널 0(첫 번째 CSV) 의 다음 스텝입니다. 나머지 채널은 보조 피처로 사용됩니다.

bash
코드 복사
--csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv
⚙️ 공통 인자 요약
인자	설명	기본값
--mode	train / finetune / infer	(필수)
--task	regress / classify	regress
--backbone	백본 이름	llm_ts
--context-len	윈도우 길이	(필수)
--epochs	학습 반복 횟수	1
--batch-size	배치 크기	4096
--lr / --weight-decay	최적화 하이퍼파라미터	2e-4 / 0.01
--device	cuda/cpu 자동선택	(환경 감지)
--seed	시드	777
--out	출력 디렉터리(없으면 자동 생성)	None
--plot-samples	샘플 플롯 개수	3
--log-every	로그 주기(미니배치)	50
분류 전용		
--alpha	BCE vs SoftF1 혼합 가중치	0.5
--pos-weight	양성 가중치 방식 (global/batch/none)	global
--thresh-default	기본 임계값(τ)	0.5
--val-ratio	검증 비율	0.2
--bin-rule	타깃 이진화 규칙 (nonzero/gt/ge)	nonzero
--bin-thr	이진화 기준값	0.0
리소스/성능		
--num-workers	DataLoader 워커 수	4
--amp	자동 혼합정밀(가능한 디바이스에서만 활성)	False
--compile	torch.compile 모드 ("", reduce-overhead, max-autotune)	""
--eval-batch-size	평가 배치 크기	4096
스플릿 제어(신규)		
--split-mode	group / item / time / window	group
--eval-split	평가에 사용할 스플릿 (val/train/all)	val
--plot-split	플롯에 사용할 스플릿 (val/train/all)	val
--no-plots	플롯 생성 비활성화	False
체크포인트		
--ckpt	불러올 모델 경로	None

--eval-split/--plot-split 기본값을 val로 고정하여, 학습 중 평가/플롯이 전체 데이터가 아닌 검증셋에 대해서만 수행되도록 했습니다. 필요 시 all 또는 train으로 변경하세요.

🚀 실행 예제
🧠 Training (분류)
bash
코드 복사
python run_llmts.py --mode train --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 --epochs 30 --batch-size 4096 \
  --alpha 0.5 --pos-weight global \
  --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5 \
  --split-mode group --val-ratio 0.2 \
  --eval-split val --plot-split val \
  --amp --compile reduce-overhead
입력: [N,3,T] (다변량), 타깃=첫 CSV(채널 0)

출력 폴더(예): artifacts/train_classify_ctx31_ch3_*

model.pt

train_report.txt

plots/

🔧 Finetune
bash
코드 복사
python run_llmts.py --mode finetune --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 --epochs 10 --batch-size 4096 \
  --alpha 0.5 --pos-weight global \
  --bin-rule nonzero --bin-thr 0.0 --thresh-default 0.5 \
  --split-mode group --val-ratio 0.2 \
  --ckpt ./artifacts/train_classify_ctx31_ch3_alpha0.50_pwglobal_llm_ts_seed777_splitgroup/model.pt \
  --eval-split val --plot-split val \
  --amp --compile reduce-overhead
🔍 Inference
bash
코드 복사
python run_llmts.py --mode infer --task classify \
  --csv ./data/x_main.csv ./data/x_aux1.csv ./data/x_aux2.csv \
  --context-len 31 \
  --ckpt ./artifacts/train_classify_ctx31_ch3_alpha0.50_pwglobal_llm_ts_seed777_splitgroup/model.pt \
  --eval-split val --plot-split val \
  --amp --compile reduce-overhead
출력:

infer_report.txt (F1 / Acc / Precision / Recall / MSE / MAE)

plots/ (ROC/PR, F1-τ, Confusion, Prob-Hist 등)

필요 시 model.pt 갱신(메타 포함)

🧾 출력 구조
파일/폴더	설명
model.pt	state_dict + 메타 정보
train_report.txt	학습 성능 지표(Before/After 포함 시)
infer_report.txt	추론 결과 리포트
plots/	손실 곡선, 분류 곡선(PR/ROC), 히스토그램, Confusion, 샘플 플롯

⚡ 성능 최적화 팁
Tensor Core(Ampere+)에서 속도 향상을 원하시면 시작부에 아래를 권장합니다:

python
코드 복사
import torch
torch.set_float32_matmul_precision("high")
nn.Linear, Attention 등에서 FP32→TF32 텐서코어 사용으로 20~40% 가속 기대

--amp는 CUDA/MPS 등 가능한 디바이스에서만 활성화됩니다. CPU에서는 자동으로 비활성.

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
🧩 모드 요약
모드	설명	주요 출력
train	새 모델 학습	model.pt, train_report.txt, plots/
finetune	기존 모델 이어 학습	동일 폴더 내 재저장
infer	추론 및 리포트	infer_report.txt, plots/

