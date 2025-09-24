# AttentionProject - LLM-TS Backbone

시계열 Transformer (llm_ts) 백본을 이용한 **제로부터 학습 및 평가** 파이프라인.


## 3. From-scratch train
python -u run_llmts.py \
  --mode train \
  --backbone llm_ts \
  --context-len 32 \
  --csv1 data/varA.csv --csv2 data/varB.csv --csv3 data/varC.csv \
  --out artifacts/train_l32 \
  --epochs 60 --lr 2e-4 --weight-decay 0.01 \
  --batch-size 4096 --num-workers 4 --amp --compile reduce-overhead \
  --plot-samples 3 --log-every 20


## 4. Finetune (사전학습 ckpt 로드 → 추가 학습)
python -u run_llmts.py \
  --mode finetune \
  --ckpt artifacts/train_l32/model.pt \
  --backbone llm_ts \
  --context-len 32 \
  --csv1 data/varA.csv --csv2 data/varB.csv --csv3 data/varC.csv \
  --out artifacts/finetune_l32 \
  --epochs 5 --lr 1e-4 --weight-decay 0.01 \
  --batch-size 2048 --num-workers 4 --amp \
  --plot-samples 3


## 5. Infer (추론 전용 / 학습 없음, 리치 박스 UI + PNG)
python -u run_llmts.py \
  --mode infer \
  --ckpt artifacts/train_l32/model.pt \
  --backbone llm_ts \
  --context-len 32 \
  --csv1 data/varA.csv --csv2 data/varB.csv --csv3 data/varC.csv \
  --out artifacts/infer_l32 \
  --plot-samples 5
