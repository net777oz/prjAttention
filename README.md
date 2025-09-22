# AttentionProject - LLM-TS Backbone

시계열 Transformer (llm_ts) 백본을 이용한 **제로부터 학습 및 평가** 파이프라인.

---

## 1. 학습 (제로부터 시작)

3변량 CSV (`varA.csv`, `varB.csv`, `varC.csv`)를 입력으로  
`context_len=32`, `epochs=500` 조건으로 학습:

```bash
python run_finetune_1step.py \
  --backbone llm_ts \
  --context-len 32 \
  --csv1 data/varA.csv --csv2 data/varB.csv --csv3 data/varC.csv \
  --out artifacts/llm_ts_e500_l32 \
  --epochs 500 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --plot-samples 3 \
  --log-every 5

## 1. 학습 후 평가 (체크포인트 사용)

python run_eval_rolling_3csv.py \
  --backbone llm_ts \
  --context-len 32 \
  --csv1 data/varA.csv --csv2 data/varB.csv --csv3 data/varC.csv \
  --ckpt artifacts/llm_ts_e500_l32/model.pt \
  --out artifacts/llm_ts_e500_l32_eval \
  --plot-samples 5 \
  --save-csv
