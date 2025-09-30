python run_llmts.py \
  --mode train \
  --task classify \
  --csv ./data/SPa.csv \
  --context-len 31 \
  --epochs 60 \
  --batch-size 4096 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --pos-weight global \
  --alpha 0.5 \
  --bin-rule nonzero \
  --thresh-default 0.5 \
  --split-mode group \
  --val-ratio 0.2 \
  --amp \
  --compile reduce-overhead \
  --out train_cls_ctx31_e10


python run_llmts.py \
  --mode finetune \
  --task classify \
  --csv ./data/SD0a.csv \
  --ckpt ./artifacts/train_cls_ctx31_e10/model.pt \
  --context-len 31 \
  --epochs 0 \
  --batch-size 4096 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --pos-weight global \
  --alpha 0.6 \
  --bin-rule nonzero \
  --thresh-default 0.5 \
  --split-mode group \
  --val-ratio 0.2 \
  --amp \
  --compile reduce-overhead \
  --out ft_cls_ctx31_e5
