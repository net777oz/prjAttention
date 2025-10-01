Train (from scratch)
분류(classify) + F1 단독(α=0) + group-aware split + AMP + compile
python run_llmts.py --mode train --task classify \
  --csv-list data/SPa.csv,data/SPb.csv,data/SPc.csv \
  --context-len 32 --epochs 50 \
  --alpha 0 --pos-weight global --val-ratio 0.2 \
  --split-mode group \
  --batch-size 4096 --eval-batch-size 4096 \
  --lr 2e-4 --weight-decay 0.01 \
  --amp --compile reduce-overhead \
  --out cls_alpha0_ctx32


산출물: artifacts/cls_alpha0_ctx32/

model.pt, train_report.txt, plots/(confusion, PR/ROC, F1-vs-τ, precision/recall-vs-τ, prob_hist, train_loss 등)

Finetune (pretrained 가중치 이어 학습)
분류(classify) + 하이브리드(α=0.5) + group-aware split
python run_llmts.py --mode finetune --task classify \
  --csv-list data/SPa.csv,data/SPb.csv,data/SPc.csv \
  --context-len 32 --epochs 10 \
  --alpha 0.5 --pos-weight global --val-ratio 0.2 \
  --split-mode group \
  --batch-size 4096 --eval-batch-size 4096 \
  --lr 1e-4 --weight-decay 0.01 \
  --amp --compile reduce-overhead \
  --ckpt artifacts/cls_alpha0_ctx32/model.pt \
  --out cls_finetune_alpha05_ctx32


--ckpt에 이전 train에서 저장한 model.pt 경로 지정.

Infer (추론만)
분류(classify) + 임계값(τ) 기본값 지정
python run_llmts.py --mode infer --task classify \
  --csv-list data/SPa.csv,data/SPb.csv,data/SPc.csv \
  --context-len 32 \
  --thresh-default 0.99 \
  --ckpt artifacts/cls_alpha0_ctx32/model.pt \
  --out cls_infer_ctx32_t099


산출물: infer_report.txt(F1/Acc/P/R, MSE/MAE(참고)), plots/(confusion/PR/ROC/F1-vs-τ/등), model.pt(동일 가중치 백업)

참고 옵션 빠르게 정리

--task regress로 바꾸면 회귀 모드. 리포트는 MSE/MAE, plots/에 시계열 예측 PNG(*_rowXXXX.png) 생성.

--split-mode:

group(기본, 권장): 같은 아이템(행)에서 나온 윈도우를 함께 묶어 누수 방지

item: 아이템 단위 분할

time: 각 아이템 내 앞/뒤 시점 분할(겹침 없음)

window: 윈도우 단위 랜덤(누수 위험)

--alpha:

분류에서 손실 = alpha*BCE + (1-alpha)*(1-SoftF1)

0이면 F1만, 1이면 BCE만

--pos-weight {global|batch|none}: 심각한 불균형일 때 global 추천

--amp: 혼합정밀. 켜는 걸 기본 권장

--compile {reduce-overhead|max-autotune}: PyTorch 2.x 컴파일러. 문제시 빈 문자열로 비활성화


👍 회귀 모드(regress) 실행 예시 3종류(train / finetune / infer) 정리

🔹 Train (from scratch, 회귀)
순수 MSE 학습 + group-aware split + AMP + compile
python run_llmts.py --mode train --task regress \
  --csv-list data/SPa.csv,data/SPb.csv,data/SPc.csv \
  --context-len 32 --epochs 50 \
  --batch-size 4096 --eval-batch-size 4096 \
  --lr 2e-4 --weight-decay 0.01 \
  --split-mode group \
  --amp --compile reduce-overhead \
  --out reg_ctx32


산출물: artifacts/reg_ctx32/

model.pt, train_report.txt(before/after MSE·MAE),

plots/after_rowXXXX.png (시계열 그래프: 파란색=실제, 주황색 점선=예측)

🔹 Finetune (pretrained 이어 학습, 회귀)
이전 학습된 모델(reg_ctx32) 이어서 fine-tune
python run_llmts.py --mode finetune --task regress \
  --csv-list data/SPa.csv,data/SPb.csv,data/SPc.csv \
  --context-len 32 --epochs 10 \
  --batch-size 4096 --eval-batch-size 4096 \
  --lr 1e-4 --weight-decay 0.01 \
  --split-mode group \
  --amp --compile reduce-overhead \
  --ckpt artifacts/reg_ctx32/model.pt \
  --out reg_finetune_ctx32


--ckpt에 이전 학습 결과의 model.pt 경로를 지정.

새로 학습한 결과는 artifacts/reg_finetune_ctx32/에 저장.

🔹 Infer (추론, 회귀)
학습된 모델로 새 CSV 데이터 평가 + 그래프 저장
python run_llmts.py --mode infer --task regress \
  --csv-list data/SPa.csv,data/SPb.csv,data/SPc.csv \
  --context-len 32 \
  --ckpt artifacts/reg_ctx32/model.pt \
  --out reg_infer_ctx32


산출물: infer_report.txt (MSE·MAE),

plots/infer_rowXXXX.png (실제 vs 예측 그래프)

📌 분류 vs 회귀 차이 정리
항목	분류(classify)	회귀(regress)
손실 함수	α·BCE + (1-α)·(1-SoftF1)	MSE
지표	Accuracy, Precision, Recall, F1, Confusion 등	MSE, MAE
τ (임계값)	검증셋에서 F1 최적화로 자동 선택	없음
결과 플롯	Confusion matrix, PR curve, ROC, prob_hist 등
