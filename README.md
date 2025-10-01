# llm_ts — 단일 CSV/단일 채널 LLM-스타일 시계열 Transformer

## 디렉터리 구조

llm_ts/
├─ run_llmts.py # 엔트리포인트
├─ cli.py # argparse & main
├─ pipeline.py # E2E 오케스트레이션
├─ config.py # 상수
├─ data.py # CSV 로딩/시드/출력폴더
├─ windows.py # 롤링 윈도우
├─ splits.py # split-mode 구현
├─ losses.py # SoftF1, BCE, pos_weight
├─ metrics.py # 지표/τ 탐색
├─ evaler.py # OOM-안전 평가
├─ trainer.py # AMP/compile 대응 학습 루프
├─ viz.py # 플롯(샘플/PR/ROC/Hist/Confusion/Loss)
└─ utils.py # 공통 유틸 & rich 콘솔

╔═══════════════════════════════════════════════════════════════════════════╗
║ SYSTEM OVERVIEW							    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE 단일 CSV(헤더/인덱스 없음, [N,T]) 기반의 시계열 예측/분류 파이프  ║
║ MODES --mode {train, finetune, infer} 				    ║
║ TASKS --task {regress, classify}					    ║
║ CONTEXT 롤링 윈도우 teacher-forcing 유지, τ는 검증 F1 최대 기준 선택      ║
║ OUTPUTS artifacts/<run>/{model.pt, _report.txt, plots/.png}		    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ MODULE GRAPH								    ║
║ run_llmts → cli → pipeline → (config, data, windows, splits, trainer,	    ║
║ evaler, metrics, losses, viz, utils)					    ║
║ 모델 로드: ttm_flow.model.load_model_and_cfg(in_channels=1)		    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ KEY INTERFACES							    ║
║ data.parse_csv_single(args) -> Tensor[N,1,T]				    ║
║ windows.build_windows_dataset(X,L) -> (Xw[NW,1,L], Yw[NW,1], groups,W)    ║
║ splits.make_splits(...) -> (train_idx, val_idx)			    ║
║ trainer.train_all_epochs(...) -> List[float] (epoch loss history)	    ║
║ evaler.eval_model(...) -> (mse, mae, y_true, y_prob)			    ║
║ metrics.find_best_threshold_for_f1(...) -> (tau, f1)			    ║
║ viz.plot_* -> PNG 산출						    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ IN/OUT SUMMARY							    ║
║ IN : CSV [N,T], CLI 하이퍼파라미터					    ║
║ OUT: 모델 가중치, 성능 리포트, 플롯					    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ SIDE EFFECTS								    ║
║ - artifacts/<run>/ 하위 파일/폴더 생성				    ║
║ - CUDA/torch.compile/AMP 사용 시 그래프 캐시/메모리 점유		    ║
╠───────────────────────────────────────────────────────────────────────────╣
║ EXCEPTIONS								    ║
║ - CSV 형식 오류/CKPT 로드 실패/장치 미지원 → 오류 메시지 후 종료	    ║
╚════════════════════════════════════════════════════════════════════════════


## 설치 요구

- Python 3.10+
- PyTorch (CUDA 선택)
- numpy, matplotlib, scikit-learn(옵션: group split 사용 시)
- rich(옵션: 콘솔 UI)
- 외부 모델 로더: `ttm_flow.model.load_model_and_cfg` (변경 없음)

## 빠른 시작

```bash
# 학습
python run_llmts.py \
  --mode train --task classify \
  --csv ./data/your.csv \
  --context-len 31 \
  --epochs 3 \
  --batch-size 4096 \
  --alpha 0.5 --pos-weight global \
  --split-mode group

# 파인튜닝
python run_llmts.py \
  --mode finetune --task classify \
  --csv ./data/your.csv \
  --context-len 31 \
  --epochs 1 \
  --ckpt ./artifacts/prev/model.pt

# 추론
python run_llmts.py \
  --mode infer --task classify \
  --csv ./data/your.csv \
  --context-len 31 \
  --ckpt ./artifacts/model.pt

