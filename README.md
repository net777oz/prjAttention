# TTM Flow Demo (compact 4-module version)

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Rolling Evaluation with 3 CSV Inputs

이 저장소는 **세 가지 시계열 CSV 데이터**(`--csv1`, `--csv2`, `--csv3`)를 입력으로 받아,  
IBM Granite Time Series Transformer 모델을 이용해 순차 예측(rolling evaluation)을 수행하는 스크립트 예제입니다.  

---

## 🚀 실행 방법

### 1. 기본 명령
```bash
python run_eval_rolling_3csv.py \
  --csv1 data/varA.csv \
  --csv2 data/varB.csv \
  --csv3 data/varC.csv \
  --model-id ibm-granite/granite-timeseries-ttm-r2 \
  --ckpt checkpoints/ttm_r2_ft1step_best.pt \
  --out artifacts/rolling_eval \
  --device cuda:0 \
  --save-csv

#
집거 그냥 옮기는 중 깃테스트
