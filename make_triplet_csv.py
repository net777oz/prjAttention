# make_triplet_csv.py
# gen_mock_batch(B,T,C)을 이용해 3개의 CSV(varA,varB,varC)를 생성
# - 각 CSV shape: (B, T) = (행=아이템, 열=시간)
# - 헤더/인덱스 없음, 구분자 ','

import os
import argparse
import numpy as np
from ttm_flow.data import gen_mock_batch

def save_matrix_csv(mat: np.ndarray, path: str, fmt: str = "%.6g", delimiter: str = ","):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # 헤더/인덱스 없이 저장
    np.savetxt(path, mat, fmt=fmt, delimiter=delimiter)
    print(f"[write] {path}  shape={mat.shape}")

def main():
    p = argparse.ArgumentParser(description="Generate 3 CSVs (varA/B/C) from gen_mock_batch(B,T,C).")
    p.add_argument("--B", type=int, default=4200, help="아이템 수 (rows per CSV)")
    p.add_argument("--T", type=int, default=220,  help="타임스텝 수 (columns per CSV)")
    p.add_argument("--C", type=int, default=3,    help="채널(변수) 수 (기본 3; 3개 CSV 생성)")
    p.add_argument("--missing_prob", type=float, default=0.0, help="결측 마스크 비율(값은 그대로; 필요시 후처리)")
    p.add_argument("--seed", type=int, default=777, help="난수 시드")
    p.add_argument("--outdir", type=str, default="data", help="CSV 출력 폴더")
    p.add_argument("--nameA", type=str, default="varA.csv", help="첫 번째 채널 파일명")
    p.add_argument("--nameB", type=str, default="varB.csv", help="두 번째 채널 파일명")
    p.add_argument("--nameC", type=str, default="varC.csv", help="세 번째 채널 파일명")
    p.add_argument("--fmt", type=str, default="%.6g", help="np.savetxt 포맷 (예: %.6f, %.6g)")
    args = p.parse_args()

    if args.C < 3:
        raise ValueError(f"C(채널 수)는 최소 3이어야 합니다. 받음: {args.C}")

    # 1) 모의 데이터 생성: X [B,T,C]
    X, mask = gen_mock_batch(B=args.B, T=args.T, C=args.C, missing_prob=args.missing_prob, seed=args.seed)
    # 필요시: 값까지 0으로 채우려면 gen_mock_batch 내부 주석 라인 활성화

    # 2) 첫 3채널을 각각 CSV로 저장
    A = X[:, :, 0]                      # (B,T)
    B = X[:, :, 1]                      # (B,T)
    C = X[:, :, 2]                      # (B,T)

    save_matrix_csv(A, os.path.join(args.outdir, args.nameA), fmt=args.fmt)
    save_matrix_csv(B, os.path.join(args.outdir, args.nameB), fmt=args.fmt)
    save_matrix_csv(C, os.path.join(args.outdir, args.nameC), fmt=args.fmt)

    # 참고 출력
    print(f"[done] Generated 3 CSVs in '{args.outdir}'")
    print(f"        B={args.B}, T={args.T}, C={args.C}  ->  each CSV shape = ({args.B}, {args.T})")
    print(f"        Use these as inputs to run_demo.py (csv mode).")

if __name__ == "__main__":
    main()
