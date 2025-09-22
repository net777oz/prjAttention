# ttm_flow/data.py
# - 모의 3변량 시계열 생성(gen_mock_batch)
# - 미래 라벨 생성(gen_future_labels_from_series)
# - (선택) 이상치 주입(inject_anomalies)
# - 롤링 윈도우 생성(make_rolling_windows)
# - CSV 3종 로드(load_triplet_csvs, gen_from_csv3)

import numpy as np

# =========================
# 1) 합성/데모용 유틸
# =========================

def gen_mock_batch(B: int, T: int, C: int = 3, missing_prob: float = 0.0, seed: int | None = 777):
    """
    간단한 모의 3변량 시계열:
    - 각 채널: 서로 다른 주기/위상 + 완만한 추세 + 가우시안 노이즈
    - 일부 missing을 True로 마킹(값은 그대로 두며, 필요시 후처리에서 채움)
    반환: X [B,T,C], mask [B,T] (True=관측)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)

    X = np.zeros((B, T, C), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            freq = 2.0 * np.pi / (64 + 16 * c)  # 서로 다른 주기
            phase = rng.uniform(0, 2 * np.pi)
            trend = 0.001 * (t - T / 2)         # 완만한 추세
            noise = rng.normal(0, 0.2, size=T)
            X[b, :, c] = np.sin(freq * t + phase) + trend + noise

    mask = np.ones((B, T), dtype=bool)
    if missing_prob > 0:
        miss = rng.random((B, T)) < missing_prob
        mask[miss] = False
        # 값까지 0으로 채우려면 아래 주석 해제
        # X[miss, :] = 0.0

    return X, mask


def gen_future_labels_from_series(X: np.ndarray, H: int):
    """
    데모용 '미래 라벨' 생성기:
    - 각 시리즈의 마지막 H 구간을 복제
    반환: y [B,H,C]
    """
    B, T, C = X.shape
    assert H <= T, "H must be <= T"
    return X[:, -H:, :].copy()


def inject_anomalies(
    X,
    mask,
    spike_prob=0.02,
    drop_block_prob=0.1,
    max_drop_len=12,
    regime_switch_prob=0.05,
    seed=777,
):
    """
    (선택) 이상치 주입: 스파이크, 짧은 블록 드랍(값=0), 레짐 전환
    """
    rng = np.random.default_rng(seed)
    B, T, C = X.shape
    X = X.copy()
    mask = mask.copy()

    # 1) 랜덤 스파이크
    spike = rng.random((B, T, C)) < spike_prob
    sigma = X.std(axis=(0, 1), keepdims=True) + 1e-6
    # 주의: 브로드캐스트를 위해 평탄화 인덱싱 대신 마스크로 더하기
    X[spike] += rng.choice([-1, 1], size=spike.sum()) * rng.uniform(3, 6, size=spike.sum())

    # 2) 짧은 드랍 블록(0으로 떨어뜨림, 관측은 True로 유지)
    for b in range(B):
        if rng.random() < drop_block_prob:
            s = rng.integers(0, max(1, T - max_drop_len))
            L = int(rng.integers(3, max(4, max_drop_len)))
            e = min(T, s + L)
            X[b, s:e, :] = 0.0
            mask[b, s:e] = True

    # 3) 레짐 전환(평균/분산 변경)
    for b in range(B):
        if rng.random() < regime_switch_prob:
            k = rng.integers(T // 4, 3 * T // 4)
            scale = rng.uniform(0.5, 1.8)
            shift = rng.uniform(-1.5, 1.5)
            X[b, k:, :] = X[b, k:, :] * scale + shift

    return X, mask


# =========================
# 2) 롤링 윈도우
# =========================

def make_rolling_windows(
    X: np.ndarray,
    context_len: int,
    horizon: int = 1,
    step: int = 16,
):
    """
    컨텍스트 길이를 고정하고, 롤링으로 다음 horizon 스텝 라벨을 만듭니다.
    - X: [B, T, C]
    - 반환: Xw [N, context_len, C], Yw [N, horizon, C]
    """
    B, T, C = X.shape
    if T < context_len + horizon:
        raise ValueError(
            f"Time length too short: T={T}, requires >= context_len+horizon = {context_len + horizon}. "
            "Increase T (e.g., generate longer series) or reduce context_len."
        )

    X_list, Y_list = [], []
    for b in range(B):
        for i in range(0, T - context_len - horizon + 1, step):
            x_win = X[b, i : i + context_len, :]
            y_win = X[b, i + context_len : i + context_len + horizon, :]
            X_list.append(x_win)
            Y_list.append(y_win)

    Xw = np.stack(X_list, axis=0).astype(np.float32)
    Yw = np.stack(Y_list, axis=0).astype(np.float32)
    return Xw, Yw


# =========================
# 3) CSV 3종 로더
# =========================

def _load_csv(path: str, delimiter: str = ",", skiprows: int = 0, dtype=np.float32) -> np.ndarray:
    """
    단일 CSV 로드 유틸.
    - 기본 가정: 헤더/인덱스 없음 (필요 시 skiprows>0)
    - 반환 shape: (B, T) 2D
    """
    try:
        arr = np.loadtxt(path, delimiter=delimiter, dtype=dtype, skiprows=skiprows)
    except Exception:
        # 다양한 결측/비정형에 좀 더 관대한 로더
        arr = np.genfromtxt(path, delimiter=delimiter, dtype=dtype, skip_header=skiprows)
    if arr.ndim != 2:
        raise ValueError(f"CSV must be 2D (items x timesteps). Got shape {arr.shape} for {path}")
    return arr


def _ffill_rows_2d(arr: np.ndarray) -> np.ndarray:
    """
    NaN을 각 행(row) 기준 시간방향으로 forward-fill.
    선두 NaN은 그대로 남습니다(필요시 별도 채움).
    """
    out = arr.copy()
    B, T = out.shape
    for b in range(B):
        row = out[b]
        # 유효값 인덱스
        idx = np.where(np.isfinite(row))[0]
        if idx.size == 0:
            continue
        last = idx[0]
        for t in range(last + 1, T):
            if not np.isfinite(row[t]):
                row[t] = row[last]
            else:
                last = t
        out[b] = row
    return out


def load_triplet_csvs(
    csv1: str,
    csv2: str,
    csv3: str,
    *,
    delimiter: str = ",",
    skiprows: int = 0,
    dtype=np.float32,
    validate_shapes: bool = True,
    nan_policy: str = "keep",   # {"keep","zero","mean","ffill"}
    fill_value: float = 0.0,
):
    """
    CSV 3개(varA/varB/varC)를 읽어 (B,T,3) 배열을 반환합니다.
    - 각 CSV는 2D (B,T) 이어야 하며, 기본적으로 헤더/인덱스가 없다고 가정합니다.
    - B: 아이템 수(행), T: 시점 수(열), C=3: 변수 수(파일 개수)

    nan_policy:
      - "keep": NaN 유지 (후단에서 처리)
      - "zero": NaN → 0.0
      - "mean": NaN → 전체 평균(해당 CSV 전역 평균)
      - "ffill": NaN → 각 행 기준 forward-fill (선두 NaN은 남을 수 있음)

    반환:
      X: np.ndarray, shape (B,T,3), dtype=float32
    """
    A = _load_csv(csv1, delimiter=delimiter, skiprows=skiprows, dtype=dtype)
    Bm = _load_csv(csv2, delimiter=delimiter, skiprows=skiprows, dtype=dtype)
    C = _load_csv(csv3, delimiter=delimiter, skiprows=skiprows, dtype=dtype)

    if validate_shapes and not (A.shape == Bm.shape == C.shape):
        raise ValueError(f"CSV shapes must match: A={A.shape}, B={Bm.shape}, C={C.shape}")

    def _apply_nan_policy(arr: np.ndarray) -> np.ndarray:
        if nan_policy == "keep":
            return arr
        if nan_policy == "zero":
            out = arr.copy()
            out[~np.isfinite(out)] = fill_value
            return out
        if nan_policy == "mean":
            out = arr.copy()
            m = np.nanmean(out[np.isfinite(out)]) if np.isfinite(out).any() else 0.0
            out[~np.isfinite(out)] = m
            return out
        if nan_policy == "ffill":
            # ffill 전에 inf도 NaN으로 취급
            out = arr.copy()
            out[~np.isfinite(out)] = np.nan
            out = _ffill_rows_2d(out)
            return out
        raise ValueError(f"Unknown nan_policy: {nan_policy}")

    A = _apply_nan_policy(A)
    Bm = _apply_nan_policy(Bm)
    C = _apply_nan_policy(C)

    X = np.stack([A, Bm, C], axis=-1).astype(np.float32)  # (B,T,3)
    return X


def gen_from_csv3(
    csv1: str,
    csv2: str,
    csv3: str,
    *,
    delimiter: str = ",",
    skiprows: int = 0,
    dtype=np.float32,
    nan_policy: str = "keep",     # {"keep","zero","mean","ffill"}
    fill_value: float = 0.0,
):
    """
    gen_mock_batch 대체용: CSV 3종을 읽어서 (X, mask)를 반환합니다.
      - X: (B,T,3) float32
      - mask: (B,T) bool (해당 시점의 3채널 모두 finite일 때만 True)
    """
    X = load_triplet_csvs(
        csv1,
        csv2,
        csv3,
        delimiter=delimiter,
        skiprows=skiprows,
        dtype=dtype,
        validate_shapes=True,
        nan_policy=nan_policy,
        fill_value=fill_value,
    )
    # 마스크: 3채널 모두 유효할 때만 True
    mask = np.isfinite(X).all(axis=-1)
    return X, mask
