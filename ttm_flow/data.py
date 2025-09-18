# ttm_flow/data.py
# - 모의 3변량 시계열 생성(gen_mock_batch)
# - 미래 라벨 생성(gen_future_labels_from_series)
# - (선택) 이상치 주입(inject_anomalies)
# - 롤링 윈도우 생성(make_rolling_windows)
import numpy as np

def gen_mock_batch(B: int, T: int, C: int = 3, missing_prob: float = 0.0, seed: int | None = 777):
    """
    간단한 모의 3변량 시계열:
    - 각 채널: 서로 다른 주기/위상 + 완만한 추세 + 가우시안 노이즈
    - 일부 missing을 True로 마킹(값은 0으로 채워 넣을 수도 있음 — 여기선 값은 그대로)
    반환: X [B,T,C], mask [B,T] (True=관측)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)

    X = np.zeros((B, T, C), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            freq = 2.0 * np.pi / (64 + 16*c)    # 서로 다른 주기
            phase = rng.uniform(0, 2*np.pi)
            trend = 0.001 * (t - T/2)           # 완만한 추세
            noise = rng.normal(0, 0.2, size=T)
            X[b, :, c] = np.sin(freq * t + phase) + trend + noise

    # 관측 마스크 생성
    mask = np.ones((B, T), dtype=bool)
    if missing_prob > 0:
        miss = rng.random((B, T)) < missing_prob
        mask[miss] = False
        # 필요하다면 실제 값도 0으로 만들려면:
        # X[miss, :] = 0.0

    return X, mask

def gen_future_labels_from_series(X: np.ndarray, H: int):
    """
    단순한 '미래 라벨' 생성기:
    - 각 시리즈의 마지막 H 구간을 복제
    - 실제론 과거 구간에서 예측 타깃을 만들겠지만, 데모에선 형태만 맞춤
    반환: y [B,H,C]
    """
    B, T, C = X.shape
    assert H <= T, "H must be <= T"
    return X[:, -H:, :].copy()

def inject_anomalies(X, mask, spike_prob=0.02, drop_block_prob=0.1, max_drop_len=12, regime_switch_prob=0.05, seed=777):
    """
    (선택) 이상치 주입: 스파이크, 짧은 블록 드랍, 레짐 전환
    """
    rng = np.random.default_rng(seed)
    B, T, C = X.shape
    X = X.copy()
    mask = mask.copy()

    # 1) 랜덤 스파이크
    spike = rng.random((B, T, C)) < spike_prob
    sigma = X.std(axis=(0,1), keepdims=True) + 1e-6
    X[spike] += rng.choice([-1,1], size=spike.sum()) * rng.uniform(3,6, size=spike.sum()) * sigma.reshape(-1)[:spike.sum()]

    # 2) 짧은 드랍 블록(0으로 떨어뜨림)
    for b in range(B):
        if rng.random() < drop_block_prob:
            s = rng.integers(0, max(1, T-max_drop_len))
            L = int(rng.integers(3, max(4, max_drop_len)))
            e = min(T, s+L)
            X[b, s:e, :] = 0.0
            mask[b, s:e] = True  # 관측은 됐다고 가정(값이 0으로 드랍)

    # 3) 레짐 전환(평균/분산 변경)
    for b in range(B):
        if rng.random() < regime_switch_prob:
            k = rng.integers(T//4, 3*T//4)
            scale = rng.uniform(0.5, 1.8)
            shift = rng.uniform(-1.5, 1.5)
            X[b, k:, :] = X[b, k:, :] * scale + shift

    return X, mask

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
