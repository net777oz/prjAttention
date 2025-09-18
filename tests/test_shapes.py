from ttm_flow.data import gen_mock_batch, gen_future_labels_from_series

def test_shapes():
    """
    오프라인 단위 테스트:
    - 데이터 생성과 라벨 생성 shape만 검증
    """
    B, T, C, H = 8, 64, 3, 12
    X, mask = gen_mock_batch(B, T, C=C, missing_prob=0.0, seed=123)
    assert X.shape == (B, T, C)
    assert mask.shape == (B, T)
    y = gen_future_labels_from_series(X, H)
    assert y.shape == (B, H, C)
