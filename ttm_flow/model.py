# -*- coding: utf-8 -*-
# 통합 모델 로더 (기존 함수명 유지)
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

# 기존 다른 백본 임포트 ...
# from ttm_flow.backbones.ttm_r1 import TTM_R1
# from ttm_flow.backbones.ttm_r2 import TTM_R2
# 등등...

# 새 백본
from ttm_flow.backbones.llm_ts import LLMTimeSeries

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_model_and_cfg(
    backbone: str = "llm_ts",
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    backbone: "llm_ts", "ttm_r1", "ttm_r2", ...
    kwargs:
        - in_channels (기본 3)
        - d_model, n_layer, n_head, mlp_ratio, dropout, max_len
        - 기타 기존 백본이 쓰는 인자들
    """
    backbone = (backbone or "llm_ts").lower()

    if backbone == "llm_ts":
        model = LLMTimeSeries(
            in_channels=kwargs.get("in_channels", 3),
            d_model=kwargs.get("d_model", 256),
            n_layer=kwargs.get("n_layer", 6),
            n_head=kwargs.get("n_head", 8),
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
            dropout=kwargs.get("dropout", 0.0),
            max_len=kwargs.get("max_len", 4096),
        )
        cfg = dict(
            backbone="llm_ts",
            in_channels=kwargs.get("in_channels", 3),
            d_model=kwargs.get("d_model", 256),
            n_layer=kwargs.get("n_layer", 6),
            n_head=kwargs.get("n_head", 8),
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
            dropout=kwargs.get("dropout", 0.0),
            max_len=kwargs.get("max_len", 4096),
        )
        return model, cfg

    # -----------------------------
    # 기존 TTM/다른 백본 로딩 분기들
    # -----------------------------
    # elif backbone == "ttm_r1":
    #     model = TTM_R1(...)
    #     cfg = {...}
    #     return model, cfg
    #
    # elif backbone == "ttm_r2":
    #     model = TTM_R2(...)
    #     cfg = {...}
    #     return model, cfg
    #
    # ...
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

# -----------------------------------------------------------------------------
# Embedding 추출 어댑터 (기존 함수명 유지)
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings_ttm(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    기존 파이프라인에서 호출하던 임베딩 추출 함수 이름을 그대로 유지합니다.
    - 모델이 extract_embeddings(x) 메서드를 제공하면 그걸 사용
    - 아니면 마지막 히든을 평균해서 근사 (fallback)
    입력: x [B, C, T]
    출력: feats [B, D]
    """
    if hasattr(model, "extract_embeddings"):
        return model.extract_embeddings(x)

    # Fallback: 모델 forward가 (pred, feats) 형태를 낸다면 feats를 평균
    try:
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            feats = out[1]  # [B, T, D]
            if feats.ndim == 3:
                return feats.mean(dim=1)
    except Exception:
        pass

    # 최후의 수단: 입력 평균을 임베딩처럼 반환(차원 낮지만 파이프라인 안전성 확보)
    return x.mean(dim=2)
