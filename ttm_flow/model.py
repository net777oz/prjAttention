# ttm_flow/model.py
# TinyTimeMixer(TTM-R2) 모델 로드/정합/임베딩 추출 유틸
# - granite-tsfm의 get_model() 경로 사용 (AutoModel/AutoConfig 사용 X)

from typing import Optional, Tuple
import numpy as np
import torch

# ✅ granite-tsfm 툴킷의 공식 유틸
from tsfm_public.toolkit.get_model import get_model


def load_model_and_cfg(model_id: str, context_len: int, pred_len: int, device: str = "cuda"):
    """
    granite-tsfm의 get_model()로 TTM-R2를 로드합니다.
    - context_len / pred_len 조합에 맞는 브랜치를 자동 선택합니다.
    - 반환: (model, cfg)  (cfg는 없을 수도 있으니 None 가능)
    """
    model = get_model(
        model_path=model_id,
        model_name="ttm",
        context_length=context_len,
        prediction_length=pred_len,
    )
    model = model.to(device)
    cfg = getattr(model, "config", None)
    return model, cfg


def match_context_and_horizon(
    X: np.ndarray, mask: np.ndarray, y: Optional[np.ndarray], cfg
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    TTM-R2의 forward는 mask= 인자를 받지 않습니다.
    여기서는 단순히 numpy→torch 변환만 수행합니다. (패딩/트렁크 없음)
    - X: [B, T, C], mask: [B, T], y: [B, H, C] (옵션)
    - 반환: (X_t, mask_t, y_t)  # mask_t는 이후 사용하지 않더라도 형태 유지
    """
    X_t = torch.from_numpy(X).to(torch.float32)
    mask_t = torch.from_numpy(mask).to(torch.bool)
    y_t = torch.from_numpy(y).to(torch.float32) if y is not None else None
    return X_t, mask_t, y_t


@torch.no_grad()
def extract_embeddings_ttm(out, X: torch.Tensor) -> torch.Tensor:
    """
    TinyTimeMixer 출력 객체에서 임베딩을 추출합니다(강건 버전).
    - out.hidden_states[-1]이 3D([B,T,D])면: T-mean + 마지막 시점 concat
    - out.hidden_states[-1]이 4D([B,C,R,D])면: (C,R)-mean + 마지막 R(채널평균) concat
    - out.decoder_hidden_state가 있으면: 마지막 디코더 스텝을 추가 concat
    반환: [B, D*(2 또는 3)]
    """
    parts = []

    # 1) encoder/backbone 측 hidden states
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) == 0:
        raise RuntimeError("hidden_states가 없습니다. model(..., output_hidden_states=True)로 호출했는지 확인하세요.")
    H_last = hs[-1]

    if H_last.dim() == 3:
        # [B, T, D]
        B, T, D = H_last.shape
        mean_enc = H_last.mean(dim=1)                 # [B, D]
        last_idx = X.shape[1] - 1
        last_enc = H_last[:, last_idx, :]             # [B, D]
        parts += [mean_enc, last_enc]
    elif H_last.dim() == 4:
        # [B, C, R, D]  (채널×해상도 피라미드)
        B, C, R, D = H_last.shape
        mean_enc = H_last.mean(dim=(1, 2))            # [B, D]  C,R 평균
        last_r   = H_last[:, :, -1, :].mean(dim=1)    # [B, D]  마지막 R에서 채널 평균
        parts += [mean_enc, last_r]
    else:
        raise RuntimeError(f"예상치 못한 hidden_states[-1] shape: {tuple(H_last.shape)}")

    # 2) decoder 측 hidden state (있으면 추가)
    dec = getattr(out, "decoder_hidden_state", None)
    if dec is not None:
        if dec.dim() == 3:
            parts.append(dec[:, -1, :])               # [B, D]
        elif dec.dim() == 4:
            parts.append(dec.mean(dim=1)[:, -1, :])   # [B, D]

    emb = torch.cat(parts, dim=-1)                    # [B, *]
    return emb
