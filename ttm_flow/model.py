# 새 백본
from ttm_flow.backbones.llm_ts import LLMTimeSeries
# LSTM 추가
from ttm_flow.backbones.lstm import build_lstm_head

def load_model_and_cfg(backbone: str = "llm_ts", **kwargs):
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

    elif backbone == "lstm":
        model = build_lstm_head(
            in_channels=kwargs.get("in_channels", 3),
            hidden=kwargs.get("lstm_hidden", 128),
            num_layers=kwargs.get("lstm_layers", 2),
            out_dim=kwargs.get("out_dim", 1),
            dropout=kwargs.get("lstm_dropout", 0.0),
        )
        cfg = dict(
            backbone="lstm",
            in_channels=kwargs.get("in_channels", 3),
            hidden=kwargs.get("lstm_hidden", 128),
            num_layers=kwargs.get("lstm_layers", 2),
            out_dim=kwargs.get("out_dim", 1),
            dropout=kwargs.get("lstm_dropout", 0.0),
        )
        return model, cfg

    else:
        raise ValueError(f"Unknown backbone: {backbone}")
