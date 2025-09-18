# ttm_flow/viz.py
# - 라인 플롯(예측 vs 정답) PNG 저장: plot_predictions
# - 배치 샘플 시계열 저장(옵션): plot_batch_examples
# - UMAP 2D/3D 임베딩 시각화 저장: umap_scatter_2d / umap_scatter_3d

from __future__ import annotations
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    """경로에 맞춰 디렉터리를 생성합니다."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outpath: str,
    title: str = "Prediction vs True",
    sample_idx: int = 0,
    channels: Optional[list[int]] = None,
):
    """
    Ground Truth vs Prediction을 PNG로 저장합니다.
    - y_true: [B, H, C]
    - y_pred: [B, H, C]
    - sample_idx: 배치에서 시각화할 샘플 인덱스
    - channels: None이면 모든 채널을 겹쳐 그림. [0,2]처럼 지정 가능.
    """
    assert y_true.ndim == 3 and y_pred.ndim == 3, "y_true/y_pred는 [B,H,C]여야 합니다."
    B, H, C = y_true.shape
    assert 0 <= sample_idx < B, "sample_idx 범위 오류"
    if channels is None:
        channels = list(range(C))

    _ensure_dir(outpath)

    t = np.arange(H)
    plt.figure(figsize=(12, 4))
    for c in channels:
        plt.plot(t, y_true[sample_idx, :, c], label=f"True-Ch{c}")
        plt.plot(t, y_pred[sample_idx, :, c], "--", label=f"Pred-Ch{c}")
    plt.xlabel("future step")
    plt.ylabel("value")
    plt.title(title)
    plt.legend(ncol=min(4, len(channels) * 2))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[plot] saved {outpath}")


def plot_batch_examples(
    X_np: np.ndarray,
    y_np: np.ndarray,
    yhat: Optional[np.ndarray] = None,
    n_examples: int = 3,
    outdir: str = "plots",
    tag: str = "",
):
    """
    입력/정답/예측을 채널별로 나눠 여러 개 PNG로 저장합니다.
    - X_np: [B, T, C]
    - y_np: [B, H, C]
    - yhat: [B, H, C] (옵션)
    """
    os.makedirs(outdir, exist_ok=True)
    B, T, C = X_np.shape
    H = y_np.shape[1]
    t_input = np.arange(T)
    t_future = np.arange(T, T + H)

    for b in range(min(n_examples, B)):
        fig, axs = plt.subplots(C, 1, figsize=(12, 2.5 * C), sharex=True)
        if C == 1:
            axs = [axs]
        ttl = f"Sample {b}" + (f" — {tag}" if tag else "")
        fig.suptitle(ttl, fontsize=14)

        for c in range(C):
            axs[c].plot(t_input, X_np[b, :, c], label="input", color="black")
            axs[c].plot(t_future, y_np[b, :, c], label="true future", color="green")
            if yhat is not None:
                axs[c].plot(
                    t_future,
                    yhat[b, :, c],
                    label="pred",
                    color="red",
                    linestyle="--",
                )
            axs[c].legend(loc="upper right")
            axs[c].set_ylabel(f"Ch{c}")
        axs[-1].set_xlabel("time step")

        out_path = os.path.join(outdir, f"{'direct' if not tag else tag}_sample_{b}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[plot] saved {out_path}")


def umap_scatter_2d(
    emb: np.ndarray,
    labels: np.ndarray,
    outpath: str,
    title: str = "UMAP 2D",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
):
    """
    UMAP 2D 임베딩 PNG 저장.
    - emb: [N, d]
    - labels: [N]
    - random_state 제거(병렬 경고 방지, 속도 향상)
    """
    import umap  # pip install umap-learn

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        # random_state=None  # 명시 안 하면 None
    )
    Z = reducer.fit_transform(emb)  # [N,2]

    _ensure_dir(outpath)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=24, alpha=0.85, cmap="tab10")
    plt.colorbar(sc, label="cluster")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[umap] saved 2D plot -> {outpath}")


def umap_scatter_3d(
    emb: np.ndarray,
    labels: np.ndarray,
    outpath: str,
    title: str = "UMAP 3D",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
):
    """
    UMAP 3D 임베딩 PNG 저장.
    """
    import umap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=3,
        # random_state=None
    )
    Z = reducer.fit_transform(emb)  # [N,3]

    _ensure_dir(outpath)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=labels, s=18, alpha=0.85, cmap="tab10")
    fig.colorbar(p, ax=ax, shrink=0.6, label="cluster")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[umap] saved 3D plot -> {outpath}")
