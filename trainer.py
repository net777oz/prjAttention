# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: trainer.py                                                          ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  학습 루프(AMP/torch.compile 대응), 분류 하이브리드 손실 적용       ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils import USE_RICH, console, build_train_table, fmt_time
from losses import SoftF1Loss, build_bce, compute_pos_weight_from_labels

from torch import amp as torch_amp


def train_all_epochs(
    model,
    dl,
    opt,
    scaler: torch_amp.GradScaler,
    epochs: int,
    amp_enabled: bool = True,
    log_every: int = 50,
    compiled: str | None = None,
    task: str = "regress",
    alpha: float = 0.5,
    pos_weight_mode: str = "global",
    global_pos_weight: Optional[float] = None,
) -> List[float]:

    dev = next(model.parameters()).device
    total_steps = epochs * len(dl)
    step_counter = 0
    last = time.time()
    avg_step_time = None

    # epochs==0: 학습 스킵
    if epochs <= 0:
        if USE_RICH:
            print("[INFO] epochs=0 → training skipped.", flush=True)
        return []

    softf1 = SoftF1Loss() if task == "classify" else None
    bce_loss_fn, _pw = (build_bce(pos_weight_mode, dev, global_pos_weight)
                        if task == "classify" else (None, None))

    # Live 패널 초기 렌더
    live = None
    if USE_RICH:
        from rich.live import Live
        bs0 = getattr(dl, "batch_size", 0)
        mem0 = (torch.cuda.memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        panel0 = build_train_table(
            0, epochs, 0, len(dl),
            0, total_steps,
            np.nan, np.nan,
            0.0, 0.0, 0.0,
            mem0, bs0,
            amp_enabled,
            compiled
        )
        live = Live(panel0, refresh_per_second=10, console=console)
        live.start()

    def update_table(ep, bi, loss_cur, loss_avg, bs):
        nonlocal avg_step_time, last, step_counter
        now = time.time()
        dt = now - last
        last = now
        avg_step_time = dt if avg_step_time is None else 0.9 * avg_step_time + 0.1 * dt
        done = step_counter
        remaining = (total_steps - done) * (avg_step_time if (avg_step_time and total_steps > 0) else 0.0)
        lr = opt.param_groups[0].get("lr", 0.0)
        mem = (torch.cuda.memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

        if USE_RICH and live is not None:
            panel = build_train_table(
                ep, epochs, bi, len(dl),
                done, total_steps,
                float(loss_cur) if loss_cur is not None else np.nan,
                float(loss_avg) if loss_avg is not None else np.nan,
                lr,
                avg_step_time if avg_step_time else 0.0,
                remaining,
                mem, bs,
                amp_enabled,
                compiled
            )
            live.update(panel)
        else:
            print(
                f"\r[train] ep {ep}/{epochs} batch {bi}/{len(dl)} "
                f"done {done}/{total_steps} loss={loss_cur:.6f} avg={loss_avg:.6f} "
                f"lr={lr:.2e} step={avg_step_time:.2f}s ETA={fmt_time(remaining)} "
                f"mem={int(mem)}MB",
                end="",
                flush=True,
            )

    epoch_loss_hist: List[float] = []

    try:
        for ep in range(1, epochs + 1):
            model.train()
            total, steps = 0.0, 0

            for bi, (xb, yb) in enumerate(dl):
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                if amp_enabled:
                    with torch_amp.autocast(device_type="cuda", enabled=True):
                        logits, _ = model(xb)
                        if task == "regress":
                            loss = F.mse_loss(logits, yb)
                        else:
                            if pos_weight_mode == "batch":
                                pw = compute_pos_weight_from_labels(yb)
                                bce_loss_fn.pos_weight = torch.tensor([pw], dtype=torch.float32, device=dev)
                            bce = bce_loss_fn(logits, yb)  # type: ignore[arg-type]
                            s1f = softf1(logits, yb)       # type: ignore[operator]
                            loss = alpha * bce + (1 - alpha) * s1f

                    if not torch.isfinite(loss):
                        step_counter += 1
                        update_table(ep, bi, float("nan"), float("nan"), xb.size(0))
                        continue

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                else:
                    logits, _ = model(xb)
                    if task == "regress":
                        loss = F.mse_loss(logits, yb)
                    else:
                        if pos_weight_mode == "batch":
                            pw = compute_pos_weight_from_labels(yb)
                            bce_loss_fn.pos_weight = torch.tensor([pw], dtype=torch.float32, device=dev)
                        bce = bce_loss_fn(logits, yb)      # type: ignore[arg-type]
                        s1f = softf1(logits, yb)           # type: ignore[operator]
                        loss = alpha * bce + (1 - alpha) * s1f

                    if not torch.isfinite(loss):
                        step_counter += 1
                        update_table(ep, bi, float("nan"), float("nan"), xb.size(0))
                        continue

                    loss.backward()
                    opt.step()

                total += loss.item()
                steps += 1
                step_counter += 1
                avg = total / steps

                if (bi % max(1, log_every)) == 0:
                    update_table(ep, bi, loss.item(), avg, xb.size(0))

            ep_avg = total / max(1, steps)
            epoch_loss_hist.append(ep_avg)
            print(f"\n[EPOCH {ep}/{epochs}] avg_loss={ep_avg:.6f}", flush=True)

    finally:
        if USE_RICH and live is not None:
            live.stop()
        else:
            print()

    return epoch_loss_hist
