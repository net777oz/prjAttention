# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║ FILE: utils.py                                                            ║
╠───────────────────────────────────────────────────────────────────────────╣
║ PURPOSE  공통 유틸: 시간 포맷, rich 콘솔(표/패널) 빌더                     ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
import numpy as np

USE_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    USE_RICH = True
    console = Console()
except Exception:
    console = None
    Table = Panel = None  # type: ignore


def fmt_time(secs):
    if secs is None or (isinstance(secs, float) and (np.isnan(secs) or secs < 0)):
        return "-"
    m, s = divmod(int(secs), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_train_table(ep, epochs, bi, num_batches, done, total,
                      loss_cur, loss_avg, lr, step_time, eta_sec, mem_mb, bs, amp_on, compiled):
    if not USE_RICH:
        return None
    from rich.table import Table
    from rich.panel import Panel

    pct = 0.0 if not total else (100.0 * done / total)
    t = Table(expand=True, show_header=False, pad_edge=False, box=None)

    left = Table(show_header=False, pad_edge=False, box=None)
    left.add_row("Epoch", f"{ep}/{epochs}")
    left.add_row("Batch", f"{bi}/{num_batches}")
    left.add_row("Progress", f"{done}/{total} ({pct:5.1f}%)")
    left.add_row("ETA", fmt_time(eta_sec if total else 0.0))

    mid = Table(show_header=False, pad_edge=False, box=None)
    mid.add_row("Loss(cur)", f"{loss_cur:.6f}" if isinstance(loss_cur, (float, int)) and np.isfinite(loss_cur) else "nan")
    mid.add_row("Loss(avg)", f"{loss_avg:.6f}" if isinstance(loss_avg, (float, int)) and np.isfinite(loss_avg) else "nan")
    mid.add_row("LR", f"{lr:.2e}")
    mid.add_row("Step", f"{(step_time if step_time else 0.0):.2f}s")

    right = Table(show_header=False, pad_edge=False, box=None)
    right.add_row("BatchSize", str(bs))
    right.add_row("AMP", "ON" if amp_on else "OFF")
    right.add_row("Compile", compiled or "OFF")
    right.add_row("GPU Mem", f"{int(mem_mb)}MB")

    t.add_row(left, mid, right)
    return Panel(t, title="train", border_style="cyan")


def build_eval_table(phase: str, i: int, N: int, metric_name: str, metric_val: float, eta_sec: float):
    if not USE_RICH:
        return None
    from rich.table import Table
    from rich.panel import Panel
    t = Table(expand=True, show_header=False, pad_edge=False, box=None)
    t.add_row("Phase", phase)
    t.add_row("Row", f"{i}/{N}")
    t.add_row(metric_name, f"{metric_val:.6f}" if isinstance(metric_val, (float, int)) and np.isfinite(metric_val) else "nan")
    t.add_row("ETA", fmt_time(eta_sec))
    return Panel(t, title="eval", border_style="magenta")

