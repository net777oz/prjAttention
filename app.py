# app.py — prjAttention Streamlit (terminal logging, no in-app live log)
import os
import sys
import json
import time
import signal
import platform
import subprocess
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd  # for preview CSVs

APP_TITLE = "prjAttention – Streamlit (Terminal Logs)"
ARTIFACTS_DIR = Path("./artifacts")

# -------------------- small helpers --------------------
def brace_expand(pattern: str) -> List[str]:
    """Minimal {a..b} expansion with zero-padding. Space-separated tokens."""
    parts: List[str] = []
    for token in pattern.split():
        if "{" in token and "}" in token and ".." in token:
            pre = token[: token.index("{")]
            mid = token[token.index("{") + 1 : token.index("}")]
            post = token[token.index("}") + 1 :]
            a, b = mid.split("..")
            if a.isdigit() and b.isdigit():
                width = max(len(a), len(b))
                sa, sb = int(a), int(b)
                step = 1 if sb >= sa else -1
                for i in range(sa, sb + step, step):
                    parts.append(f"{pre}{str(i).zfill(width)}{post}")
            else:
                parts.append(token)
        else:
            parts.append(token)
    return parts


def make_cmd(mode: str, task: str, csvs: List[str], context_len: int, opts: dict) -> List[str]:
    # run_llmts.py 미사용 → cli.py 직접 호출
    cmd = ["python", "-u", "cli.py", "--mode", mode, "--task", task, "--csv"] + csvs
    cmd += ["--context-len", str(context_len)]

    # 공통 옵션
    if opts.get("label_src_ch", 0) != 0:
        cmd += ["--label-src-ch", str(opts["label_src_ch"])]
    if opts.get("drop_label_from_x", False):
        cmd += ["--drop-label-from-x"]
    if opts.get("label_offset", 1) != 1:
        cmd += ["--label-offset", str(opts["label_offset"])]

    # 스플릿/지표
    cmd += ["--split-mode", opts["split_mode"], "--val-ratio", str(opts["val_ratio"])]
    cmd += ["--eval-split", opts["eval_split"], "--plot-split", opts["plot_split"]]

    if task == "classify":
        cmd += ["--bin-rule", opts["bin_rule"]]
        if opts["bin_rule"] != "nonzero":
            cmd += ["--bin-thr", str(opts["bin_thr"])]
        cmd += ["--thresh-default", str(opts["thresh_default"])]
        if opts.get("pos_weight"):
            cmd += ["--pos-weight", str(opts["pos_weight"])]
        if opts.get("alpha") is not None:
            cmd += ["--alpha", str(opts["alpha"])]

    # 학습/파인튜닝
    if mode in ("train", "finetune"):
        cmd += ["--epochs", str(opts["epochs"]), "--batch-size", str(opts["batch_size"])]
        if opts.get("lr"):
            cmd += ["--lr", str(opts["lr"])]

    # 체크포인트
    if mode in ("finetune", "infer"):
        ckpt = opts.get("ckpt", "").strip()
        if ckpt:
            cmd += ["--ckpt", ckpt]

    # 런타임
    if opts.get("amp", True):
        cmd += ["--amp"]
    if opts.get("compile", "reduce-overhead") and opts["compile"] != "off":
        cmd += ["--compile", opts["compile"]]

    # 백본: llm_ts / lstm
    if opts.get("backbone", "llm_ts"):
        cmd += ["--backbone", opts["backbone"]]

    # outdir: 입력됐을 때만 전달 (비면 파이프라인이 자동 생성)
    if opts.get("outdir"):
        cmd += ["--outdir", opts["outdir"]]

    if opts.get("seed") is not None:
        cmd += ["--seed", str(opts["seed"])]
    if opts.get("desc"):
        cmd += ["--desc", opts["desc"]]

    return cmd


def ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(exist_ok=True)


def fmt_elapsed(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# -------------------- Streamlit state --------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

def _init_ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init_ss("proc", None)
_init_ss("status", "IDLE")       # IDLE/RUNNING/STOPPING/FINISHED/ERROR
_init_ss("stop_requested", False)
_init_ss("start_time", None)
_init_ss("end_time", None)
_init_ss("_last_refresh", 0.0)
_init_ss("events", [])

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Run Configuration")
    mode = st.radio(
        "Mode",
        ["train", "finetune", "infer"],
        index=0,
        horizontal=True,
        help="학습/파인튜닝/추론 모드를 선택합니다.",
    )
    task = st.radio(
        "Task",
        ["classify", "regress"],
        index=1,
        horizontal=True,
        help="분류(classify) 또는 회귀(regress)를 선택합니다.",
    )

    st.markdown("---")
    st.subheader("Data")
    csv_pattern = st.text_input(
        "CSV paths",
        "./data/LP0a.csv" if task == "classify" else "./data/x_main.csv",
        help="여러 개 가능(공백 구분). 브레이스 확장 지원 예: ./data/LP{1..31}a.csv 또는 ./data/LP{001..031}a.csv",
    )
    exp = brace_expand(csv_pattern)
    st.caption(f"Expanded CSVs: {len(exp)} file(s)")
    with st.expander("Show expanded list"):
        st.text("\n".join(exp) if exp else "(none)")

    context_len = st.number_input(
        "context-len",
        min_value=1,
        value=31,
        step=1,
        help="윈도우 길이 L (t-L..t-1 → t 예측). 데이터 T보다 작아야 합니다.",
    )
    label_src_ch = st.number_input(
        "label-src-ch",
        min_value=0,
        value=0,
        step=1,
        help="라벨(정답)을 가져올 채널 인덱스. 분류/회귀 모두 ch0 타깃으로 평가합니다.",
    )
    drop_label_from_x = st.checkbox(
        "drop-label-from-x",
        value=False,
        help="체크하면 라벨 소스 채널을 입력 특징에서 제거합니다(누수 방지용).",
    )
    label_offset = st.number_input(
        "label-offset",
        min_value=1,
        value=1,
        step=1,
        help="t 시점의 라벨을 얼마나 미래로 당길지. 일반적으로 1(다음 스텝 예측).",
    )

    st.markdown("---")
    st.subheader("Task Options")
    if task == "classify":
        bin_rule = st.selectbox(
            "bin-rule",
            ["nonzero", "gt", "lt", "quantile"],
            index=0,
            help="연속 라벨을 이진화하는 규칙. nonzero | 임계(gt/lt) | 분위수(quantile).",
        )
        bin_thr = st.number_input(
            "bin-thr",
            value=0.0,
            step=0.05,
            format="%.6f",
            help="bin-rule이 gt/lt일 때 사용하는 기준값(임계치).",
        )
        thresh_default = st.number_input(
            "thresh-default",
            value=0.5,
            step=0.01,
            format="%.3f",
            help="분류 확률을 양성으로 판단하는 기본 임계값.",
        )
    else:
        bin_rule, bin_thr, thresh_default = "nonzero", 0.0, 0.5

    split_mode = st.selectbox(
        "split-mode",
        ["group", "rolling", "holdout"],
        index=0,
        help="학습/검증 분할 전략: 그룹 단위, 롤링 윈도우, 단순 홀드아웃.",
    )
    val_ratio = st.slider(
        "val-ratio",
        min_value=0.0,
        max_value=0.9,
        value=0.2,
        step=0.05,
        help="검증 비율(holdout/rolling에서 사용). group은 내부 규칙 적용.",
    )
    eval_split = st.selectbox(
        "eval-split",
        ["train", "val", "test"],
        index=1,
        help="평가 대상 스플릿(로그/플롯용 태그).",
    )
    plot_split = st.selectbox(
        "plot-split",
        ["train", "val", "test"],
        index=1,
        help="그림 저장 시 스플릿 태그.",
    )

    st.markdown("---")
    st.subheader("Training")
    epochs = st.number_input(
        "epochs",
        min_value=1,
        value=(30 if mode == "train" else 10),
        step=1,
        help="총 학습 에폭 수. 0이면 학습 스킵.",
    )
    batch_size = st.number_input(
        "batch-size",
        min_value=1,
        value=4096,
        step=1,
        help="학습 배치 크기.",
    )
    lr = st.text_input(
        "lr (optional)",
        "",
        help="러닝레이트. 비우면 모델 기본값 사용(옵티마이저 기본).",
    )
    alpha = st.number_input(
        "alpha (classification only)",
        value=0.5,
        step=0.05,
        format="%.3f",
        help="분류에서 BCE와 SoftF1 가중치: loss = α·BCE + (1-α)·SoftF1",
    ) if task == "classify" else None
    pos_weight = st.text_input(
        "pos-weight",
        "global" if task == "classify" else "",
        help="분류에서 양성 가중치 모드(global/batch/none) 또는 수치 입력.",
    )
    seed = st.number_input(
        "seed (optional)",
        min_value=0,
        value=777,
        step=1,
        help="난수 시드. 고정하면 재현성 향상.",
    )

    st.markdown("---")
    st.subheader("Runtime")
    amp = st.checkbox(
        "amp",
        value=True,
        help="CUDA/MPS에서 자동 혼합 정밀도(AMP) 사용.",
    )
    compile_mode = st.selectbox(
        "compile",
        ["off", "reduce-overhead"],
        index=1,
        help="PyTorch torch.compile 모드. off면 미사용.",
    )

    # 백본 선택: llm_ts / lstm
    backbone = st.selectbox(
        "backbone",
        ["llm_ts", "lstm"],
        index=0,
        help="모델 백본 선택. llm_ts(기본 Transformer) 또는 lstm(경량).",
    )

    ckpt = st.text_input(
        "ckpt (for finetune/infer)",
        "",
        help="파인튜닝/추론에 사용할 체크포인트 경로(model.pt). 비우면 미사용.",
    )

    st.markdown("---")
    st.subheader("Output")
    # outdir는 비워두면 자동 생성
    outdir = st.text_input(
        "outdir (artifacts/<name>)",
        value="",
        help="비우면 자동 폴더 생성. 입력 시 artifacts/<name>으로 고정 저장.",
    )
    desc = st.text_input(
        "desc (tag for run folder)",
        "",
        help="런 폴더명에 태그로 반영되는 간단한 설명.",
    )

# pack opts
opts = dict(
    label_src_ch=int(label_src_ch),
    drop_label_from_x=bool(drop_label_from_x),
    label_offset=int(label_offset),
    bin_rule=bin_rule,
    bin_thr=float(bin_thr),
    thresh_default=float(thresh_default),
    split_mode=split_mode,
    val_ratio=float(val_ratio),
    eval_split=eval_split,
    plot_split=plot_split,
    epochs=int(epochs),
    batch_size=int(batch_size),
    lr=lr.strip(),
    alpha=alpha if alpha is not None else None,
    pos_weight=pos_weight.strip(),
    seed=int(seed) if seed is not None else None,
    amp=bool(amp),
    compile=compile_mode,
    backbone=backbone,
    ckpt=ckpt.strip(),
    outdir=outdir.strip(),
    desc=desc.strip(),
)

# composed command
try:
    cmd_list = make_cmd(mode, task, exp, int(context_len), opts)
    st.markdown("**Composed Command**")
    st.code(" ".join([c if " " not in c else f'\"{c}\"' for c in cmd_list]), language="bash")
except Exception as e:
    st.error(f"Command build error: {e}")
    cmd_list = []

# -------------------- launch/stop --------------------
def _launch(cmd: List[str]):
    st.session_state.events.append(f"Launching: {' '.join(cmd)}")
    st.session_state.status = "RUNNING"
    st.session_state.stop_requested = False
    st.session_state.start_time = time.time()
    st.session_state.end_time = None

    # 중요: stdout/stderr를 캡처하지 않음(None) → 터미널로 그대로 흘러감
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="UTF-8")

    if platform.system() == "Windows":
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        proc = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            stdin=None,
            env=env,
            creationflags=creation,
        )
    else:
        # 새 세션으로 띄워 신호 관리만 분리(출력 FD는 부모 터미널 그대로 상속)
        proc = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            stdin=None,
            env=env,
            start_new_session=True,
        )
    st.session_state.proc = proc


def _force_stop():
    st.session_state.events.append("Stop requested by user.")
    st.session_state.stop_requested = True
    proc = st.session_state.proc
    if not proc:
        return
    try:
        if platform.system() == "Windows":
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
            time.sleep(0.5)
            if proc.poll() is None:
                proc.terminate()
            time.sleep(0.5)
            if proc.poll() is None:
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            time.sleep(0.7)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
    finally:
        st.session_state.status = "STOPPING"

# controls
ensure_artifacts_dir()
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
is_running = bool(st.session_state.proc) and st.session_state.status in ("RUNNING", "STOPPING")
toggle_label = "⏹ Stop" if is_running else "▶ Run"
if c1.button(toggle_label, type=("primary" if not is_running else "secondary")):
    if is_running:
        _force_stop()
    else:
        if cmd_list:
            try:
                _launch(cmd_list)
                st.success("실시간 로그는 이 페이지가 아니라, **Streamlit를 실행한 터미널**에 출력됩니다.")
            except Exception as e:
                st.session_state.status = "ERROR"
                st.error(f"Failed to start process: {e}")

c2.write("Artifacts:")
c2.caption(str(ARTIFACTS_DIR.resolve()))
c3.link_button("Open artifacts", ARTIFACTS_DIR.resolve().as_uri())

# status
d0, d1, d2, d3 = st.columns(4)
d0.metric("Status", st.session_state.status)
d1.metric(
    "Elapsed",
    fmt_elapsed((time.time() - st.session_state.start_time) if st.session_state.start_time else 0),
)
d2.metric("PID", str(st.session_state.proc.pid) if st.session_state.proc else "-")
d3.metric("Stop Requested", "Yes" if st.session_state.stop_requested else "No")

# detect end
proc = st.session_state.proc
if proc and (code := proc.poll()) is not None:
    st.session_state.proc = None
    st.session_state.end_time = time.time()
    st.session_state.status = "FINISHED" if code == 0 else "ERROR"
    st.session_state.events.append(f"Process exited with code {code}.")

# events
st.markdown("### App Events")
st.text("\n".join(st.session_state.events[-100:]) if st.session_state.events else "(no events)")

# artifacts preview (simple)
st.markdown("### Recent Artifacts")
if ARTIFACTS_DIR.exists():
    runs = sorted(
        [p for p in ARTIFACTS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:6]
    if not runs:
        st.write("(no runs yet)")
    for r in runs:
        with st.expander(r.name, expanded=False):
            rp = r.resolve()
            st.write(f"Path: `{rp}`")
            st.markdown(f"[Open this folder]({rp.as_uri()})")
            # 분류/회귀 공용 JSON 미리보기
            for f in ["metrics.json", "summary.json"]:
                jf = r / f
                if jf.exists():
                    try:
                        st.json(json.load(open(jf, "r", encoding="utf-8")))
                    except Exception as e:
                        st.warning(f"Failed to read {f}: {e}")
            # 회귀 after_pred.csv 미리보기 (있으면 표시)
            ap = r / "after_pred.csv"
            if ap.exists():
                try:
                    st.caption("after_pred.csv (head)")
                    st.dataframe(pd.read_csv(ap).head(1000), use_container_width=True, height=280)
                except Exception as e:
                    st.warning(f"Failed to read after_pred.csv: {e}")
else:
    st.write("(artifacts dir not found)")

# auto refresh every 1s while running/ stopping
auto_refresh = True
if (st.session_state.status in ("RUNNING", "STOPPING")) and auto_refresh:
    now = time.time()
    if now - st.session_state._last_refresh >= 1.0:
        st.session_state._last_refresh = now
        st.rerun()
