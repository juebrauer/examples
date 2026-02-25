"""Simple LR sweep + headless online eval for CNN / ResNet18 / ViT.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# -----------------
# Configuration
# -----------------

PYTHON = sys.executable

DATASET_DIR = Path("./dataset1000")
EPOCHS = 100
ONLINE_EPISODES = 500
DEVICE = "auto"  # auto|cpu|cuda|mps

# Environment params for online evaluation
W, H, P = 16, 12, 0.22
CELL = 32
SEED = 0

# Training speed knobs
AMP = "auto"           # auto|on|off
NUM_WORKERS = 4
PIN_MEMORY = "auto"    # auto|on|off

SCHED = "cosine"       # none|plateau|cosine
WEIGHT_DECAY = 5e-2

# Learning-rate sweeps (small / medium / large)
ARCHS = ["cnn", "resnet18", "vit"]

# Same learning rates for all architectures (better comparability).
# Adjust these three values if you want a different sweep.
LRS: list[tuple[str, str]] = [
    ("small", "1e-5"),
    ("medium", "1e-4"),
    ("large", "1e-3"),
    ("huge", "1e-2"),
]


def _log(line: str, log_f) -> None:
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()


def run_cmd(cmd: list[str], env: dict[str, str], log_f) -> None:
    _log("", log_f)
    _log("$ " + " ".join(cmd), log_f)

    # Stream stdout+stderr to both console and logfile.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for out_line in proc.stdout:
        # out_line already includes newline
        print(out_line, end="", flush=True)
        log_f.write(out_line)
    rc = proc.wait()
    log_f.flush()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("./runs") / f"lr_sweep_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    log_path = run_root / "experiments.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        _log(f"Run root: {run_root.resolve()}", log_f)
        _log(f"Log file: {log_path.resolve()}", log_f)
        _log(f"Dataset : {DATASET_DIR}", log_f)
        _log(f"Epochs  : {EPOCHS}", log_f)
        _log(f"Online episodes: {ONLINE_EPISODES}", log_f)
        _log(f"ARCHS   : {ARCHS}", log_f)
        _log(f"LRS     : {LRS}", log_f)
        _log(f"DEVICE  : {DEVICE}", log_f)
        _log(f"AMP={AMP} NUM_WORKERS={NUM_WORKERS} PIN_MEMORY={PIN_MEMORY}", log_f)
        _log(f"SCHED={SCHED} WEIGHT_DECAY={WEIGHT_DECAY}", log_f)

        # Always run Qt in offscreen mode (server-friendly).
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"

        for arch in ARCHS:
            for lr_label, lr_value in LRS:
                run_dir = run_root / f"{arch}__{lr_label}__lr{lr_value}"
                run_dir.mkdir(parents=True, exist_ok=True)

                _log(f"\n=== TRAIN arch={arch} lr={lr_value} ({lr_label}) ===", log_f)
                run_cmd(
                    [
                        PYTHON,
                        "robot_navigation.py",
                        "train",
                        "--dataset",
                        str(DATASET_DIR),
                        "--run-dir",
                        str(run_dir),
                        "--arch",
                        arch,
                        "--epochs",
                        str(EPOCHS),
                        "--device",
                        DEVICE,
                        "--lr",
                        str(lr_value),
                        "--weight-decay",
                        str(WEIGHT_DECAY),
                        "--sched",
                        SCHED,
                        "--amp",
                        AMP,
                        "--num-workers",
                        str(NUM_WORKERS),
                        "--pin-memory",
                        PIN_MEMORY,
                    ],
                    env=env,
                    log_f=log_f,
                )

                model_path = run_dir / f"{arch}.pt"
                if not model_path.exists():
                    raise FileNotFoundError(f"Expected model checkpoint not found: {model_path}")

                _log(f"\n=== TEST-ONLINE (HEADLESS) arch={arch} lr={lr_value} ({lr_label}) ===", log_f)
                run_cmd(
                    [
                        PYTHON,
                        "robot_navigation.py",
                        "test-online",
                        "--model",
                        str(model_path),
                        "--arch",
                        arch,
                        "--episodes",
                        str(ONLINE_EPISODES),
                        "--w",
                        str(W),
                        "--h",
                        str(H),
                        "--p",
                        str(P),
                        "--cell",
                        str(CELL),
                        "--seed",
                        str(SEED),
                        "--device",
                        DEVICE,
                        "--headless",
                    ],
                    env=env,
                    log_f=log_f,
                )

        _log("\nAll runs complete. Results appended to overall_results.md (and overall_results.txt).", log_f)


if __name__ == "__main__":
    main()
