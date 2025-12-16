#!/usr/bin/env python3
import sys
import subprocess
import traceback
from pathlib import Path

Q_TST_CONFIG = Path("./configs/tsm_R50_MMA_q-tst.py")

def ok(msg):
    print(f"[OK] {msg}")

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def check_imports():
    try:
        import torch
        ok(f"torch imported (version={torch.__version__})")
    except Exception:
        fail("torch import failed\n" + traceback.format_exc())

    try:
        import cv2
        ok(f"cv2 imported (version={cv2.__version__})")
    except Exception:
        fail("cv2 import failed\n" + traceback.format_exc())

    try:
        import mmaction
        ok(f"mmaction imported (version={getattr(mmaction, '__version__', 'unknown')})")
    except Exception:
        fail("mmaction import failed\n" + traceback.format_exc())


def check_cuda():
    import torch
    if torch.cuda.is_available():
        ok(f"CUDA available (device={torch.cuda.get_device_name(0)})")
    else:
        print("[WARN] CUDA not available (CPU-only mode)")


def run_dummy_train(cfg):
    # cfg = Path(Q_TST_CONFIG)
    data_dir = Path("data")

    if not cfg.exists():
        fail(f"Config not found: {cfg}")

    if not data_dir.exists():
        fail("Expected ./data directory does not exist")

    train_py = Path("extern/mmaction2/tools/train.py")
    if not train_py.exists():
        fail("MMAction2 train.py not found (extern/mmaction2/tools/train.py)")

    cmd = [
        sys.executable,
        str(train_py),
        str(cfg),
        "--work-dir", "work_dirs/_env_check",
        "--cfg-options",
        "train_cfg.max_epochs=1",
        "train_dataloader.batch_size=1",
        "val_cfg=None",
        "val_dataloader=None",
        "val_evaluator=None",
    ]

    print("[INFO] Running dummy training command:")
    print("       " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        ok("Dummy training run completed successfully")
    except subprocess.CalledProcessError as e:
        fail(f"Dummy training failed with exit code {e.returncode}")


def main():
    print("=== weSmart environment verification ===")

    ok(f"Python executable: {sys.executable}")
    ok(f"Python version: {sys.version.split()[0]}")

    check_imports()
    check_cuda()
    run_dummy_train(Q_TST_CONFIG)

    print("\n=== ALL CHECKS PASSED ===")
    print("Environment is valid and ready for offline use.")


if __name__ == "__main__":
    main()
