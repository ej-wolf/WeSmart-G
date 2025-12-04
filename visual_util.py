import re
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from my_local_utils import _load_log_lines

log_path ="./work_dirs/R50_bbrfm_01/20251202_042950/20251202_042950.log"
# log_path = "20251202_042950.log"


def play_frames(frames_dir, log=None, x: float = 1.0, event_color=False):
    """Play frames as a cartoon using timestamps in filenames.

    Filenames must be frm_<frame>_<time_ms>.jpg.
    event_color controls coloring of boxes/centers during events:
      - False: no color change
      - True: use red (255,0,0)
      - (r,g,b): custom RGB tuple/list (0-255 or 0-1 floats)
    """

    frames_dir = Path(frames_dir)
    if x <= 0:
        x = 1.0

    # Collect frame files and parse their timestamps
    frame_files = []
    for p in sorted(frames_dir.glob("frm_*_*.jpg")):
        stem = p.stem  # frm_<frame>_<time_ms>
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            frame_num = int(parts[1])
            time_ms = int(parts[2])
        except ValueError:
            continue
        frame_files.append((time_ms, frame_num, p))

    if not frame_files:
        print(f"[INFO] No frames found in {frames_dir}")
        return

    # Sort by original time
    frame_files.sort(key=lambda t: t[0])

    # Load events log (if any)
    events_by_time: dict[int, str] = {}

    if log is not None:
        log_lines = _load_log_lines(log)
    else:
        candidates = list(frames_dir.glob("*_events.log"))
        log_lines = _load_log_lines(candidates[0]) if candidates else []

    for line in log_lines:
        parts = line.split(",", 2)
        if len(parts) < 3:
            continue
        try:
            t_ms = int(parts[0].strip())
        except ValueError:
            continue
        events_str = parts[2].strip()
        events_by_time[t_ms] = events_str

    # Prepare event color in BGR
    color_bgr = None
    if isinstance(event_color, bool):
        if event_color:
            color_bgr = (0, 0, 255)  # red in BGR
    else:
        try:
            r, g, b = event_color
            vals = []
            for v in (r, g, b):
                if isinstance(v, float) and 0.0 <= v <= 1.0:
                    v = int(round(v * 255))
                v = int(max(0, min(255, int(v))))
                vals.append(v)
            r, g, b = vals
            color_bgr = (b, g, r)
        except Exception:
            color_bgr = None

    window_name = f"play_frames: {frames_dir.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time_ms = None
    for time_ms, frame_num, path in frame_files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if not  bool(log_lines):
            events_str =  "No relevant log found"
        # events_str = events_by_time.get(time_ms)
        else:
            events_str = events_by_time.get(time_ms)

        # Color boxes/centers if event and color are set
        if events_str and color_bgr is not None:
            mask = (img == 0)
            display[mask] = color_bgr

        # Overlay events text
        if events_str:
            org = (5, display.shape[0] - 10)
            cv2.putText(display, events_str, org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0),1, cv2.LINE_AA)
        cv2.imshow(window_name, display)

        if prev_time_ms is None:
            delay_ms = 1
        else:
            dt = max(1, time_ms - prev_time_ms)
            delay_ms = max(1, int(dt / x))
        prev_time_ms = time_ms

        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyWindow(window_name)


def plot_my_log(log_path, **kwargs):
    """     Parse an MMEngine/MMAction text log and plot:
    kwargs:
      plot_loss: if True plot train loss vs epoch (fig-1). Default: True
      plot_acc: polt accuracy - i.e. train top1 and val metric vs epoch (fig-2). Default: True
    """
    plot_loss = kwargs.get("plot_loss", True)
    plot_acc = kwargs.get("plot_acc", True)

    log_path = Path(log_path)
    if not log_path.is_file():
        print(f"[ERROR] Log file not found: {log_path}")
        return

    # Example lines we match:
    # Epoch(train)  [3][60/73]  ...  loss: 0.3629  top1_acc: 0.9375 ...
    # Epoch(val) [10][18/18]    acc/top1: 2.0000  acc/top5: 2.0000 ...
    train_pat = re.compile(
        r"Epoch\(train\)\s+\[(\d+)\]\[\d+/\d+\].*?"
        r"loss:\s+([0-9.]+)\s+top1_acc:\s+([0-9.]+)"
    )
    val_pat = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[\d+/\d+\].*?acc/top1:\s+([0-9.]+)"
    )

    # Aggregate per epoch (mean over all train iters)
    train_losses = {}  # epoch -> [losses]
    train_accs = {}    # epoch -> [top1_acc]
    val_acc = {}       # epoch -> scalar

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m_tr = train_pat.search(line)
            if m_tr:
                epoch = int(m_tr.group(1))
                loss = float(m_tr.group(2))
                acc = float(m_tr.group(3))
                train_losses.setdefault(epoch, []).append(loss)
                train_accs.setdefault(epoch, []).append(acc)
                continue

            m_val = val_pat.search(line)
            if m_val:
                epoch = int(m_val.group(1))
                acc = float(m_val.group(2))
                val_acc[epoch] = acc  # last val entry for that epoch wins

    if not train_losses:
        print("[WARN] No train entries found in log; nothing to plot.")
        return

    # Prepare epoch-wise averages
    epochs = sorted(train_losses.keys())
    mean_train_loss = [ sum(train_losses[e])/len(train_losses[e]) for e in epochs ]
    mean_train_acc =  [ sum(train_accs[e])/len(train_accs[e]) for e in epochs ]
    val_epochs = sorted(val_acc.keys())
    val_acc_vals = [val_acc[e] for e in val_epochs]

    # 1) Plot loss
    if plot_loss:
        plt.figure()
        plt.plot(epochs, mean_train_loss, marker="o", label="train loss")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.title("Training loss vs epoch")
        plt.grid(True, alpha=0.3)
        plt.legend(); plt.tight_layout()

    # 2) Plot accuracy-ish (train top1 and val metric)
    if plot_acc:
        plt.figure()
        plt.plot(epochs, mean_train_acc, marker="o", label="train top1")
        if val_epochs:
            plt.plot(val_epochs, val_acc_vals, marker="s", label="val metric (acc/top1)")
        plt.xlabel("epoch"); plt.ylabel("accuracy")
        plt.title("Train / val accuracy vs epoch")
        plt.grid(True, alpha=0.3)
        plt.legend(); plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_my_log(log_path)
