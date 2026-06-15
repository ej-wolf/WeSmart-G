""" visual_util
Plot, inspect, and debug visual results produced by the project pipelines.

Public API includes:
    play_frames(...)
    plot_my_log(...)
    plot_roc_curve(...)
    draw_confusion_matrix(...)
    plot_timeline(...)
    json_to_box_frames(...)
"""

import re, csv, json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, time as dt_time
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common.my_local_utils import _load_log_lines, _make_unique_dir, _save_log, print_color, get_unique_name

FRAME_H, FRAME_W = 1080, 1920
CENTER_SCALE = 0.12

def _close_all_on_key(event):
    if event.key in ('x', 'C'):
        plt.close('all')

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
                v = max(0, min(255, int(v)))
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


def json_to_box_frames(json_path, out_root, H: int = FRAME_H, W: int = FRAME_W, **kwargs):
    """Render one current stream JSON into monochrome box frames for debugging."""
    def _event_pairs(raw_events):
        pairs = []
        for ev in raw_events or []:
            if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                pairs.append((int(ev[0]), ev[1]))
            elif isinstance(ev, (int, float)) and not isinstance(ev, bool):
                pairs.append((int(ev), None))
        return pairs

    json_path, out_root = Path(json_path), Path(out_root)

    with json_path.open('r', encoding='utf-8') as fh:
        data = json.load(fh)

    frames = data.get('frames', [])
    title  = data.get('video' , '')
    if not frames:
        return None, {'Duration': 0.0, 'Frames': 0, 'Events': 0, 'T_evn': 0.0, 'stat': {}}

    if kwargs.get('clip_name'):
        clip_name = str(kwargs['clip_name'])
    elif kwargs.get('use_json_field', False) and isinstance(title, str) and title:
        clip_name = Path(title).stem
    else:
        clip_name = json_path.stem.replace('.', '_')

    clip_dir, clip_name = _make_unique_dir(out_root, clip_name, no_space=True)
    event_lines, event_flags, times_s = [] , [], []
    stat = {}

    for idx, fr in enumerate(frames, start=1):
        img = np.full((H, W), 255, dtype=np.uint8)
        for det in fr.get('detection_list', []) or []:
            box = det.get('bbox', [])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(float, box)

            if kwargs.get('y_from_bottom', False):
                y1, y2 = 1.0 - y1, 1.0 - y2

            px1 = max(0, min(W - 1, int(x1 * W)))
            px2 = max(0, min(W - 1, int(x2 * W)))
            py1 = max(0, min(H - 1, int(y1 * H)))
            py2 = max(0, min(H - 1, int(y2 * H)))
            if px2 <= px1 or py2 <= py1:
                continue

            cv2.rectangle(img, (px1, py1), (px2, py2), 0, thickness=1)
            if kwargs.get('draw_center', False):
                bb_w, bb_h = px2 - px1, py2 - py1
                if bb_w > 0 and bb_h > 0:
                    cx = px1 + bb_w // 2
                    cy = py1 + bb_h // 2
                    radius = max(1, int(CENTER_SCALE * min(bb_w, bb_h)))
                    cv2.circle(img, (cx, cy), radius, 0, thickness=-1)

        frame_num = fr.get('f', idx)
        t_sec = float(fr.get('t', 0.0))
        time_ms = int(t_sec * 1000)
        times_s.append(t_sec)

        pairs_ls = _event_pairs(fr.get('group_events') or fr.get('individual_events') or [])
        event_flags.append(bool(pairs_ls))
        for flag, _ in pairs_ls:
            stat[flag] = stat.get(flag, 0) + 1
        if pairs_ls:
            event_txt = '[' + '; '.join(f"{flag}:{'NA' if conf is None else f'{float(conf):.4f}'}"
                                        for flag, conf in pairs_ls) + ']'
            event_lines.append(f"{time_ms},{frame_num},{event_txt}")

        cv2.imwrite(str(clip_dir/f"frm_{frame_num:05d}_{time_ms:08d}.jpg"), img)

    if event_lines:
        _save_log(event_lines, clip_dir/f"{clip_name}_events.log")

    duration = max(times_s) if times_s else 0.0
    if len(times_s) > 1 and duration > 0:
        dt = (max(times_s) - min(times_s)) / (len(times_s) - 1)
        t_event = sum(event_flags) * dt
    else:
        t_event = 0.0

    events_count, prev = 0, False
    for cur in event_flags:
        if cur and not prev:
            events_count += 1
        prev = cur

    clip_info = {'Duration': duration, 'Frames': len(frames),
                 'Events': events_count, 'T_evn': t_event, 'stat': stat}
    return clip_name, clip_info
#103

def plot_my_log(log_path, **kwargs):
    """  Parse an MMEngine/MMAction text log and plot:
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
    train_pat = re.compile(r"Epoch\(train\)\s+\[(\d+)]\[\d+/\d+].*?loss:\s+([0-9.]+)\s+top1_acc:\s+([0-9.]+)")
    val_pat   = re.compile(r"Epoch\(val\)\s+\[(\d+)]\[\d+/\d+].*?acc/top1:\s+([0-9.]+)")

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

    plotted = False
    #* 1) Plot loss
    if plot_loss:
        fig = plt.figure()
        plt.plot(epochs, mean_train_loss, marker="o", label="train loss")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.title("Training loss vs epoch")
        plt.grid(True, alpha=0.3)
        plt.legend(); plt.tight_layout()
        plotted = True

    #* 2) Plot accuracy-ish (train top1 and val metric)
    if plot_acc:
        fig = plt.figure()
        plt.plot(epochs, mean_train_acc, marker="o", label="train top1")
        if val_epochs:
            plt.plot(val_epochs, val_acc_vals, marker="s", label="val metric (acc/top1)")
        plt.xlabel("epoch"); plt.ylabel("accuracy")
        plt.title("Train / val accuracy vs epoch")
        plt.grid(True, alpha=0.3)
        plt.legend(); plt.tight_layout()
        plotted = True

    if plotted:
        plt.show()

#****** ROC plotting *****
def plot_roc_curve(roc: dict, **kwargs):
    """Render ROC plot and optional CSV/PNG outputs from `roc_from_scores` data."""
    def _allow_save_print(policy) -> bool:
        policy = 'all' if policy is None else str(policy).strip().lower()
        return policy in {'all', 'save'}

    fig_size = kwargs.get('figsize', (6, 5))
    dpi = int(kwargs.get('dpi', 120))
    save_to = kwargs.get('save_to', None)
    save_csv = bool(kwargs.get('save_csv', True))
    print_policy = kwargs.get('print_policy', 'all')
    if 'show' in kwargs:
        show = bool(kwargs['show'])
    else:
        show = save_to is None
    if save_to is None and not show:
        return

    fpr = np.asarray(roc['fpr'], dtype=np.float64)
    tpr = np.asarray(roc['tpr'], dtype=np.float64)
    thresholds = np.asarray(roc['thresholds'], dtype=np.float64)
    auc = float(roc['auc'])

    title = kwargs.get('title', "ROC Curve")
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right')
    if title:
        ax.text(0.02, 0.98, str(title), transform=ax.transAxes,
                ha='left', va='top', fontsize=12)
    fig.tight_layout()

    if save_to is not None:
        save_to = Path(save_to)
        if save_to.suffix.lower() != '.png':
            save_to = save_to.with_suffix('.png')
        try:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to, dpi=dpi)
            if _allow_save_print(print_policy):
                print_color(f"  ROC plot saved to   :{save_to}", 'b')
            if save_csv:
                csv_path = save_to.with_suffix('.csv')
                roc_table = np.column_stack([fpr, tpr, thresholds])
                np.savetxt(csv_path, roc_table, delimiter=';', header='fpr;tpr;thresholds', comments='')
                if _allow_save_print(print_policy):
                    print_color(f"  ROC table saved to  :{csv_path}", 'b')
        except Exception as e:
            print_color(f"Failed to save ROC outputs to {save_to}: {e}", 'r')

    if show:
        plt.show()
    plt.close(fig)


def draw_confusion_matrix(cm, **kwargs):
    """Draw a 2x2 confusion matrix with optional rates and heat coloring.
    Expected layout is [[tn, fp], [fn, tp]].
    """
    def _allow_save_print(policy) -> bool:
        policy = 'all' if policy is None else str(policy).strip().lower()
        return policy in {'all', 'save'}

    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (2, 2):
        raise ValueError(f"confusion matrix must have shape (2, 2), got {cm.shape}")

    title = kwargs.get('title', None)
    label_mode = kwargs.get('label_mode', 'both')
    add_heat = bool(kwargs.get('add_heat', False))
    fig_size = kwargs.get('figsize', (5.5, 4.8))
    dpi = int(kwargs.get('dpi', 120))
    save_to = kwargs.get('save_to', None)
    show = bool(kwargs.get('show', save_to is None))

    if label_mode not in {'number', 'ratio', 'both'}:
        raise ValueError(f"label_mode must be one of 'number', 'ratio', 'both', got {label_mode!r}")
    if save_to is None and not show:
        return

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    neg_total = tn + fp
    pos_total = fn + tp
    rate_cm = np.asarray([
        [tn/neg_total if neg_total > 0 else 0.0, fp/neg_total if neg_total > 0 else 0.0],
        [fn/pos_total if pos_total > 0 else 0.0, tp/pos_total if pos_total > 0 else 0.0],
    ], dtype=np.float64)

    if add_heat:
        heat = rate_cm
        if np.max(heat) > 0:
            heat = heat/np.max(heat)
    else:
        heat = np.zeros_like(cm, dtype=np.float64)

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.imshow(heat, cmap='Greens', vmin=0.0, vmax=1.0)

    ax.set_xticks([0, 1], labels=['negative', 'positive'])
    ax.set_yticks([0, 1], labels=['negative', 'positive'])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    if title:
        ax.set_title(title)
    ax.text(0.5, 1.08, 'Prediction', transform=ax.transAxes, ha='center', va='bottom', fontsize=11)
    ax.text(-0.08, 1.02, 'GT', transform=ax.transAxes, ha='center', va='bottom', fontsize=11)

    cell_tags = np.asarray([['TN', 'FP'], ['FN', 'TP']])
    for i in range(2):
        for j in range(2):
            num_txt = str(int(cm[i, j]))
            rate_txt = f"{rate_cm[i, j]:.3f}"
            if label_mode == 'number':
                txt = f"{cell_tags[i, j]}\n{num_txt}"
            elif label_mode == 'ratio':
                txt = f"{cell_tags[i, j]}\n{rate_txt}"
            else:
                txt = f"{cell_tags[i, j]}\n{num_txt}\n({rate_txt})"
            ax.text(j, i, txt, ha='center', va='center', color='black')

    # Keep the matrix grid visually explicit even without heat coloring.
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.tight_layout()

    if save_to is not None:
        save_to = Path(save_to)
        if save_to.suffix.lower() != '.png':
            save_to = save_to.with_suffix('.png')
        try:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_to, dpi=dpi)
            if _allow_save_print(kwargs.get('print_policy', 'all')):
                print_color(f"  Confusion matrix saved to :{save_to}", 'b')
        except Exception as e:
            print_color(f"Failed to save confusion matrix to {save_to}: {e}", 'r')

    fig.canvas.mpl_connect('key_press_event', _close_all_on_key)
    if show:
        plt.show()
    # plt.close(fig)


def plot_timeline(csv_path, t_span=None, **kwargs):
    """Plot one stream timeline CSV produced by the stream analysis flow."""
    def _allow_save_print(policy) -> bool:
        policy = 'all' if policy is None else str(policy).strip().lower()
        return policy in {'all', 'save'}

    BASE_TICK = 60.0 #s
    def _time_to_seconds(value):
        """Convert numeric/string/datetime-like time values into seconds."""
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, datetime):
            return (value.hour * 3600.0 + value.minute * BASE_TICK +
                    value.second + value.microsecond / 1e6)
        if isinstance(value, dt_time):
            return (value.hour * 3600.0 + value.minute * BASE_TICK +
                    value.second + value.microsecond / 1e6)
        if isinstance(value, str):
            txt = value.strip()
            if ':' not in txt:
                return float(txt)
            parts = txt.split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                secs = float(parts[1])
                return mins*BASE_TICK + secs
            if len(parts) == 3:
                hrs = int(parts[0])
                mins = int(parts[1])
                secs = float(parts[2])
                return hrs * 3600.0 + mins*BASE_TICK + secs
        raise TypeError(f"unsupported time value: {value!r}")

    def _fmt_seconds_as_stamp(x_val, _pos=None):
        """Format elapsed seconds as mm:ss or hh:mm:ss."""
        sign = '-' if x_val < 0 else ''
        x_val = abs(float(x_val))
        hours = int(x_val // 3600)
        minutes = int((x_val % 3600) // 60)
        seconds = x_val % 60
        if hours > 0:
            return f"{sign}{hours:02d}:{minutes:02d}:{seconds:05.2f}".rstrip('0').rstrip('.')
        return f"{sign}{minutes:02d}:{seconds:05.2f}".rstrip('0').rstrip('.')

    def _merge_positive_spans(starts, ends, labels):
        """Merge overlapping positive windows into continuous shaded spans."""
        spans = []
        active = None
        for s, e, y in zip(starts, ends, labels):
            if float(y) <= 0:
                continue
            s = float(s)
            e = float(e)
            if active is None:
                active = [s, e]
            elif s <= active[1]:
                active[1] = max(active[1], e)
            else:
                spans.append(tuple(active))
                active = [s, e]
        if active is not None:
            spans.append(tuple(active))
        return spans

    def _minute_tick_step(span_sec: float) -> int:
        """ Pick a readable minute-based tick spacing for timestamp axes."""
        if span_sec <= 10*60:
            return 60
        if span_sec <= 20*60:
            return 2*60
        if span_sec <= 40*60:
            return 5*60
        return 10*60

    def _smooth_series(values, kernel_size: int):
        """Apply light 1D moving-average smoothing; kernel=1 leaves data unchanged."""
        if kernel_size <= 1 or len(values) == 0:
            return values
        kernel_size = int(kernel_size)
        kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
        return np.convolve(values, kernel, mode='same')

    def _default_title():
        """Infer a short title from CSV content/name, preferring the stream JSON name."""
        if rows and rows[0].get('video'):
            return Path(rows[0]['video']).stem

        base = csv_path.stem
        for suffix in ('_stream-timeline', '_timeline'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        for marker in ('_stream-tst_', '_raw_'):
            if marker in base:
                return base.split(marker, 1)[1]
        return base

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"timeline csv not found: {csv_path}")

    rows = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"timeline csv is empty: {csv_path}")

    label_key = 'gt_label' if 'gt_label' in rows[0] else 'y_true' if 'y_true' in rows[0] else None
    if 't_start' not in rows[0]:
        raise KeyError(f"timeline csv has no 't_start' column: {csv_path}")
    t_frm = np.asarray([float(row['t_frm']) for row in rows], dtype=np.float64)
    t_start = np.asarray([float(row['t_start']) for row in rows], dtype=np.float64)
    y_prob = np.asarray([float(row['y_prob']) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row['y_pred']) for row in rows], dtype=np.float64)
    gt_label = (np.asarray([float(row[label_key]) for row in rows], dtype=np.float64)
                if label_key is not None else None)

    plot_gt = bool(kwargs.get('plot_gt', gt_label is not None))
    fig_size = kwargs.get('figsize', (13.5, 4.8))
    dpi = int(kwargs.get('dpi', 120))
    title = kwargs.get('title', _default_title())
    x_format = kwargs.get('x_format', 'time_stamp')
    save_to = kwargs.get('save_to', None)
    print_policy = kwargs.get('print_policy', 'all')
    show = bool(kwargs.get('show', save_to is None))
    threshold = float(kwargs.get('threshold', 0.5))
    smooth = max(1, int(kwargs.get('smooth', 1)))

    if x_format not in {'seconds', 's', 'time_stamp'}:
        raise ValueError(f"x_format must be 'seconds', 's', or 'time_stamp', got {x_format!r}")
    if save_to is None and not show:
        return

    t_min = t_frm[0]
    t_max = t_frm[-1]
    if t_span is None:
        t1, t2 = t_min, t_max
    else:
        if not hasattr(t_span, '__iter__'):
            raise TypeError("t_span must be a 2-value iterable or None")
        t_span = list(t_span)
        if len(t_span) != 2:
            raise ValueError(f"t_span must have exactly 2 values, got {len(t_span)}")
        t1 = _time_to_seconds(t_span[0])
        t2 = _time_to_seconds(t_span[1])
        if t1 > t2:
            raise ValueError(f"t_span start must be <= end, got {t1} > {t2}")

    if t1 < t_min:
        print(f"[WARN] t_span start {t1:.3f}s is before timeline start {t_min:.3f}s; plotting available part only")
    if t2 > t_max:
        print(f"[WARN] t_span end {t2:.3f}s is after timeline end {t_max:.3f}s; plotting available part only")

    mask = (t_frm >= max(t1, t_min)) & (t_frm <= min(t2, t_max))
    if not np.any(mask):
        print(f"[WARN] Requested t_span [{t1:.3f}, {t2:.3f}] has no overlap with the timeline")
        mask = np.zeros_like(t_frm, dtype=bool)
    elif np.any(t_frm < t1):
        # Keep the previous point so step lines remain visually continuous at the left clip boundary.
        prev_idx = np.where(t_frm < t1)[0]
        if len(prev_idx) > 0:
            mask[prev_idx[-1]] = True

    plot_t = t_frm[mask]
    plot_t_start = t_start[mask]
    plot_prob = _smooth_series(y_prob[mask], smooth)
    plot_pred = y_pred[mask]
    plot_gt_vals = gt_label[mask] if gt_label is not None else None

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    gt_spans = _merge_positive_spans(plot_t_start, plot_t, plot_gt_vals) if (plot_gt and plot_gt_vals is not None) else []
    pred_spans = _merge_positive_spans(plot_t_start, plot_t, plot_pred)

    for s, e in gt_spans:
        ax.fill_between([s, e], [0.0, 0.0], [0.17, 0.17],
                        facecolor='#8BCB88', edgecolor='#63B56C',
                        alpha=0.35, linewidth=1.0, zorder=0)
    for s, e in pred_spans:
        ax.fill_between([s, e], [0.65, 0.65], [1.0, 1.0],
                        facecolor='#A9D3F0', edgecolor='#7FB9E5',
                        alpha=0.40, linewidth=1.0, zorder=1)

    ax.plot(plot_t, plot_prob, color='black', linewidth=1.9, label='raw probability', zorder=3)
    plot_max = np.max(plot_prob) if len(plot_t) > 0 else 0.0
    if plot_gt:
        if plot_gt_vals is None:
            raise KeyError(f"timeline csv has no GT label column: {csv_path}")
    ax.axhline(threshold, color='#E53935', linestyle='--', linewidth=1.4, zorder=2)
    plot_max = max(plot_max, threshold, 1.0)

    x_left, x_right = t1, t2
    if x_format == 'time_stamp' and t2 > t1:
        step_sec = _minute_tick_step(t2 - t1)
        x_left = int(np.ceil(t1 / step_sec) * step_sec)
        x_right = int(np.ceil(t2 / step_sec) * step_sec)
        if x_right <= x_left:
            x_right = x_left + step_sec
        x_ticks = np.arange(x_left, x_right + 0.1, step_sec, dtype=np.float64)
    elif t2 > t1:
        x_ticks = np.linspace(t1, t2, num=6)
    else:
        x_ticks = np.asarray([t1], dtype=np.float64)
    ax.set_xlim(x_left, x_right)
    ax.set_xticks(x_ticks)
    ax.set_xlabel('Time (sec)' if x_format in {'seconds', 's'} else 'Time')
    ax.set_ylabel('Probability')
    y_top = plot_max * 1.03 if plot_max > 0 else 1.0
    ax.set_ylim(0.0, y_top)
    ax.set_title(title)
    if x_format == 'time_stamp':
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_seconds_as_stamp))
    ax.grid(alpha=0.25, color='#B8B8B8', linewidth=0.8)

    legend_handles = [Patch(facecolor='#8BCB88', edgecolor='#63B56C', alpha=0.35, label='GT positive'),
                      Patch(facecolor='#A9D3F0', edgecolor='#7FB9E5', alpha=0.40, label='Predicted positive'),
                      Line2D([0], [0], color='black', linewidth=1.9, label='raw probability'),
                      Line2D([0], [0], color='#E53935', linestyle='--', linewidth=1.4, label=f'Threshold={threshold:.2f}'),
                      ]

    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9)

    if pred_spans:
        ax.text(t2 - 0.005 * max(t2 - t1, 1e-6), y_top * 0.90, 'Pred spans',
                color='#1E73BE', ha='right', va='center', fontsize=11)
    if gt_spans:
        ax.text(t2 - 0.005 * max(t2 - t1, 1e-6), y_top * 0.09, 'GT spans',
                color='#2E9B33', ha='right', va='center', fontsize=11)
    plt.tight_layout()

    if save_to is not None:
        save_to = Path(save_to)
        if save_to.suffix.lower() != '.png':
            save_to = save_to.with_suffix('.png')
        save_to = get_unique_name(save_to)
        try:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_to, dpi=dpi)
            if _allow_save_print(print_policy):
                print_color(f"  Timeline plot saved to :{save_to}", 'b')
        except Exception as e:
            print_color(f"Failed to save timeline plot to {save_to}: {e}", 'r')

    fig.canvas.mpl_connect('key_press_event', _close_all_on_key)
    if show:
        plt.show()
    plt.close(fig)

def tst_plot_timeline():
    # tst_f = Path(f"work_dirs/json_models/draft/stream-tst_J-RWL_25ft_3w-1o5/tst-12_F_141_0_0_0_0_timeline.csv")
    tst_f = Path(f"work_dirs/json_models/testing/cam6_11_5_y26/JRWL_25ft_3ws15_bm-148_cam6_11_5_y26_timeline.csv")
    plot_timeline(tst_f, plot_gt=True, gt_offset=0.1)
    plot_timeline(tst_f, ('00:00','05:15'),plot_gt=True,gt_offset=0.5)
    plot_timeline(tst_f, ('04:45', '10:15'), plot_gt=True, gt_offset=0.5)
    plot_timeline(tst_f, ('09:45', '15:15'), plot_gt=True, gt_offset=0.5)
    plot_timeline(tst_f, ('14:45', '20:15'), plot_gt=True, gt_offset=0.5)
    plot_timeline(tst_f, ('19:45', '25:15'), plot_gt=True, gt_offset=0.5)
    plot_timeline(tst_f, ('24:45', '30:15'), plot_gt=True, gt_offset=0.5)
    pass

#748(20,14,5)
if __name__ == "__main__":

    log_path = "./work_dirs/R50_bbrfm_01/20251209_042904/20251209_042904.log"
    rlv_log = "./work_dirs/tsm_R50_MMA_RLVS/20251214_112723/20251214_112723.log"
    #rwf_log = "./work_dirs/tsm_R50_MMA_RWF/20251214_094114/20251214_094114.log"
    rwf_log = "/work_dirs/tsm_R50_MMA_RWF/20251215_023321/20251215_023321.log"
    # l =  "./work_dirs/tsm_R50_MMA_JOINT/20251215_041і232/20251215_041232.log"
    l =  "work_dirs/tsm_R50_MMA_nc2-l4-b4-v/20251231_104730/20251231_104730.log"
    # plot_my_log(rlv_log)
    # plot_my_log(rwf_log)
    # plot_my_log(l)
    # plot_my_log("/mnt/local-data/Python/Projects/weSmart/work_dirs/tsm_R50_MMA_RWF/20251215_023321/20251215_023321.log")
    tst_plot_timeline()
