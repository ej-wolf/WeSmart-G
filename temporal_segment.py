import os
import random
from pathlib import Path

VIOLENCE_FLAGS = {3, 4}

WIN_LEN = 20
STRIDE = 10
MIN_FOR_EVENT = 1

#* split parameters
TRAIN_VAL_RATIO = 0.8
RANDOM_SEED = 42


#* file name
#ToDo: use config files
SEGMENTS_DIR = "data/cache"
LABELS_FILE = "all_label.txt"

def load_events_log(log_path: Path):
    """Parse <clip>_events.log into a set of frames that contain violence."""
    violence_frames = set()

    if not log_path or not log_path.is_file():
        return violence_frames

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # time_ms,frame,[flag:conf; ...]
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            try:
                frame_num = int(parts[1].strip())
            except ValueError:
                continue
            events_str = parts[2].strip()
            # events_str like: [3:0.7000; 5:0.9000]
            if not (events_str.startswith("[") and events_str.endswith("]")):
                continue
            inner = events_str[1:-1]
            for seg in inner.split(";"):
                seg = seg.strip()
                if not seg:
                    continue
                # flag:conf or flag:NA
                flag_part = seg.split(":", 1)[0]
                try:
                    flag = int(flag_part)
                except ValueError:
                    continue
                if flag in VIOLENCE_FLAGS:
                    violence_frames.add(frame_num)
                    break  # this frame is violent, no need to check more
    return violence_frames


def segment_clip_dir( clip_dir: Path,  dst_root: Path,
    win_len:int = WIN_LEN,  stride:int = STRIDE,
    min_for_event:int = MIN_FOR_EVENT):
    """  Segment a single clip_dir into fixed-length windows using hardlinks.
    Returns:  list of (segment_name, num_frames, label)
    """
    assert clip_dir.is_dir()
    dst_root.mkdir(parents=True, exist_ok=True)

    clip_name = clip_dir.name
    log_path = next(clip_dir.glob("*_events.log"), None)
    violence_frames = load_events_log(log_path) if log_path else set()

    # Collect frames sorted by frame number (from filename frm_<frame>_<time>.jpg)
    frame_entries = []
    for p in sorted(clip_dir.glob("frm_*_*.jpg")):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            frame_num = int(parts[1])
            time_ms = int(parts[2])
        except ValueError:
            continue
        frame_entries.append((frame_num, time_ms, p))

    if not frame_entries:
        print(f"[INFO] No frames in {clip_dir}")
        return []

    # Sort by frame number (you could sort by time_ms instead; both should be monotonic)
    frame_entries.sort(key=lambda x: x[0])

    segments = []
    num_frames_total = len(frame_entries)

    # Sliding window over indices
    start_idx = 0
    while start_idx + win_len <= num_frames_total:
        window = frame_entries[start_idx : start_idx + win_len]
        window_frames = [f[0] for f in window]

        # Label: violence if at least min_pos_frames frames are in violence_frames
        violent_count = sum(1 for f in window_frames if f in violence_frames)
        label = 1 if violent_count >= min_for_event else 0

        # Make destination dir name
        segment_name = f"{clip_name}_w{start_idx:05d}"
        segment_dir = dst_root/segment_name
        segment_dir.mkdir(parents=True, exist_ok=False)

        #* Hardlink frames into segment dir
        # for frame_num, time_ms, src_path in window:
        #     dst_name = segment_dir/src_path.name
        #* Hardlink frames into segment dir as img_00001.jpg, img_00002.jpg, ...
        for i, (frame_num, time_ms, src_path) in enumerate(window, start=1):
            #  dst_name = segment_dir/f"img_{i:05d}.jpg"
            try:
                # os.link(src_path, dst_name)
                os.link(src_path, segment_dir/f"img_{i:05d}.jpg")
            except FileExistsError:
                # If link already exists, skip
                pass

        segments.append((segment_name, win_len, label))

        start_idx += stride

    return segments


def segment_all_clips(src_root, dst_root,
                      win_len: int = WIN_LEN, stride: int = STRIDE,
                      min_for_event:int = MIN_FOR_EVENT):
    """  Segment all clip dirs under src_root into window clips under dst_root.
    Returns: list of (segment_name, num_frames, label)
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    all_segments = []
    for clip_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        segs = segment_clip_dir(clip_dir, dst_root,
                                win_len=win_len, stride=stride,
                                min_for_event=min_for_event )
        all_segments.extend(segs)

    print(f"[INFO] Created {len(all_segments)} segments in {dst_root}")
    return all_segments


def make_labels_file(segments, file_name=None):

    if file_name is None:
        file_name = Path(os.getcwd())/SEGMENTS_DIR/LABELS_FILE

    with file_name.open("w", encoding="utf-8") as f:
        for name, n_frm, lbl in segments:
            f.write(f"{name} {n_frm} {lbl}\n")
    print(f"[INFO] Create annotation file: {file_name}")


def load_labels_file(path):
    path = Path(path)
    segments = []
    segments2 = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, n_frm, lbl = line.split()
            segments.append((name, int(n_frm), int(lbl)))
            segments2 += [(name, int(n_frm), int(lbl))]
    return segments

def split_segments(segments, ratio=TRAIN_VAL_RATIO, seed=RANDOM_SEED):

    rng = random.Random(seed)
    segs = list(segments)
    rng.shuffle(segs)
    # n_total = len(segs)
    n_train = int(round(ratio * len(segs)))
    return segs[:n_train], segs[n_train:]


def make_train_val_ds(full_ds, dst_path, ratio=TRAIN_VAL_RATIO):

    if isinstance(full_ds, str) or isinstance(full_ds, Path):
        segments = load_labels_file(full_ds)
    elif isinstance(full_ds, list):
        segments =  full_ds
    else:
        print("Some Error"); return

    trn_segs, val_segs = split_segments(segments, ratio=ratio)

    make_labels_file(trn_segs, dst_path/"train.txt")
    make_labels_file(val_segs, dst_path/"val.txt")


if __name__ == "__main__":
    pass
    # make_train_val_ds ()
    # Example usage; adapt paths as needed
    # cwd = Path(os.getcwd())
    # src_path = cwd/'data'/'frames'
    # dst_path = cwd/'cache'
    # segments = segment_all_clips(src_path, dst_path)
    #
    # ann_path = dst_path/"all_windows.txt"
    # with ann_path.open("w", encoding="utf-8") as f:
    #     for name, n_frames, label in segments:
    #         f.write(f"{name} {n_frames} {label}\n")
    # print(f"[INFO] Wrote annotations to {ann_path}")
