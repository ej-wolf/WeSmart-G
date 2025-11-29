import cv2
import json
from pathlib import Path
import numpy as np

from my_local_utils import _make_unique_dir, _save_log, _load_log_lines

FRAME_H, FRAME_W = 256, 256
CENTER_SIZE = 0.2

def json_to_box_frames(
    json_path, out_root,
    H: int = FRAME_H,   W: int = FRAME_W,
    # clip_name: str | None = None,
    **kwargs):
    """  Convert detector JSON into a sequence of box-frame images.
    Each frame is rendered on an H×W monochrome canvas using the
    normalized TL/BR bounding boxes in data["frames"][i]["bbs"].
    :param
        json_path: Path to detector JSON.
        out_root: Root directory for the output clip folder.
        H, W: Output frame height and width in pixels.
    Returns:  (clip_name, num_frames)
    """

    json_path = Path(json_path)
    out_root = Path(out_root)

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]

    # Derive clip name if not given (replace '.' to be safe in folder names)
    # if clip_name is None:
    #     clip_name = json_path.stem.replace(".", "_")
    # clip_dir = out_root/clip_name
    if kwargs.get("clip_name", None) is None:
        video_field = data.get("video", "")
        if isinstance(video_field, str) and video_field:
            clip_name = Path(video_field).stem
        else:
            clip_name = json_path.stem.replace(".", "_")

    # clip_dir.mkdir(parents=True, exist_ok=True)
    # Create unique directory for this clip
    clip_dir, clip_name = _make_unique_dir(out_root, clip_name)

    event_lines = []
    for idx, fr in enumerate(frames, start=1):
        # White background
        img = np.full((H, W), 255, dtype=np.uint8)

        bb_ls = fr.get('bbs',[]) or fr.get('bbs_list_of_keypoints', [])
        #for bb in fr.get("bbs", []):
        for bb in bb_ls:
               # print(f"bb = {len(bb)}")
            if len(bb) < 6:
                continue
            cls_id, cls_conf, tl_x, tl_y, br_x, br_y = bb[0:6]

            # Optional vertical flip if y is measured from bottom
            if kwargs.get('y_from_bottom',):
                tl_y = 1.0 - tl_y
                br_y = 1.0 - br_y

            # Convert normalized coords to pixel coords
            x1 = int(tl_x*W)
            y1 = int(tl_y*H)
            x2 = int(br_x*W)
            y2 = int(br_y*H)

            # Clip to image bounds
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Intensity mapping (for now: black box outlines).
            intensity = 0
            cv2.rectangle(img, (x1, y1), (x2, y2), int(intensity), thickness=1)

            if kwargs.get("draw_center", False):
                bb_w = x2 - x1
                bb_h = y2 - y1
                if bb_w > 0 and bb_h > 0:
                    cx = x1 + bb_w//2
                    cy = y1 + bb_h//2
                    diameter = max(5, int(CENTER_SIZE*min(bb_w, bb_h)))
                    radius = max(1, diameter // 2)
                    cv2.circle(img, (cx, cy), radius, int(intensity), thickness=-1)

        frame_num = fr.get("f", idx)
        time_ms = int(fr.get("t", 0) * 1000)
        # Extract group events for this frame
        raw_events = fr.get("event_grouped") or fr.get("group_events") or []
        pairs = []
        for ev in raw_events:
            if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                flag = int(ev[0])
                conf = ev[1]
            elif isinstance(ev, (int, float)):
                flag = int(ev)
                conf = None
            else:
                continue
            conf_str = "NA" if conf is None else f"{float(conf):.4f}"
            pairs.append(f"{flag}:{conf_str}")

        if pairs:
            events_str = "[" + "; ".join(pairs) + "]"
            event_lines.append(f"{time_ms},{frame_num},{events_str}")

        # out_name = clip_dir / f"img_{idx:05d}.jpg"
        # out_name = clip_dir/f"frm_{fr.get('f', idx):05d}_{int(fr.get('t', 0)*1000):08d}.jpg"
        out_name = clip_dir/f"frm_{frame_num:05d}_{time_ms:08d}.jpg"
        cv2.imwrite(str(out_name), img)

    if event_lines:  #* Write event log if any
        _save_log(event_lines, clip_dir / f"{clip_name}_events.log")

    num_frames = len(frames)
    return clip_name, len(frames)


def process_json_folder(json_root, out_root, H: int = 256, W: int = 256, **kwargs):
    """ Process all JSON files in a folder into box-frame clips.
    Args:
    :param json_root: Folder containing detector JSON files (recursively).
    :param out_root: Root folder for all generated clips.
    :param H, W: Output frame size.
    :param kwargs: arguments to pass forward
    Returns:
        A list of (clip_name, num_frames, json_path).
    """
    json_root = Path(json_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for json_path in sorted(json_root.rglob("*.json")):
        # clip_name, num_frames = json_to_box_frames(
        #     json_path, out_root, H=H, W=W, clip_name=None,y_from_bottom=y_from_bottom,
        # )
        try:
            clip_name, num_frames = json_to_box_frames(
                json_path, out_root, H=H, W=W, clip_name=None, **kwargs)
            results.append((clip_name, num_frames, str(json_path)))
        except Exception as e:
            # Robust processing: report failure and continue with next file
            print(f"[ERROR] Failed processing {json_path}:\n-{e}")
            results.append((None, 0, str(json_path)))
    return results



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
