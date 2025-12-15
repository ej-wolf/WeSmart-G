import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

#* imports from this project
from my_local_utils import _make_unique_dir, _save_log, _load_log_lines

FRAME_H, FRAME_W = 256, 256
CENTER_SIZE = 0.2


def json_to_box_frames( json_path, out_root,
    H: int = FRAME_H,   W: int = FRAME_W, **kwargs):
    """  Convert detector JSON into a sequence of box-frame images.
    Each frame is rendered on an H×W monochrome canvas using the
    normalized TL/BR bounding boxes in data["frames"][i]["bbs"].
    :param json_path: Path to detector JSON.
    :param out_root: Root directory for the output clip folder.
    :param H, W: Output frame height and width in pixels.
    Returns:  (clip_name, num_frames)
    """

    json_path = Path(json_path)
    out_root = Path(out_root)

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    title = data.get("video", "")

    # If no frames, return empty info
    if not frames:
        return None, {'Duration': 0.0, 'Frames': 0, 'Events': 0, 'T_evn': 0.0, 'stat': {},}
        # clip_name = kwargs.get("clip_name") or json_path.stem
        # clip_info = {'Time': 0.0, 'Frames': 0, 'Events': 0, 'T_evn': 0.0, 'stat': {},}
        # return clip_name, clip_info

    #* Derive clip name if not given (replace '.' to be safe in folder names)
    if kwargs.get("clip_name", None) is not None:
        clip_name = kwargs.get("clip_name", None)
    elif  kwargs.get("use_file_name", False):
        # Use the JSON filename (no extension), sanitized
        clip_name = json_path.stem.replace(".", "_")
    elif isinstance(title, str) and len(title) > 0:
        clip_name = Path(title).stem
    else:
        clip_name = json_path.stem.replace(".", "_")
    #* Create unique directory for this clip
    clip_dir, clip_name = _make_unique_dir(out_root, clip_name, no_space=True)

    #* collect logs lines
    event_lines = []
    #* Collect clip-level stats
    times_s: list[float] = []
    event_flags: list[bool] = []
    stat: dict[int, int] = {}

    for idx, fr in enumerate(frames, start=1):
        # White background
        img = np.full((H, W), 255, dtype=np.uint8)

        bb_ls = fr.get('bbs',[]) or fr.get('bbs_list_of_keypoints', [])
        for bb in bb_ls:
            if len(bb) < 6:
                continue
            cls_id, cls_conf, tl_x, tl_y, br_x, br_y = bb[0:6]

            #* Optional vertical flip if y is measured from bottom
            if kwargs.get('y_from_bottom',):
                tl_y = 1.0 - tl_y
                br_y = 1.0 - br_y

            #* Convert normalized coords to pixel coords
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

        frame_num = fr.get('f', idx)
        # time_ms = int(fr.get("t", 0) * 1000)
        t = float(fr.get('t', 0.0))
        time_ms = int(t*1000)
        times_s += [t]


        #* Extract group events for this frame
        raw_events = fr.get('event_grouped') or fr.get('group_events') or []
        pairs = []
        is_event = False
        for ev in raw_events:
            if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                flag = int(ev[0])
                conf = ev[1]
            elif isinstance(ev, (int, float)):
                flag = int(ev)
                conf = None
            else:
                continue

            is_event = True
            #* Per-flag counts (by frame occurrences)
            stat[flag] = stat.get(flag, 0) + 1

            conf_str = "NA" if conf is None else f"{float(conf):.4f}"
            pairs.append(f"{flag}:{conf_str}")

        event_flags += [is_event]
        if pairs:
            events_str = "[" + "; ".join(pairs) + "]"
            event_lines.append(f"{time_ms},{frame_num},{events_str}")

        out_name = clip_dir/f"frm_{frame_num:05d}_{time_ms:08d}.jpg"
        cv2.imwrite(str(out_name), img)

    if event_lines:  #* Write event log if any
        _save_log(event_lines, clip_dir / f"{clip_name}_events.log")


    #*** handel clip_info ***
    # if times_s:
    #     if len(times_s) > 1:
    #         # duration: from first to last timestamp
    #         duration = max(times_s) - min(times_s)
    #         dt = duration/(len(times_s) - 1) if duration > 0 else 0.0
    #         # if duration > 0:
    #         #     dt = duration / (len(times_s) - 1)
    #         # else:
    #         #     dt = 0.0
    #     else:
    #         duration = 0.0
    #         dt = 0.0
    #
    #     # Approx total event time as (num event frames) * dt
    #     # event_frames = sum(1 for v in event_flags if v)
    #     # t_event = event_frames * dt
    #     t_event = sum(event_flags)*dt
    # else:
    #     duration = 0.0
    #     t_event = 0.0
    if times_s:
        duration = max(times_s) - min(times_s)
        dt = duration/(len(times_s) - 1) if duration > 0 else 0.0
        t_event = sum(event_flags)*dt
    else:
        duration = 0.0
        t_event = 0.0


    #* Count contiguous event segments (any event flag)
    events_count = 0
    prev = False
    for cur in event_flags:
        if cur and not prev:
            events_count += 1
        prev = cur
    # num_frames = len(frames)
    clip_info = {'Duration': max(times_s),  # float(total_time),
                 'Frames': len(frames),
                 'Events': events_count,
                 'T_evn': float(t_event),
                 'stat': stat,}

    return clip_name, clip_info # len(frames)


def process_json_folder_(json_root, out_root, H: int = 256, W: int = 256, **kwargs):
    """ Process all JSON files in a folder into box-frame clips.
    Args:
    :param json_root: Folder containing detector JSON files (recursively).
    :param out_root: Root folder for all generated clips.
    :param H, W: Output frame size.
    :param kwargs: arguments to pass forward
    Returns:  A list of (clip_name, num_frames, json_path).
    """
    json_root = Path(json_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for json_path in sorted(json_root.rglob("*.json")):
        try:
            clip_name, num_frames = json_to_box_frames(
                json_path, out_root, H=H, W=W, clip_name=None, **kwargs)
            results.append((clip_name, num_frames, str(json_path)))
        except Exception as e:
            # Robust processing: report failure and continue with next file
            print(f"[ERROR] Failed processing {json_path}:\n-{e}")
            results.append((None, 0, str(json_path)))
    return results


def process_json_folder(json_root, data_root, H: int = FRAME_H, W: int = FRAME_W, **kwargs):
    """  Process all JSON files in a folder into box-frame clips.
    Returns:  list of (clip_name, clip_info, json_path)
    """

    json_root = Path(json_root)
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # # Control whether to start a new clips-stat log or append to the latest one
    # new_stat_log = bool(kwargs.pop('new_stat_log', False))

    results = []
    for json_path in sorted(json_root.rglob("*.json")):
        try:
            clip_name, clip_info = json_to_box_frames(
                json_path, data_root, H=H, W=W, clip_name=None, **kwargs)
            results.append((clip_name, clip_info, str(json_path)))
        except Exception as e:
            # Robust processing: report failure and continue with next file
            print(f"[ERROR] Failed processing {json_path}: {e}")
            # results.append((None, {'duration': 0.0, 'Frames': 0, 'Events': 0, 'T_evn': 0.0, 'stat': {}}, str(json_path)))
            results+= [None, {'duration': 0.0, 'Frames': 0, 'Events': 0, 'T_evn': 0.0, 'stat': {}}, str(json_path) ]

    # Build per-clip stats log under data_root
    stats_rows: list[str] = []
    for clip_name, clip_info, json_path_str in results:
        if not clip_name or not clip_info:
            continue
        stat_dict = clip_info.get('stat', {}) or {}
        if stat_dict:
            pairs = [f"{k}:{v}" for k, v in sorted(stat_dict.items())]
            stat_str = '[' + '.'.join(pairs) + ']'
        else:
            stat_str = ''
        row = ','.join([str(clip_name),
                        str(clip_info.get('Duration', 0.0)),
                        str(clip_info.get('Frames', 0)),
                        str(clip_info.get('Events', 0)),
                        str(clip_info.get('T_evn', 0.0)),
                        stat_str])
        stats_rows.append(row)

    # Control whether to start a new clips-stat log or append to the latest one
    # new_stat_log = bool(kwargs.pop('new_stat_log', False))

    if stats_rows:
        # Decide which log file to use
        if kwargs.pop('new_stat_log', False):
            log_path = None
        else:
            existing_logs = sorted(data_root.glob('clips_stat_*.log'),
                                   key=lambda p: p.stat().st_mtime,)
            log_path = existing_logs[-1] if existing_logs else None

        if log_path is None:
            # stamp = datetime.now().strftime("%y%m%d-%H%S")
            log_path = data_root/f"clips_stat_{datetime.now().strftime('%y%m%d-%H%S')}.log"
            #* New file: write header
            with log_path.open("w", encoding="utf-8") as f:
                f.write("clip_name,duration,Frames,Events,T_evn,stat\n")
                for row in stats_rows:
                    f.write(row + "\n")
        else:
            # Append to existing stats log
            with log_path.open('a', encoding='utf-8') as f:
                for row in stats_rows:
                    f.write(row + '\n')

    return results
