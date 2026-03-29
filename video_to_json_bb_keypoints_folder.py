"""
    * The unit converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default tags.

    * The unit is wrapped into CLI in convert_json.py via process_video function.

    * JSON structure:
           {'video' : "...",           # video/file name
            'fps': 25.0,
            'step': 5,                # step size
            'frames':[
                    {'f': 15,               #* frame index
                     't': 0.6,              #* time from start in seconds
                     'individual_events': [],
                     'group_events': [2, 3],
                     'detection_list':[     #* list of YOLO detections (human)
                              {'class':     #* normal,  fall ...
                               'conf':      #* confidence of 'class'
                               'bbox':      #* [x1, y1, x2, y2] in normalized unit
                               'key_points': [x1, y1, c1, x2, y2, c2, ....  x17, y17, c17]
                                            #* flatten vector of 17 key points (x,y) and it's confidence (c)
                                       ]
                    },]
             `      ...
"""

import cv2, json, re, shutil, statistics, subprocess, torch
from pathlib import Path
from ultralytics import YOLO
#* import from my utils
from my_local_utils import get_unique_name, print_color

#* Events Thresholds  -------------------------------------------------------------------
# SINGLE_THRESHOLDS = { 0: 0.5,   #* normal
#                       1: 0.9,   #* abnormal
#                       2: 0.7,   #* fall
#                       5: 0.9,}  #* kick
# GROUPED_THRESHOLDS = {3: 0.7,   #* tension
#                       4: 0.7,}  #* violence
DETECTION_THRESHOLD = 0.5
DEFAULT_SAMPELING = 5
# STEP = 5

#* Events flags
TAG_NO_EVENT = 0
TAG_FALL     = 2
TAG_TENSION  = 3
TAG_FIGHT    = 4

MODELS_DIR = 'models/'
DEFAULT_YOLO = MODELS_DIR + "yolo26x-pose.pt"
#* "yolo26x-pose.pt" / "yolo11x-pose.pt" / "yolov8s.pt"
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv', '.mts', '.m2ts'}


def parse_rational(value):
    """Convert ffprobe rational strings like '30000/1001' to float."""
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        num, den = value.split("/", 1)
        den = float(den)
        if den == 0:
            return None
        return float(num) / den
    except (TypeError, ValueError):
        return None


def probe_video_timing_ffprobe(video_path: Path):
    """Read stream timing metadata and per-frame timestamps with ffprobe."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return None

    stream_cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,time_base,nb_frames,duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        stream_proc = subprocess.run(stream_cmd, check=True, capture_output=True, text=True)
        stream_payload = json.loads(stream_proc.stdout or "{}")
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"[WARN] ffprobe stream metadata failed for {video_path}: {exc}")
        return None

    frame_cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    frame_times = []
    try:
        with subprocess.Popen(frame_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    frame_times.append(float(line))
                except ValueError:
                    continue
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            return_code = proc.wait()
        if return_code != 0:
            print(f"[WARN] ffprobe frame timestamps failed for {video_path}: {stderr.strip()}")
            return None
    except OSError as exc:
        print(f"[WARN] Could not launch ffprobe for {video_path}: {exc}")
        return None

    stream_info = (stream_payload.get("streams") or [{}])[0]
    avg_frame_rate = parse_rational(stream_info.get("avg_frame_rate"))
    nominal_frame_rate = parse_rational(stream_info.get("r_frame_rate"))
    deltas = [curr - prev for prev, curr in zip(frame_times, frame_times[1:]) if curr >= prev]
    median_delta = statistics.median(deltas) if deltas else None
    jitter_tolerance = max(0.002, 0.10 * median_delta) if median_delta else None
    variable_timing = bool(
        median_delta
        and jitter_tolerance
        and any(abs(delta - median_delta) > jitter_tolerance for delta in deltas)
    )
    return {
        "frame_times": frame_times,
        "frame_timestamp_count": len(frame_times),
        "avg_frame_rate": avg_frame_rate,
        "nominal_frame_rate": nominal_frame_rate,
        "stream_time_base": stream_info.get("time_base"),
        "stream_duration_sec": float(stream_info["duration"]) if stream_info.get("duration") not in (None, "N/A") else None,
        "stream_nb_frames": int(stream_info["nb_frames"]) if stream_info.get("nb_frames", "").isdigit() else None,
        "median_frame_delta": median_delta,
        "variable_frame_timing": variable_timing,
    }


def is_video_file(path: Path | str) -> bool:
    """Return True when the path looks like a supported video file."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES

def parse_sec_str(t):
    """ Convert 'HH:MM:SS' or 'MM:SS' or 'SS' to seconds"""
    if t is None:
        return None
    t = t.strip()
    if t == '':
        return None

    parts = t.split(':')
    parts = [int(p) for p in parts]
    if   len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:  #* seconds only
        h,m = 0, 0
        s = parts[0]
    return h * 3600 + m * 60 + s


def parse_interval(s):
    """ Parses 'start-end' where start or end can be empty.
    Examples:
        '00:01:00-00:02:00'
        '00:03:00-'
        '-00:00:30'
        '00:01:00'   -> (start, None)
    Returns (start_sec, end_sec) where any can be None.
    """
    if s is None:
        return None, None

    s = s.strip()
    if s == "":
        # Completely empty → ignore this interval later
        return None, None

    # No '-' → interpret as single time = from that time until end
    if "-" not in s:
        start = parse_sec_str(s)
        end = None
        return start, end

    start_str, end_str = s.split("-", 1)

    start = parse_sec_str(start_str) if start_str.strip() != '' else None
    end   = parse_sec_str(end_str)   if end_str.strip()   != '' else None
    return start, end


def parse_annotation_file(ann_file):
    """ Parse an annotation text file into event intervals by flag.
        Expected row format:
        start_time,\t   end_time,\t event_flag
        '#' Lines starting with '#' are ignored.
    """
    event_intervals = {TAG_FALL: [], TAG_TENSION: [], TAG_FIGHT: []}
    ann_path     = Path(ann_file)

    with ann_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",\t")]
            if len(parts) != 3:
                parts = [part.strip() for part in line.split("\t")]

            if len(parts) != 3:
                print(f"[WARN] Invalid annotation row at {ann_path}:{line_no}: {raw_line.rstrip()}")
                continue

            start_str, end_str, flag_str = parts
            try:
                event_flag = int(flag_str)
            except ValueError:
                print(f"[WARN] Invalid event flag at {ann_path}:{line_no}: {flag_str}")
                continue

            if event_flag not in event_intervals:
                continue

            event_intervals[event_flag].append((parse_sec_str(start_str), parse_sec_str(end_str)))

    return event_intervals

#*  --- Main function, to be used from other unit ---------------------------------------
def process_video(input_path: Path|str,
                  output_path:Path|str=None,
                  sample_rate=DEFAULT_SAMPELING,
                  conf_thresh=DETECTION_THRESHOLD,
                  ann_file=None,
                  default_group_tag=None,
                  tension_intervals=None, fight_intervals=None, fall_intervals=None,
                  **kwargs):
    """ Converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default
    tags. Process video file/dir of files.
    Parameters:
    :param Path input_path: file or dir
            - If file  → process single video
            - If dir   → process all files inside
    :param output_path : file, dir or None
        Single input:
            - None            → save next to video (same name, .json)
            - directory       → save inside directory with video name
            - file path       → save exactly to that file
        Multiple input:
            - None            → create 'jsons' subdir and enumerate
            - directory       → enumerate inside directory
            - file path       → enumerate using its stem (e.g. train_001.json)
    Conversion related parameters:
    :param sample_rate:        → Sampling rate for JSON conversion (in Hz)
    :param conf_thresh:        → YOLO detection confidence threshold
    :param default_group_tag:  → Assign default events to all frames by default
    :param ann_file:           → Optional annotation text file with rows:
                               start_time,\t  end_time,\t    event_flag
                               Supported event flags: 2 (fall), 3 (tension), 4 (fight)
                               Precedence:
                               defaults < ann_file intervals < explicit *_intervals arguments
                               If ann_file is None, the code will look next to each video for:
                               <video_stem>.txt, <video_stem>.ann, or <video_stem>
    :param kwargs['time_source']:
                               'fps'     → derive time from frame_idx / fps metadata
                               'ffprobe' → derive time from real per-frame timestamps
    """

    #* local helpers sub-function
    def find_annotation_file(video_path):
        """  Find a sibling annotation file matching the video stem."""
        video_path = Path(video_path)
        candidates = (video_path.with_suffix(".txt"),
                      video_path.with_suffix(".ann"),
                      video_path.with_suffix(""))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def find_txt_annotation_file(video_path):
        """Find a sibling .txt annotation file matching the video stem."""
        candidate = Path(video_path).with_suffix(".txt")
        return candidate if candidate.is_file() else None

    def sanitize_name_part(part):
        sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', str(part).strip())
        return sanitized.strip('_') or 'unnamed'

    def build_output_stem(video_path):
        video_path = Path(video_path)
        if json_name is not None:
            return json_name
        if not name_from_video_path:
            return sanitize_name_part(video_path.stem)

        stem_path = video_path.with_suffix('')
        parts = list(stem_path.parts)
        anchor_idx = None
        for idx, part in enumerate(parts):
            if part.lower() in {'video', 'videos'}:
                anchor_idx = idx
        if anchor_idx is not None and anchor_idx + 1 < len(parts):
            raw_parts = parts[anchor_idx + 1:]
        elif stem_path.parent.name:
            raw_parts = [stem_path.parent.name, stem_path.name]
        else:
            raw_parts = [stem_path.name]

        clean_parts = [sanitize_name_part(part) for part in raw_parts]
        return '__'.join(part for part in clean_parts if part) or sanitize_name_part(video_path.stem)

    def as_event_list(tags):
        if tags is None:
            return []
        if isinstance(tags, (list, tuple, set)):
            return list(tags)
        return [tags]

    def in_any_interval(t, intervals):
        """ Return True if time_sec falls inside any valid interval.
            Supports open intervals (None bounds) """
        for start, stop in intervals:
            # * (None, None) means "ignore" → skip
            if start is None and stop is None:
                continue
            if start is None:
                start = 0.0
            if stop is None:
                stop = video_duration_sec
            if start <= t <= stop:
                return True
        return False

    def write_incomplete_marker(marker_path: Path, lines: list[str]):
        marker_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tension_intervals = tension_intervals or []
    fight_intervals = fight_intervals or []
    fall_intervals = fall_intervals or []
    allow_incomplete = bool(kwargs.get('allow_incomplete', False))
    requested_time_source = str(kwargs.get('time_source', 'fps')).lower()
    only_with_txt_ann = bool(kwargs.get('only_with_txt_ann', False))
    name_from_video_path = bool(kwargs.get('name_from_video_path', False))
    if requested_time_source not in {'fps', 'ffprobe'}:
        raise ValueError(f"Invalid time_source: {requested_time_source}")

    model_path = kwargs.get('model_path', None) or DEFAULT_YOLO
    model = YOLO(model_path if Path(model_path).is_file() else DEFAULT_YOLO)
    show = kwargs.get('show', False)
    
    input_path = Path(input_path)
    if input_path.is_dir():
        all_files = [p for p in input_path.iterdir() if p.is_file()]
        vid_list = [p for p in all_files if is_video_file(p)]
        ignored_non_video_count = len(all_files) - len(vid_list)
        if ignored_non_video_count:
            print(f"[INFO] Ignoring {ignored_non_video_count} non-video files in {input_path}")
        if only_with_txt_ann and ann_file is None:
            filtered_vid_list = []
            for vid_path in vid_list:
                if find_txt_annotation_file(vid_path) is not None:
                    filtered_vid_list.append(vid_path)
                else:
                    print(f"[INFO] Skipping {vid_path.name}: no sibling .txt annotation file")
            vid_list = filtered_vid_list
    else:
        vid_list = [input_path]

    output_path = Path(output_path) if output_path else None

    if output_path is None:
        json_dir = input_path if input_path.is_dir() else input_path.parent
        json_name = None
    elif output_path.suffix == '.json':
        json_dir = output_path.parent
        json_name = output_path.stem
    elif output_path.is_dir():
        json_dir = output_path
        json_name = None
    else:
        #ToDo: handle it
        json_dir = output_path
        json_name = None

    json_dir.mkdir(parents=True, exist_ok=True)

    detector_info = {'model':Path(model_path).stem, 'version':model.ckpt['version'], 'threshold':conf_thresh}
    print_color(f"YOLO:\nversion - {detector_info['version']}\nthreshold = {detector_info['threshold']}", 'b')
    print(f"Default group event: {default_group_tag}\n")
    if name_from_video_path:
        print("[INFO] Output naming mode: path-based names after the last 'video'/'videos' folder")

    ann_intervals = parse_annotation_file(ann_file) if ann_file else None
    for vid_path in vid_list:
        output_stem = build_output_stem(vid_path)
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video (probably corrupted): {vid_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        meta_frame_count = int(round(frame_count)) if frame_count > 0 else None
        print('frame_count', frame_count)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('frame_size', w, h)
        if w != 1920:
            h = int(w*1080/1920)
        if fps <= 0:
            fps = 25.0  # fallback if metadata is broken
        timing_info = {'requested_source': requested_time_source,
                       'source': 'fps',
                       'opencv_fps': fps}
        frame_times = None
        if requested_time_source == 'ffprobe':
            ffprobe_timing = probe_video_timing_ffprobe(vid_path)
            if ffprobe_timing is None or not ffprobe_timing['frame_times']:
                print(f"[WARN] Falling back to fps-based timing for {vid_path.name}")
            else:
                frame_times = ffprobe_timing['frame_times']
                timing_info.update({k: v for k, v in ffprobe_timing.items() if k != 'frame_times'})
                timing_info['source'] = 'ffprobe'
        reference_duration = frame_count / fps if frame_count > 0 else 0.0
        if frame_times:
            reference_duration = max(reference_duration, frame_times[-1])
        video_duration_sec = reference_duration
        print(f"*** Converting {vid_path.name},{(w, h)}p {video_duration_sec} s")
        print(
            f"Timing source: {timing_info['source']}; "
            f"opencv fps={fps:.3f}; "
            f"ffprobe avg fps={timing_info.get('avg_frame_rate')}"
        )
        if timing_info.get('variable_frame_timing'):
            print(f"[WARN] {vid_path.name} has variable frame timing. Use time_source='ffprobe' for accurate timestamps.")
        video_ann_intervals = ann_intervals
        if video_ann_intervals is None:
            auto_ann_file = find_txt_annotation_file(vid_path) if only_with_txt_ann else find_annotation_file(vid_path)
            if auto_ann_file is not None:
                print(f"Using annotation file: {auto_ann_file}")
                video_ann_intervals = parse_annotation_file(auto_ann_file)

        tension_intervals_sec = list(video_ann_intervals[TAG_TENSION]) if video_ann_intervals is not None else []
        fight_intervals_sec = list(video_ann_intervals[TAG_FIGHT]) if video_ann_intervals is not None else []
        fall_intervals_sec = list(video_ann_intervals[TAG_FALL]) if video_ann_intervals is not None else []

        tension_intervals_sec.extend(parse_interval(s) for s in tension_intervals if s and s.strip())
        fight_intervals_sec.extend(parse_interval(s) for s in fight_intervals if s and s.strip())
        fall_intervals_sec.extend(parse_interval(s) for s in fall_intervals if s and s.strip())

        target_sampling = float(kwargs.get('sample_rate', sample_rate))
        if target_sampling <= 0 or target_sampling > fps  :
        #* i.e    0 <= sampling_rate_Hz <= fps
            raise ValueError(f"Invalid sampling rate: {target_sampling} Hz")

        sample_period = 1.0 / target_sampling
        legacy_step = max(1, int(round(fps/target_sampling)))
        frame_interval_sec = timing_info.get('median_frame_delta') or (1.0 / fps)
        frame_time_tolerance = 0.5 * frame_interval_sec
        print_color(
            f"target sampling = {target_sampling:.3f} Hz -> sample period = {sample_period:.3f} s "
            f"(legacy approx step = {legacy_step})"
        )

        print(
            f"Sampling setup: Video fps={fps:.3f}, Sampling rate={target_sampling:.3f} Hz,"
            f" sample period={sample_period:.3f} s, legacy approx step={legacy_step} frames"
        )

        frames = []
        frame_idx = 0
        decode_stop_idx = None
        decode_stop_pos = None
        next_sample_time = 0.0
        ffprobe_short_warning_shown = False
        THRESH = 0.5
        while True:
            ret, frame = cap.read()
            if not ret:
                decode_stop_idx = frame_idx
                decode_stop_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"[INFO] cv2.VideoCapture.read() returned False at frame_idx={frame_idx}, cap_pos={decode_stop_pos}")
                break
            if frame is None:
                decode_stop_idx = frame_idx
                decode_stop_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"[WARN] Decoder returned an empty frame at frame_idx={frame_idx}, cap_pos={decode_stop_pos}")
                break

            frame = cv2.resize(frame, (w, h))
            if frame_times is not None and frame_idx < len(frame_times):
                time_sec = frame_times[frame_idx]
            else:
                if frame_times is not None and not ffprobe_short_warning_shown:
                    print(
                        f"[WARN] ffprobe timestamps ended at frame {len(frame_times)} for {vid_path.name}; "
                        f"falling back to fps-based timing from frame {frame_idx}"
                    )
                    ffprobe_short_warning_shown = True
                time_sec = frame_idx / fps
            if time_sec + frame_time_tolerance < next_sample_time:
                frame_idx += 1
                continue

            #* run model on current frame
            try:
                results = model(frame, conf=conf_thresh, verbose=False)[0]
            except Exception as exc:
                print(f"[ERROR] YOLO inference failed at frame_idx={frame_idx}: {exc}")
                raise

            detection_list = []
            if results.boxes:
                for box, kpts_norm, conf_kpts in zip(results.boxes, results.keypoints.xyn, results.keypoints.conf):
                    cls_id = int(box.cls)
                    conf = float(box.conf)

                    kpts_xyc = torch.cat([kpts_norm, conf_kpts.unsqueeze(-1)], dim=-1)

                    if cls_id not in [3,4]:
                        x1, y1, x2, y2 = map(float, box.xyxyn[0])
                        for kp_pair in kpts_xyc:
                            cx = int(float(kp_pair[0])*w)
                            cy = int(float(kp_pair[1])*h)
                            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                        detection_list.append({ 'class': cls_id,
                                                'conf': conf,
                                                'bbox': [x1, y1, x2, y2],
                                                'key_points': kpts_xyc[:].flatten().tolist(),
                                                })
                    else:
                        continue

            #* ---- Per-frame event logic ----
            group_events = []
            if in_any_interval(time_sec, fall_intervals_sec):
                group_events.append(TAG_FALL)
            if in_any_interval(time_sec, tension_intervals_sec):
                group_events.append(TAG_TENSION)
            if in_any_interval(time_sec, fight_intervals_sec):
                group_events.append(TAG_FIGHT)

            default_tags = as_event_list(default_group_tag)
            group_tags = group_events if group_events else default_tags

            frames.append({'f': frame_idx, 't': time_sec,
                           'individual_events': [],
                            'group_events': sorted(set(group_tags), reverse=True),
                            'detection_list': detection_list,}
                          )

            next_sample_time += sample_period
            while next_sample_time <= time_sec + frame_time_tolerance:
                next_sample_time += sample_period

            if show:
                cv2.imshow("head_center_debug", frame)
                #Todo: Anna should add keypoints
                key = cv2.waitKey(1) & 0xFF
                if key == 27:   #* ESC to quit
                    break
            frame_idx += 1
            
        if frame_idx == 0:
            print(f"[WARN] Skipping video with no decodable frames: {vid_path} "
                  f"(unsupported codec/container or corrupted file)")
            cap.release()
            cv2.destroyAllWindows()
            continue
        decoded_frame_count = frame_idx
        decode_incomplete = (
            meta_frame_count is not None
            and decode_stop_idx is not None
            and decoded_frame_count + 1 < meta_frame_count
        )
        incomplete_reason = None
        marker_path = json_dir / f"{output_stem}.incomplete.txt"
        if decode_incomplete:
            incomplete_reason = (
                f"Decoder stopped before metadata frame count: decoded={decoded_frame_count}, "
                f"metadata={meta_frame_count}, cap_pos={decode_stop_pos}"
            )
            print(f"[WARN] {incomplete_reason}. This usually means a truncated file, bad frames near the end, or incorrect container metadata.")
        print(
            f"Decoded frames: {decoded_frame_count}; sampled frames saved: {len(frames)}; "
            f"metadata frame count: {meta_frame_count if meta_frame_count is not None else 'unknown'}"
        )
        if incomplete_reason is not None and not allow_incomplete:
            write_incomplete_marker(
                marker_path,
                [
                    f"video={vid_path}",
                    incomplete_reason,
                    f"sample_rate_hz={target_sampling:.6f}",
                    "action=skipped_json_write",
                ],
            )
            print(f"[WARN] Skipping JSON write for incomplete decode. Marker saved to {marker_path}")
            cap.release()
            if show:
                cv2.destroyAllWindows()
            continue
        event_intervals = {'tension': {'raw': tension_intervals, 'sec': tension_intervals_sec},
                           'fight': {'raw': fight_intervals,   'sec': fight_intervals_sec},
                           'fall': {'raw': fall_intervals,    'sec': fall_intervals_sec},
                           },
        data = {'video': str(vid_path),
                'fps': fps,
                'timing': timing_info,
                'sampling rate': {'target': target_sampling,
                                  'effective': len(frames)/max(frames[-1]['t'] - frames[0]['t'], 1.0/fps) if len(frames) > 1 else target_sampling,
                                  'mode': 'time_hz'},
                'step': legacy_step,
                'detector': detector_info,
                'decoded_frame_count': decoded_frame_count,
                'metadata_frame_count': meta_frame_count,
                'decode_complete': incomplete_reason is None,
                'event_intervals':event_intervals,
                'frames': frames,
                }

        #Todo: resolve case when video_path is dir while output_path is a file name
        json_path = get_unique_name(json_dir / f"{output_stem}.json", 4)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if marker_path.exists():
            marker_path.unlink()

        print_color(f"Saved::{len(frames)} sampled frames to {json_path}\n----------------\n'",'b')

        cap.release()
        if show:
            cv2.destroyAllWindows()


#* ------  local runner (wrapper) -----------------------------
#* CLI interface moved to designated wrapper convert_to_json.py
def local_runner(tst_path, **kwargs):
    process_video(tst_path, **kwargs)


if __name__ == "__main__":
    local_runner("/mnt/local-data/Projects/Wesmart/Video-datasets/test_ds/tst_conv",
                 output_path="/mnt/local-data/Python/Projects/weSmart/data/json_files/tst_conv/try_05",
                 default_group_tag = 0,
                 )

#534(5,19,27) -> 333(5,7,6)- 460(4,8,8)/
# 460(5,9,8)->412(4,4,2)
