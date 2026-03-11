"""
    * The unit converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default tags.

    * The unit warped into CLI in convert_json.py via process_video function.

    * JSON structure:
           {'video' : "...",           # video/file name
            'fps': 25.0,
            'step': 5,                # Sampling rate
            'frames':[
                    {'f': 15,               #* frame index
                     't': 0.6,              #* time from start in seconds
                     'event_single': 2,     #* 0,1,2,5 or -1
                     'event_grouped': 3,    #* 3,4 or -1
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

import cv2, json, torch
from pathlib import Path
from ultralytics import YOLO
#* import from my utils
from my_local_utils import get_unique_name, collection, print_color
MODELS_DIR = 'models/'

#* Events Thresholds  -------------------------------------------------------------------
SINGLE_THRESHOLDS = { 0: 0.5,   #* normal
                      1: 0.9,   #* abnormal
                      2: 0.7,   #* fall
                      5: 0.9,}  #* kick
GROUPED_THRESHOLDS = {3: 0.7,   #* tension
                      4: 0.7,}  #* violence
DETECTION_THRESHOLD = 0.5
STEP = 5
#* Events flags
TAG_NO_EVENT = 0
TAG_FALL     = 2
TAG_TENSION  = 3
TAG_FIGHT    = 4



DEFAULT_YOLO = MODELS_DIR + "yolo26x-pose.pt"
    # "yolo26x-pose.pt" / "yolo11x-pose.pt" / "yolov8s.pt"
DEFAULT_JSON_DIR = 'jsons'

def parse_sec_str(t):
    """ Convert 'HH:MM:SS' or 'MM:SS' or 'SS' to seconds"""
    if t is None:
        return None
    t = t.strip()
    if t == '':
        return None

    parts = t.split(':')
    parts = [int(p) for p in parts]
    # print(parts)
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
                  # model_path:Path|str=None,
                  step=STEP, conf_thresh=DETECTION_THRESHOLD,  # if_usual=False, videos_folder='',
                  ann_file=None,
                  default_group_tag=None,
                  tension_intervals=[], fight_intervals=[], fall_intervals=[],
                  **kwargs):
    """
    Converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default tags.
    Process video file/dir of files.
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
    Convertion related parameters:
    :param step:                sampling rate (or fps)
    :param conf_thresh:         YOLO detection confidence threshold
    :param default_group_tag:
        - Assign default events to all frames by default
        - individual_events is kept in the output only for compatibility
    :param ann_file:
        Optional annotation text file with rows:
        start_time,\\tend_time,\\tevent_flag
        Supported event flags: 2 (fall), 3 (tension), 4 (fight)
        Precedence:
        defaults < ann_file intervals < explicit *_intervals arguments
        If ann_file is None, the code will look next to each video for:
        <video_stem>.txt, <video_stem>.ann, or <video_stem>
    """

    #* local helpers sub-fucntion
    def find_annotation_file(video_path):
        """Find a sibling annotation file matching the video stem."""
        video_path = Path(video_path)
        candidates = (video_path.with_suffix(".txt"),
                      video_path.with_suffix(".ann"),
                      video_path.with_suffix(""))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

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


    model_path = kwargs.get('model_path', None) or DEFAULT_YOLO
    model = YOLO(model_path if Path(model_path).is_file() else DEFAULT_YOLO)

    input_path = Path(input_path)
    if input_path.is_dir(): #* former video_path
        vid_list = [p for p in input_path.iterdir() if p.is_file()]
    else:
        vid_list = [input_path]

    output_path = Path(output_path) if output_path else None

    if output_path is None:
        json_dir = input_path if input_path.is_dir() else input_path.parent
        json_name = None
    elif output_path.suffix == '.json': # it's a file name
        json_dir = output_path.parent if output_path.parent.is_dir() else input_path
        json_name = output_path.stem
    elif output_path.is_dir():
        json_dir = output_path
        json_name = None
    else:
        pass  # ToDo: handel it
        json_dir = output_path
        json_name = None

    json_dir.mkdir(parents=True, exist_ok=True)

    detector_info = {'model':Path(model_path).stem, 'version':model.ckpt['version'], 'threshold':conf_thresh}
    # detector_info = {'model':Path(model_path).stem, 'threshold':conf_thresh}
    print_color(f"YOLO:\nversion - {detector_info['version']}\nthreshold = {detector_info['threshold']}", 'b')
    print(f"Default group event: {default_group_tag}\n")

    """ gree online should be ready earlier """
    # default_group_tag = collection(default_group_tag)  # []

    ann_intervals = parse_annotation_file(ann_file) if ann_file else None

    for vid_path in vid_list:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video (probably corrupted): {vid_path}")
            continue #return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w != 1920:
            h = int(w*1080/1920)
        # print(w, h)
        video_duration_sec = frame_count/fps
        print(f"*** Converting {vid_path.name},{(w, h)}p {video_duration_sec} s")

        # group_events = default_group_tag  # []

        video_ann_intervals = ann_intervals
        if video_ann_intervals is None:
            auto_ann_file = find_annotation_file(vid_path)
            if auto_ann_file is not None:
                print(f"Using annotation file: {auto_ann_file}")
                video_ann_intervals = parse_annotation_file(auto_ann_file)

        tension_intervals_sec = list(video_ann_intervals[TAG_TENSION]) if video_ann_intervals is not None else []
        fight_intervals_sec = list(video_ann_intervals[TAG_FIGHT]) if video_ann_intervals is not None else []
        fall_intervals_sec = list(video_ann_intervals[TAG_FALL]) if video_ann_intervals is not None else []

        tension_intervals_sec.extend(parse_interval(s) for s in tension_intervals if s and s.strip())
        fight_intervals_sec.extend(parse_interval(s) for s in fight_intervals if s and s.strip())
        fall_intervals_sec.extend(parse_interval(s) for s in fall_intervals if s and s.strip())
        # print(tension_intervals_sec)
        # print(fight_intervals_sec)
        # print(fall_intervals_sec)

        if fps <= 0:
            fps = 25.0  # fallback if metadata is broken

        frames = []
        frame_idx = 0
        #HEAD_IDX      = [0, 1, 2, 3, 4]  # первые 5: голова / лиц        #SHOULDER_IDX  = [5, 6]           # левое и правое плечо
        THRESH = 0.5
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (w, h))
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            # run model on current frame
            results = model(frame, conf=conf_thresh, verbose=False)[0]

            detection_list = []
            if results.boxes:
                for box, kpts_norm, conf_kpts in zip(results.boxes, results.keypoints.xyn, results.keypoints.conf):
                    cls_id = int(box.cls)
                    conf = float(box.conf)

                    kpts_xyc = torch.cat([kpts_norm, conf_kpts.unsqueeze(-1)], dim=-1)

                    if cls_id not in [3,4]:
                        x1, y1, x2, y2 = map(float, box.xyxyn[0])
                        for kp_pair in kpts_xyc:#[5:]
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

            time_sec = frame_idx/fps

            #* ---- Per-frame event logic ----
            #* All event types are emitted through group_events.
            #* individual_events stays empty for output compatibility.
            group_events = []

            # if in_any_interval(time_sec, fall_intervals_sec, video_duration_sec):
            if in_any_interval(time_sec, fall_intervals_sec):
                group_events.append(TAG_FALL)
            # if in_any_interval(time_sec, tension_intervals_sec, video_duration_sec):
            if in_any_interval(time_sec, tension_intervals_sec):
                group_events.append(TAG_TENSION)
                #print('3 added to events', event_grouped)
            # if in_any_interval(time_sec, fight_intervals_sec, video_duration_sec):
            if in_any_interval(time_sec, fight_intervals_sec):
                group_events.append(TAG_FIGHT)
                #print('4 added to events', event_grouped)

            default_tags = as_event_list(default_group_tag)
            group_tags = group_events if group_events else default_tags

            frames.append({'f': frame_idx, 't': time_sec,
                           'individual_events': [],
                            'group_events': sorted(set(group_tags), reverse=True),
                            'detection_list': detection_list,}
                          )

            if kwargs.get('show', False):
                cv2.imshow("head_center_debug", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC to quit
                break
            frame_idx += 1

        #cap.release()
        if frame_idx == 0:
            print(f"[WARN] Skipping video with no decodable frames: {vid_path} "
                  f"(unsupported codec/container or corrupted file)")
            cap.release()
            cv2.destroyAllWindows()
            continue

        data = {'video': str(vid_path),
                'fps': fps,
                'step': step,
                'detector': detector_info,
                'frames': frames,
                }

        #Todo: resolve case, when video_path is dir while out_json is a file name
        # Done !!! (01/03/26)
        json_path = get_unique_name(json_dir/f"{json_name if json_name else vid_path.stem}.json",4)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print(f"Saved::{len(frames)} frame to {json_path}\n -------------\n\n'")
        print_color(f"Saved::{len(frames)} frame to {json_path}\n----------------\n'",'b')

        cap.release()
        cv2.destroyAllWindows()

#195 -> 272 ->252->  220


#* ------  local runner (wrapper) -----------------------------
#* CLI interface moved to designated wrapper  convect_to_json.py
def local_runner(tst_path, **kwargs):

        process_video(tst_path, **kwargs)


def test_process_video():
    process_video(
        "/mnt/local-data/Projects/Wesmart/Video-datasets/test_ds/tst_conv",
        output_path="/mnt/local-data/Python/Projects/weSmart/data/json_files/tst_conv",
    )


if __name__ == "__main__":
    test_process_video()

#534(5,19,27) -> 333(5,7,6)
