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

#* Events Thresholds  -------------------------------------------------------------------
SINGLE_THRESHOLDS = { 0: 0.5,   #* normal
                      1: 0.9,   #* abnormal
                      2: 0.7,   #* fall
                      5: 0.9,}  #* kick
GROUPED_THRESHOLDS = {3: 0.7,   #* tension
                      4: 0.7,}  #* violence
#* Events flags
TAG_NO_EVENT = 0
TAG_FALL     = 2
TAG_TENSION  = 3
TAG_FIGHT    = 4

DEFAULT_YOLO = "yolo11x-pose.pt" # "yolov8s.pt"
DEFAULT_JSON_DIR = 'jsons'
def parse_sec_str(t):
    """ Convert 'HH:MM:SS' or 'MM:SS' or 'SS' to seconds"""
    if t is None:
        return None
    t = t.strip()
    if t == "":
        return None

    parts = t.split(":")
    parts = [int(p) for p in parts]
    #print(parts)
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:  # seconds only
        h = 0
        m = 0
        s = parts[0]
    return h * 3600 + m * 60 + s


def parse_interval(s):
    """
    Parses 'start-end' where start or end can be empty.
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

    start = parse_sec_str(start_str) if start_str.strip() != "" else None
    end   = parse_sec_str(end_str)   if end_str.strip()   != "" else None

    return start, end


#*  --- Main function, to be used from other unit ---------------------------------------
def process_video(input_path: Path | str,
                  model_path:Path|str=None,
                  output_path: Path | str=None,
                  step=5, conf_thresh=0.6,  # if_usual=False, videos_folder='',
                  default_group_tag=None, default_individual_tag=None,
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
    :param  default_group_tag, default_individual_tag:
        - Assign default group/individual event to all frames by default
        - Interval-based tags (tension/fight/fall) override them
    """
    print(f"{input_path}: {input_path.is_file()}")
    # print_color(default_group_tag)
    #* local helpers sub-fucntion
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


    if model_path and Path(model_path).is_file():
        model = YOLO(str(model_path))
    else:
        print("Using default YOLO model\n")
        model = YOLO(DEFAULT_YOLO)

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

    """ gree online should be ready earlier """
    default_group_tag = collection(default_group_tag)  # []
    default_individual_tag = collection(default_individual_tag)

    for vid_path in vid_list:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video (probably corrupted): {vid_path}")
            continue #return
        # '''if not cap.isOpened():
        #     raise RuntimeError(f"Cannot open video {video_path}")'''

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w != 1920:
            h = int(w*1080/1920)
        # print(w, h)
        video_duration_sec = frame_count/fps
        print(f"*** Converting {vid_path.name},{(w, h)}p {video_duration_sec} s")

        group_events = default_group_tag  # []
        individual_events = default_individual_tag
        print(f"Default group event: {default_group_tag}\n"
              f"Default individual event: {default_individual_tag}\n")

        tension_intervals_sec = [parse_interval(s) for s in tension_intervals if s and s.strip()]
        fight_intervals_sec   = [parse_interval(s) for s in fight_intervals   if s and s.strip()]
        fall_intervals_sec    = [parse_interval(s) for s in fall_intervals    if s and s.strip()]
        # print(tension_intervals_sec)
        # print(fight_intervals_sec)
        # print(fall_intervals_sec)

        if fps <= 0:
            fps = 25.0  # fallback if metadata is broken

        frames = []
        #detections = [
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
                    #if conf < conf_thresh:
                    #continue

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
            #* Set defaults
            # group_default = list(default_group_tag)
            # individual_default = list(default_individual_tag)
            # Interval overrides (cumulative between intervals,
            # but override defaults if any interval is active)
            group_events = []
            individual_events = []

            # if in_any_interval(time_sec, fall_intervals_sec, video_duration_sec):
            if in_any_interval(time_sec, fall_intervals_sec):
                individual_events.append(TAG_FALL)
                print(fall_intervals_sec)
            # if in_any_interval(time_sec, tension_intervals_sec, video_duration_sec):
            if in_any_interval(time_sec, tension_intervals_sec):
                group_events.append(TAG_TENSION)
                print(tension_intervals_sec)
                #print('3 added to events', event_grouped)
            # if in_any_interval(time_sec, fight_intervals_sec, video_duration_sec):
            if in_any_interval(time_sec, fight_intervals_sec):
                group_events.append(TAG_FIGHT)
                print(fight_intervals_sec)
                #print('4 added to events', event_grouped)

            group_tags = group_events if group_events else  list(default_group_tag)
            individual_tags = individual_events if individual_events else list(default_individual_tag)

            frames.append({'f': frame_idx, 't': time_sec,
                           'individual_events': individual_tags,
                            'group_events': sorted(group_tags, reverse=True),
                            'detection_list': detection_list,}
                          )
            #print(frame_idx, frame.shape)
            if kwargs.get('show', False):
                cv2.imshow("head_center_debug", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC to quit
                break
            frame_idx += 1

        #cap.release()

        data = {'video': str(vid_path),
                'fps': fps,
                'step': step,
                'frames': frames
                }

        #Todo: resolve case, when video_path is dir while out_json is a file name
        # Done !!! (01/03/26)


        json_path = get_unique_name(json_dir/f"{json_name if json_name else vid_path.stem}.json")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved::{len(frames)} frame to {json_path}\n -------------\n\n'")

        cap.release()
        cv2.destroyAllWindows()
#195 -> 272 ->252->  220


#* ------  local runner (wrapper) -----------------------------
#* CLI interface moved to designated wrapper  convect_to_json.py
def local_runner(tst_path, **kwargs):

        process_video(tst_path, **kwargs)

if __name__ == "__main__":
    pass
    local_runner("/mnt/local-data/Projects/Wesmart/datasets/RWF-2000/train/Train_Fight",
                 out_json = "data/json_files/RWF-2000/train_pos/",
                 conf_thresh=0.4,default_group_tag=TAG_FIGHT )
    local_runner("/mnt/local-data/Projects/Wesmart/datasets/RWF-2000/train/Train_NonFight",
                 out_json = "data/json_files/RWF-2000/train_neg/", conf_thresh=0.4, default_group_tag=TAG_NO_EVENT)

#534(5,19,27) -> 333(5,7,6)
