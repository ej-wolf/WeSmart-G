#!/usr/bin/env python3
"""
Run YOLO on a video and output a simple JSON:

{
  "video": "...",
  "fps": 25.0,
  "step": 3,
  "frames": [
    {
      "f": 15,               # frame index
      "t": 0.6,              # time from start in seconds
      "event_single": 2,     # 0,1,2,5 or -1
      "event_grouped": 3,    # 3,4 or -1
      "bbs": [               # list of [cls, x1, y1, x2, y2]
        [2, x1, y1, x2, y2],
        [3, x1, y1, x2, y2]
      ]
    },
    ...
  ]
}
"""

import argparse, os
import json, numpy as np, torch
from pathlib import Path

import cv2
from ultralytics import YOLO

#* import from my utils
from my_local_utils import get_unique_name, collection

#* Events Thresholds
#* --------------------
SINGLE_THRESHOLDS = { 0: 0.5,   #* normal
                      1: 0.9,   #* abnormal
                      2: 0.7,   #* fall
                      5: 0.9,}  #* kick
GROUPED_THRESHOLDS = {3: 0.7,   #* tension
                      4: 0.7,}  #* violence
TAG_NO_EVENT = 0
TAG_FALL     = 2
TAG_TENSION  = 3
TAG_FIGHT    = 4



# DEFAULT_YOLO = "yolov8s.pt"
DEFAULT_YOLO = "yolo11x-pose.pt"

#* Helpers
#* --------------------
def choose_single_event(detections):
    """
    :param detections: list of (cls_id, conf)
    Return int single event  {0,1,2,5 or -1 if none}
    Logic:
      * consider only cls in SINGLE_THRESHOLDS
      * apply thresholds
      * pick class with max conf
      * if only norm (0) present -> return 0
      * if nothing passes -> -1
    """
    # filter by single classes and thresholds
    candidates = []
    for cls_id, conf in detections:
        if cls_id not in SINGLE_THRESHOLDS:
            continue
        thr = SINGLE_THRESHOLDS[cls_id]
        if conf >= thr:
            candidates.append((cls_id, conf))

    if not candidates:
        return -1  # no people or all below thresholds
    # if there is at least one non-norm candidate above threshold,
    # ignore pure norm case
    non_norm = [c for c in candidates if c[0] != 0]
    if non_norm:
        # choose best among non-norm
        return max(non_norm, key=lambda x: x[1])[0]

    # only norm candidates
    # choose best norm
    return max(candidates, key=lambda x: x[1])[0]  # will be 0


def choose_grouped_event(detections):
    """
    :param detections: list of (cls_id, conf)
    Return int grouped event: (3,4 or -1)
    """
    candidates = []
    for cls_id, conf in detections:
        if cls_id not in GROUPED_THRESHOLDS:
            continue
        thr = GROUPED_THRESHOLDS[cls_id]
        if conf >= thr:
            candidates.append((cls_id, conf))

    if not candidates:
        return -1
    return max(candidates, key=lambda x: x[1])[0]

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
        return (None, None)

    s = s.strip()
    if s == "":
        # Completely empty → ignore this interval later
        return (None, None)

    # No '-' → interpret as single time = from that time until end
    if "-" not in s:
        start = parse_sec_str(s)
        end = None
        return start, end

    start_str, end_str = s.split("-", 1)

    start = parse_sec_str(start_str) if start_str.strip() != "" else None
    end   = parse_sec_str(end_str)   if end_str.strip()   != "" else None

    return start, end


def in_any_interval(time_sec, intervals, video_duration_sec):
    for start, stop in intervals:
        # (None, None) means "ignore" → skip
        if start is None and stop is None:
            continue

        if start is None:
            start = 0.0
        if stop is None:
            stop = video_duration_sec

        if start <= time_sec <= stop:
            #print('True')
            return True
    return False


#* Main function, to be used from other unit
#* --------------------
def process_video(input_path: Path | str,
                  model_path:Path|str=None,
                  out_json  :Path|str=None,
                  step=5, conf_thresh=0.5,  # if_usual=False, videos_folder='',
                  default_group_tag=[], default_individual_tag=[],
                  tension_intervals=[], fight_intervals=[], fall_intervals=[],
                  **kwargs):
    """
    Process one video file or all video files inside a directory.

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
    """


    DEFAULT_JSON_DIR = 'jsons'
    # paths_list = [] #* former video_paths

    if model_path and Path(model_path).is_file():
        model = YOLO(str(model_path))
    else:
        print("Using default YOLO model\n")
        model = YOLO(DEFAULT_YOLO)

    # if not if_usual:
    #     video_paths.append(video_path)
    # else:
    #     for entry in os.listdir(videos_folder):
    #         full_path = os.path.join(videos_folder, entry)
    #         if os.path.isfile(full_path):
    #             video_paths.append(full_path)

    input_path = Path(input_path)
    if input_path.is_dir(): #* former video_path
        vid_list = [p for p in input_path.iterdir() if p.is_file()]
    else:
        vid_list = [input_path]

    out_json = Path(out_json) if out_json else None

    if out_json is None:
        json_dir = input_path if input_path.is_dir() else input_path.parent
        json_name = None
    elif out_json.suffix == '.json': # it's a file name
        json_dir = input_path.parent
        json_name = out_json.name
    elif out_json.is_dir():
        json_dir = out_json
        json_name = None
    else:
        pass  # ToDo: handel it
        json_dir = out_json
        json_name = None

    json_dir.mkdir(parents=True, exist_ok=True)

    """ gree nline should be ready earlier """
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
        #
        # print(fall_intervals_sec)

        if fps <= 0:
            fps = 25.0  # fallback if metadata is broken

        frames = []
        #detections = []
        frame_idx = 0v
        #HEAD_IDX      = [0, 1, 2, 3, 4]  # первые 5: голова / лиц        #SHOULDER_IDX  = [5, 6]           # левое и правое плечо
        THRESH = 0.5
        while True:
            ret, frame = cap.read()
            # if  w!= 1920:
            #     h = int(w*1080/1920)
            if not ret:
                break
            frame = cv2.resize(frame, (w, h))
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            # run model on current frame
            results = model(frame, conf=conf_thresh, verbose=False)[0]
            #print(results.keypoints)
            #heads = []

            # detections = []      # for group event selection: list of (cls, conf)
            # bbs_keypoints= []    # for JSON output: [cls, conf, x1, y1, x2, y2]
            # group_events = []
            # individual_events = []]

            #print(results.boxes, results.keypoints.xyn, results.keypoints.conf)
            detection_list = []
            if results.boxes:
                for box, kpts_norm, conf_kpts in zip(results.boxes, results.keypoints.xyn, results.keypoints.conf):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    #if conf < conf_thresh:
                        #continue

                    kpts_xyc = torch.cat([kpts_norm, conf_kpts.unsqueeze(-1)], dim=-1)
                    #detections.append((cls_id, conf))
                    if cls_id not in [3,4]:
                        x1, y1, x2, y2 = map(float, box.xyxyn[0])
                        #print(type(kpts_norm))
                        #head_xy   = kpts_norm[HEAD_IDX]
                        # (5, 2)
                        #head_conf = conf_kpts[HEAD_IDX]
                        #print(head_xy.flatten().shape, head_conf.shape)
                        #mask = head_conf > THRESH
                        #head_xy_valid = head_xy[mask]
                        #if head_xy_valid.numel() > 0:  # there is at least one good keypoint
                        # 4. average → one point [x_mean, y_mean]
                            #head_center = head_xy_valid.mean(dim=0).tolist()   # tensor of shape (2,)
                        #else:
                        #shoulder_xy   = kpts_norm[SHOULDER_IDX]     # (2, 2)
                        #shoulder_conf = conf_kpts[SHOULDER_IDX]
                        #mask_sh = shoulder_conf > THRESH
                        #shoulder_xy_valid = shoulder_xy[mask_sh]
                        #if shoulder_xy_valid.numel() > 0:
                            #head_center = shoulder_xy_valid.mean(dim=0).tolist()
                        #else:
                            #head_center = [0.0, 0.0]

                        #cx = int(float(head_center[0]) * w)
                        #cy = int(float(head_center[1]) * h)
                        #cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                        for keyp_pairs in kpts_xyc:#[5:]
                            cx = int(float(keyp_pairs[0])*w)
                            cy = int(float(keyp_pairs[1])*h)
                            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                        #kpts_xyc = kpts_xyc.reshape(*kpts_xyc.shape[:-1], 3)
                        #print(kpts_xyc.shape)
                        #bbs_keypoints.append([cls_id, conf, x1, y1, x2, y2] + [kpts_xyc.tolist()])#, kpts_xyc[:].flatten().tolist()]), # head_center + kpts_norm[5:].flatten().tolist()])
                        #bbs_keypoints.append([kpts_xyc.tolist()])
                        detection_list.append({ 'class': cls_id,
                                                'conf': conf,
                                                'bbox': [x1, y1, x2, y2],
                                                'key_points': kpts_xyc[:].flatten().tolist(),
                                                })
                    else:
                        continue
                    #print(bbs_keypoints)
            #event_single = choose_single_event(detections)
            #event_grouped = choose_grouped_event(detections)
            # group_events = collection(default_group_tag) # []
            # individual_events = collection(default_individual_tag)
            # print(f"Default group event: {group_events}\n"
            #       f"Default individual event: {group_events}\n")

            time_sec = frame_idx / fps
            if in_any_interval(time_sec, fall_intervals_sec, video_duration_sec):
                individual_events.append(TAG_FALL)
                print(fall_intervals_sec)
            if in_any_interval(time_sec, tension_intervals_sec, video_duration_sec):
                group_events.append(TAG_TENSION)
                print(tension_intervals_sec)
                #print('3 added to events', event_grouped)
            if in_any_interval(time_sec, fight_intervals_sec, video_duration_sec):
                group_events.append(TAG_FIGHT)
                print(fight_intervals_sec)
                #print('4 added to events', event_grouped)

            '''if not tension_intervals and not fight_intervals:
                event_grouped = []
            
            else:
                for interval in tension_intervals:
                    start, stop = parse_sec_str(interval)
                    if not start: 
                        start = 0
                    if not stop:
                        stop = frame_count/fps
                    if start<time_sec<stop:
                        if 3 not in event_grouped:
                            event_grouped.append(3)
                    else:
                        if 3 in event_grouped:
                            event_grouped.remove(3)
                for interval in fight_intervals:
                    start, stop = parse_sec_str(interval)
                    if not start: 
                        start = 0
                    if not stop:
                        stop = frame_count/fps
                    if start<time_sec<stop:
                        if 4 not in event_grouped:
                            event_grouped.append(4)
                    else:
                        if 4 in event_grouped:
                            event_grouped.remove(4)'''

            frames.append({'f': frame_idx, 't': time_sec,
                           'individual_events': individual_events,
                            'group_events': sorted(group_events, reverse=True),
                            'detection_list': detection_list,
                          })
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

        # if len(paths_list) > 1:
        #     out_json_name = os.path.splitext(os.path.basename(video_path))[0] + '.json'
        #     # out_json = os.path.join(videos_folder, 'out_jsons', out_json_name)
        #     out_json = Path(out_json)
        # else:
        #     out_json = Path(out_json)
        #
        # if out_json:
        #     out_json_name = os.path.splitext(os.path.basename(video_path))[0] + '.json'
        #     out_json = os.path.join(videos_folder, 'out_jsons', out_json_name)
        #     out_json = Path(out_json)
        # else:
        #     out_json = Path(out_json)
        #
        # if out_json.is_dir():
        #     json_path = out_json/vid_path.with_suffix('.json')
        # elif vid_list == 1 :
        #     json_path = out_json
        # else:
        #     pass
        #Todo: resolve case, when video_path is dir while out_json is a file name

        # json_path = get_unique_name(json_dir/(json_name if json_name else vid_path.stem)/'.json')
        json_path = get_unique_name(json_dir/f"{json_name if json_name else vid_path.stem}.json")


        # json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved::{len(frames)} frame to {json_path}\n -------------\n\)

        cap.release()
        cv2.destroyAllWindows()
#195

# def main():
#     parser = argparse.ArgumentParser(description="Simple events JSON from video")
#     parser.add_argument("--video", type=Path, required=True, help="Input video path (.mp4, .mkv, ...)")
#     parser.add_argument("--model", type=Path, required=True, help="YOLO .pt model path")
#     parser.add_argument("--out", type=Path, required=True, help="Output one JSON file path with name, in case of processing the whole folder, jsons are saved inside this folder in the folder 'out_jsons', ")
#     parser.add_argument("--step", type=int, default=5, help="Process every Nth frame (3 or 5, etc.)")
#     parser.add_argument("--conf", type=float, default=0.6, help="Global detection confidence threshold")
#     parser.add_argument("--if_usual", type=bool, default=False, help="True if there is a folder with usual life")
#     parser.add_argument("--videos_folder", type=Path, help="folder of usual life")
#     #parser.add_argument("--out_jsons_folder", type=Path, help="folder of out jsons from folder of usual videos")
#     parser.add_argument(
#         "--tension", "-t",
#         action="append",
#         default=[],
#         help="Tension interval(s) in format START-END, "
#              "e.g. 00:01:00-00:01:30, -00:00:40, 00:05:00-"
#     )
#
#     parser.add_argument(
#         "--fight", "-f",
#         action="append",
#         default=[],
#         help="Fight interval(s) in format START-END, "
#              "e.g. 00:02:10-00:02:40, 00:03:00-"
#     )
#     parser.add_argument(
#         "--fall", "-fa",
#         action="append",
#         default=[],
#         help="Fall interval(s) in format START-END, "
#              "e.g. 00:02:10-00:02:40, 00:03:00-"
#     )
#     args = parser.parse_args()
#
#     process_video(
#         video_path=args.video,
#         model_path=args.model,
#         out_json=args.out,
#         step=args.step,
#         conf_thresh=args.conf,
#         if_usual=args.if_usual,
#         videos_folder=args.videos_folder,
#         #out_jsons_folder=args.out_jsons_folder,
#         tension_intervals=args.tension,
#         fight_intervals=args.fight,
#         fall_intervals=args.fall
#     )

#* Unit testing
#* --------------------
def local_runner(tst_path, **kwargs):

            process_video(tst_path, **kwargs)

if __name__ == "__main__":
    pass
    local_runner("/mnt/local-data/Projects/Wesmart/datasets/RWF-2000/train/Train_Fight",
                 out_json = "data/json_files/RWF-2000/train_pos/",
                 conf_thresh=0.4,default_group_tag=TAG_FIGHT )
    local_runner("/mnt/local-data/Projects/Wesmart/datasets/RWF-2000/train/Train_NonFight",
                 out_json = "data/json_files/RWF-2000/train_neg/", conf_thresh=0.4, default_group_tag=TAG_NO_EVENT)

