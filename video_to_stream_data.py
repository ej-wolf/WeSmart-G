"""
    * The unit converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default tags.

    * The unit is wrapped into CLI in convert_to_json.py via process_video function.

    * JSON structure:
       {'video' : "...",                #* video/file name
        'fps': 25.0,                    #* video fps according to metadata
        'sampling_rate':{'target':,     #* desired/target for analyzing (Hz)
                         'effective':   #*
                        },
        'detector': {...}                 #* info abot the detector used to detect BB and KP
        'event_intervals': {...}          #*
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
import cv2, hashlib, json, platform, torch
from pathlib import Path
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz
from annotations import load_event_ann, resolve_event_time
#* import from my utils
from common.my_local_utils import get_unique_name, print_color, _zip_one_path

#* Defaults and constants  -------------------------------------------------------------------
YOLO_THRESHOLD = 0.5
DEFAULT_SAMPLING = 5
SPLIT_EPS = 1e-9

#* Events flags
TAG_NO_EVENT = 0
TAG_ABNORMAL = 1
TAG_FALL     = 2
TAG_TENSION  = 3
TAG_FIGHT    = 4
SPLIT_TAGS   = {'S', 's'}
EXCLUDE_TAGS = {'X', 'x'}
EVENT_INTERVAL_NAMES = {TAG_ABNORMAL: 'abnormal',
                        TAG_FALL: 'fall',
                        TAG_TENSION: 'tension',
                        TAG_FIGHT: 'fight'}

MODELS_DIR = 'models/'
DEFAULT_YOLO = MODELS_DIR + "yolo26x-pose.pt"
ZIP_JSONS = True
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mkv', '.mov', '.m4v'}
#* "yolo26x-pose.pt" / "yolo11x-pose.pt" / "yolov8s.pt"

def _is_video(f:str|Path):
    f = Path(f)
    return  f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES


def parse_annotation_file(ann_file):
    """ Load event annotations into converter intervals by numeric or special flag."""
    event_intervals = {}
    split_intervals = []
    exclude_intervals = []
    for event in load_event_ann(ann_file).get('events', []):
        event_flag = event.get('flag')
        interval = (resolve_event_time(event.get('start')),
                    resolve_event_time(event.get('end')))
        if isinstance(event_flag, str):
            if event_flag in SPLIT_TAGS:
                split_intervals.append(interval)
            elif event_flag in EXCLUDE_TAGS:
                exclude_intervals.append(interval)
            continue

        event_intervals.setdefault(int(event_flag), []).append(interval)

    return {'events': event_intervals, 'split': split_intervals, 'exclude': exclude_intervals}

#*  --- Main function, to be used from other unit ---------------------------------------
def process_video(input_videos: Path | str,
                  output_path:Path|str=None,
                  sample_rate=DEFAULT_SAMPLING,
                  yolo_threshold=YOLO_THRESHOLD,
                  ann_file:str|Path=None, default_grp_tag=None,
                  **kwargs):
    """ Converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default
    tags. Process video file/dir of files.
    Parameters:
    :param Path input_videos: file or dir
            - If file  → process single video
            - If dir   → process all files inside
    :param output_path : file, dir or None
        Single input:
            - None            → save next to video (same name, .json)
            - directory       → save inside directory with video name
            - file path       → save exactly to that file
    Conversion related parameters:
    :param sample_rate:        → Sampling rate for JSON conversion (in Hz)
    :param yolo_threshold:     → YOLO detection confidence threshold
    :param default_grp_tag:    → Assign default events to all frames by default
    :param ann_file:           → Optional annotation text file with rows:
                               start_time,\t  end_time,\t    event_flag
                               Supported event flags: numeric tags, S/s (split), X/x (exclude)
                               Precedence:
                               annotation file intervals override the default group tag
                               If ann_file is None, the code will look next to each video for:
                               <video_stem>.txt, <video_stem>.ann, or <video_stem>
    :param skip_without_ann:   → If True, skip videos that have no usable annotation file.
    :param use_old_meta:       → Optional kwargs flag. If True, write the old compact detector metadata.
    """

    #* local sub-function (helpers)
    def find_annotation_file(ann_path):
        """  Find a sibling annotation file matching the video stem."""
        ann_path = Path(ann_path)
        if ann_path.is_dir():
            ann_path = ann_path/vid_path.stem
        candidates = (ann_path.with_suffix(".ann"),
                      ann_path.with_suffix(".txt"),
                      ann_path.with_suffix(""))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def resolve_annotation_source(vid_path, ann_spec):
        """Resolve one annotation source for one video."""
        if ann_spec is None:
            return find_annotation_file(vid_path)
        ann_spec = Path(ann_spec)
        if ann_spec.is_file():
            return ann_spec
        if ann_spec.is_dir():
            return find_annotation_file(ann_spec/vid_path.stem)
        return find_annotation_file(ann_spec)

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

    #* normalize arguments
    input_videos = Path(input_videos)
    output_path = Path(output_path) if output_path else None
    model_path = kwargs.get('model_path', None) or DEFAULT_YOLO
    show = kwargs.get('show', False)
    use_legacy_meta = kwargs.get('use_legacy_meta', False)
    skip_without_ann = kwargs.get('skip_without_ann', False)
    zip_output = kwargs.get('zip_output', kwargs.get('zip', ZIP_JSONS))

    #* load model
    model = YOLO(model_path if Path(model_path).is_file() else DEFAULT_YOLO)

    #* handel and set input/output paths
    #* ToDo: the list option is not 100% debugged
    if isinstance(input_videos, (list, tuple, set,)):
        input_dir = None
        vid_list = [Path(p) for p in input_videos if _is_video(p)]
                    # if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if input_videos.is_dir():
        input_dir = input_videos
        vid_list = [p for p in sorted(input_videos.iterdir()) if _is_video(p)]
                    # if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    elif input_videos.is_file():
        input_dir = input_videos.parent
        vid_list = [input_videos]
    else:
        raise FileNotFoundError(f"Can't find input path: {input_videos}")
    if not vid_list:
        raise FileNotFoundError(f"No supported video files found in {input_videos}")

    if output_path is None:
        # json_dir = input_dir if input_videos.is_dir() else input_videos.parent
        json_dir = input_dir or "."
        json_name = None
    elif output_path.suffix == '.json':
        json_dir = output_path.parent if output_path.parent.is_dir() else input_dir
        json_name = output_path.stem
    elif output_path.is_dir() or output_path.parent.is_dir():
        json_dir = output_path
        json_name = None
    else:
        raise FileNotFoundError(f"Bad output path: {output_path}\nSee --help for further information")
    json_dir.mkdir(parents=True, exist_ok=True)

    #* detection info for header
    model_file = Path(model_path if Path(model_path).is_file() else DEFAULT_YOLO)
    checksum = hashlib.sha256()
    with model_file.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            checksum.update(chunk)


   #* detection info for header
    detector_info = {'model': Path(model_path).stem, 'version': model.ckpt['version']}
    if use_legacy_meta:
        detector_info['threshold'] =  yolo_threshold
    else:
        detector_info.update({'runtime_ultralytics': ultralytics.__version__,
                               'runtime_torch': torch.__version__,
                               'runtime_python': platform.python_version(),
                               'runtime_cuda': torch.version.cuda,
                               'cuda_available': torch.cuda.is_available(),
                               'device': ('cuda' if torch.cuda.is_available() else 'cpu'),
                               'model_sha256': checksum.hexdigest(),})

    print_color(f"YOLO:\nversion - {detector_info['version']}\nthreshold = {yolo_threshold}", 'b')
    print(f"Default group event: {default_grp_tag}\n")

    # ann_intervals = parse_annotation_file(ann_file) if ann_file else None
    for vid_path in vid_list:
        #* open video
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video (probably corrupted): {vid_path}")
            continue
        #* params from metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # * fallbacks for broken metadata
        if fps <= 0:
            fps = 25.0
        if w != 1920:
            h = int(w*1080/1920)

        # img_sz = (h, w)
        infer_img_sz = tuple(check_imgsz((h, w), stride=model.stride, min_dim=2))
        # if infer_img_sz != img_sz:
        #     print_color(f"Adjusted inference size from {img_sz} to {infer_img_sz} for stride compatibility", 'o')+
        video_duration_sec = frame_count/fps
        print(f"*** Converting {vid_path.name},{(w, h)}p {video_duration_sec} s")

        auto_ann_file = resolve_annotation_source(vid_path, ann_file)
        if auto_ann_file is not None:
            print_color(f"Using annotation file: {auto_ann_file}", 'g')
            video_ann = parse_annotation_file(auto_ann_file)
        elif skip_without_ann:
            print_color(f"[WARN] Skipping {vid_path.name}: no  annotation file found", 'y')
            continue
        else:
            print_color(f"[WARN] No annotation file for {vid_path.name}; using default group tags only", 'y')
            video_ann = {'events': {}, 'split': [], 'exclude': []}


        event_intervals_by_tag = dict(video_ann['events']) if video_ann is not None else {}
        split_intervals_sec = list(video_ann['split']) if video_ann is not None else []
        exclude_intervals_sec = list(video_ann['exclude']) if video_ann is not None else []
        split_intervals_sec = sorted(split_intervals_sec,
                                     key=lambda interval: (0.0 if interval[0] is None else interval[0],
                                                           video_duration_sec if interval[1] is None else interval[1]))

        target_sampling = sample_rate or  DEFAULT_SAMPLING
        if target_sampling <= 0 or target_sampling > fps :
        #* i.e    0 <= sampling_rate_Hz <= fps
            raise ValueError(f"Invalid sampling rate: {target_sampling} Hz")

        step = max(1, int(round(fps/target_sampling)))
        effective_sampling = fps/step
        print(f"effective sampling = {effective_sampling} Hz -> step = {step}")

        print(f"Sampling setup: Video fps={fps:.3f}, Sampling rate={target_sampling:.3f} Hz,"
              f" Step= {step} frames -> effective rate ={effective_sampling:.3f} Hz")

        frames, segments = [],  []
        split_idx, frame_idx = 0, 0
        skip_until_sec = None
        # dTHRESH = 0.5
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[INFO] cv2.VideoCapture.read()  failed at frame_idx = {frame_idx+1}")
                break

            frame = cv2.resize(frame, (w, h))
            #* frame miss
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            time_sec = frame_idx/fps
            while split_idx < len(split_intervals_sec):
                split_start, split_stop = split_intervals_sec[split_idx]
                split_start = 0.0 if split_start is None else split_start
                split_stop = video_duration_sec if split_stop is None else split_stop
                if time_sec <= split_start + SPLIT_EPS:
                    break
                if frames:
                    segments.append(frames)
                    frames = []
                skip_until_sec = max(skip_until_sec, split_stop) if skip_until_sec is not None else split_stop
                split_idx += 1

            if skip_until_sec is not None:
                if time_sec < skip_until_sec - SPLIT_EPS:
                    frame_idx += 1
                    continue
                skip_until_sec = None

            if in_any_interval(time_sec, exclude_intervals_sec):
                frame_idx += 1
                continue

            #* run model on current frame
            try:
                # results = model(frame, conf=yolo_threshold, verbose=False, imgsz=infer_img_sz)[0]
                results = model(frame, conf=yolo_threshold, verbose=False)[0]
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
                        detection_list.append({ 'class': cls_id, 'conf': conf, 'bbox': [x1, y1, x2, y2],
                                                'key_points': kpts_xyc[:].flatten().tolist(),
                                                })
                    else:
                        continue

            time_sec = frame_idx/fps

            #* ---- Per-frame event logic ----
            group_events = []
            for event_tag, event_intervals in event_intervals_by_tag.items():
                if in_any_interval(time_sec, event_intervals):
                    group_events.append(event_tag)

            default_tags = as_event_list(default_grp_tag)
            group_tags = group_events if group_events else default_tags

            frames.append({'f': frame_idx, 't': time_sec,
                           'individual_events': [],
                            'group_events': sorted(set(group_tags), reverse=True),
                            'detection_list': detection_list,})

            while split_idx < len(split_intervals_sec):
                split_start, split_stop = split_intervals_sec[split_idx]
                split_start = 0.0 if split_start is None else split_start
                split_stop = video_duration_sec if split_stop is None else split_stop
                if time_sec + SPLIT_EPS < split_start:
                    break
                if frames:
                    segments.append(frames)
                    frames = []
                skip_until_sec = max(skip_until_sec, split_stop) if skip_until_sec is not None else split_stop
                split_idx += 1

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

        if frames:
            segments.append(frames)
        if not segments:
            print(f"[WARN] Skipping video with no sampled frames after split/exclude filters: {vid_path}")
            cap.release()
            if show:
                cv2.destroyAllWindows()
            continue

        event_intervals = {}

        # event_intervals = {'tension': {'raw': tension_intervals, 'sec': tension_intervals_sec},
        #                    'fight': {'raw': fight_intervals,   'sec': fight_intervals_sec},
        #                    'fall': {'raw': fall_intervals,    'sec': fall_intervals_sec},
        #                    },

        for tag in sorted(set(EVENT_INTERVAL_NAMES) | set(event_intervals_by_tag)):
            tag_name = EVENT_INTERVAL_NAMES.get(tag, str(tag))
            event_intervals[tag_name] = {'sec': list(event_intervals_by_tag.get(tag, []))}

        event_intervals.update({'split': {'sec': split_intervals_sec},
                                'exclude': {'sec': exclude_intervals_sec}})

        for segment_frames in segments:
            data = {'video': str(vid_path),'fps': fps,
                    'sampling rate': {'target':target_sampling, 'effective':effective_sampling},
                    'step': step}
            if not use_legacy_meta:
                data['detection_threshold'] = yolo_threshold
            data.update({'detector': detector_info, 'event_intervals':event_intervals, 'frames': segment_frames,})

            #Todo: resolve case when video_path is dir while output_path is a file name
            json_path = get_unique_name(json_dir/f"{json_name if json_name else vid_path.stem}.json",4)

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if zip_output:
                archive_path = _zip_one_path(json_path, protocol=kwargs.get('zip_protocol','zip'))
                print_color(f"Archived to {archive_path}", 'b')

            print_color(f"Saved::{len(segment_frames)} frame to {json_path}\n----------------\n'",'b')

        cap.release()
        if show:
            cv2.destroyAllWindows()

    return True

#397->360(1,4,4) -> 424

if __name__ == "__main__":

    video_path = Path("/mnt/local-data/Projects/Wesmart/Video-datasets/draft_set/tst_conv")
    draft_path = Path( "/mnt/local-data/Python/Projects/weSmart/data/json_files/tst_conv/draft_dir")

    # process_video(video_path, output_path = draft_path/'tst-03', default_grp_tag= 0, )

    ubi_ann  = Path("data/video/UBI_FIGHTS/ann_ws_ready")
    ubi_vid  = Path("data/video/UBI_FIGHTS/videos/fight")
    ubi_vid  = Path("data/video/UBI_FIGHTS/videos/normal")

    json_dir = Path("data/json_files/UBI/3fps/normal")
    fps = 3; grp_tag = 0
    process_video(ubi_vid, json_dir, ann_file=None, default_grp_tag=grp_tag,
                  sample_rate=fps, zip_output=False)
