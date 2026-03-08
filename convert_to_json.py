""" CLI interface for video_to_json_bb_keypoints_folder
    Converts one or more videos into structured JSON dict.
    YOLO detector used for people detection and extracting
    Bound-boxes and key-points at Frame-level.
    If needed, time-based labeling at frame level can be preformed

    Usage:
    >> convert_to_json.py [-h] --video VIDEO [-m MODEL] [-o OUT] [-s STEP] [-c CONF] [-g GROUP_ANN [GROUP_ANN ...]]
                              [-i INDIV_ANN [INDIV_ANN ...]] [-t TENSION] [-f FIGHT] [-fa FALL]
    Options:
      -h, --help              show this help message and exit
      --video       VIDEO     Path to input video or dir of videos ( .mp4, .mkv, ...) (default: None)
      -m/--model    MODEL     Path to YOLO___.pt model (default: None)
      -o/--out      OUT       Path to output JSON file or output directory (default: None)
      -s/--step     STEP      sampling rate. (default: 5)
      -c/--conf     CONF      YOLO detection confidence threshold (default: 0.6)
      -t/--tension  TENSION   Tension interval(s) in format START-END, e.g. 00:01:00-00:01:30, -00:00:40, 00:05:00- (default:[])
      -f/--fight    FIGHT     Fight interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00- (default: [])
      -fa/--fall    FALL      Fall interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00- (default: [])
      -g/--group-ann    [GROUP_ANN ...]  Default annotation for group event (default: [])
      -i/--indiv-ann    [INDIV_ANN ...] Default annotation for individual event (default: [])
"""
import argparse
from pathlib import Path
from video_to_json_bb_keypoints_folder import process_video
from my_local_utils import print_color

def main():
    parser = argparse.ArgumentParser(description="Convert video to JSON data file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', type=Path, help="Path to input video or dir of videos ( .mp4, .mkv, ...)")
    # parser.add_argument('--model', type=Path, required=True, help="Path to YOLO___.pt model")
    # parser.add_argument('--out',   type=Path, required=True, help="JSON output name or path for output directory")
    parser.add_argument( '-m', '--model',type=Path, help="Path to YOLO___.pt model")
    parser.add_argument( '-o', '--out',  type=Path, help="Path to output JSON file or output directory")
    parser.add_argument( '-s', '--step', type=int, default=5, help="sampling rate. ")
    parser.add_argument( '-c', '--conf', type=float, default=0.6, help="YOLO detection confidence threshold")
    parser.add_argument( '-g', '--group-ann', type=int, nargs='+', default=[], help="Default annotation for group event")
    parser.add_argument( '-i', '--indiv-ann', type=int, nargs='+', default=[], help="Default annotation for individual event")
    parser.add_argument( '-t', '--tension', action="append", default=[], help="Tension interval(s) in format START-END, e.g. 00:01:00-00:01:30, -00:00:40, 00:05:00-")
    parser.add_argument( '-f', '--fight', action="append", default=[], help="Fight interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-")
    parser.add_argument( '-fa','--fall', action="append", default=[], help="Fall interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-")


    args = parser.parse_args()
    print_color(args.conf)

    process_video(input_path=args.video,
                  output_path=args.out,
                  # model_path=args.model,
                  step=args.step,
                  conf_thresh=args.conf,
                  default_group_tag= args.group_ann,
                  default_individual_tag=args.indiv_ann,
                  tension_intervals=args.tension,
                  fight_intervals=args.fight,
                  fall_intervals=args.fall,
                  model_path = args.model,
                  )

if __name__ == "__main__":
    main()
