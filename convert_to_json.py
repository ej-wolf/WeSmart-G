""" CLI interface for video_to_json_bb_keypoints_folder


    usage
    >> precompute_clips  jsons_dir cache_dir [-h] [-s SPLIT_DIR] [-e]
                        [-rs RANDOM_SEED] [-ns] [-r VALID_RATIO]
    positional arguments:
      jsons_dir                     : dir containing JSONs
      cache_dir                     : path for the cached NPZ feature files
    options:
      -h, --help                    : Show help message and exit
      -s/--split-dir [path]         : Path for the train/val list files
      -e/--allow-empty              : Allow empty (None) labeling (default: True)
      -rs/--random-seed RANDOM_SEED : set the random seed (42)
      -ns/--new-split               : Force New split (default: False)
      -r/--valid-ratio VALID_RATIO  : Validation split ratio (default: 0.2)


"""
import argparse
from pathlib import Path
from video_to_json_bb_keypoints_folder import process_video

def main():
    parser = argparse.ArgumentParser(description="Simple events JSON from video")
    parser.add_argument('--video', type=Path, required=True, help="Path to input video or dir of videos ( .mp4, .mkv, ...)")
    # parser.add_argument('--model', type=Path, required=True, help="Path to YOLO___.pt model")
    # parser.add_argument('--out',   type=Path, required=True, help="JSON output name or path for output directory")
    parser.add_argument( '-m', '--model',type=Path, help="Path to YOLO___.pt model")
    parser.add_argument( '-o', '--out',  type=Path, help="JSON output name or path for output directory")
    parser.add_argument( '-s', '--step', type=int, default=5, help="sampling rate. ")
    parser.add_argument( '-c', '--conf', type=float, default=0.6, help="YOLO detection confidence threshold")
    # parser.add_argument("--out_jsons_folder", type=Path, help="folder of out jsons from folder of usual videos")
    parser.add_argument( '-g', '--group-ann'     , type=int|list, default=[], help="default annotation for group")
    parser.add_argument( '-i', '--individual-ann', type=int|list, default=[], help="default annotation for individuala")
    parser.add_argument( '-t', '--tension', action="append", default=[],
                                                help="Tension interval(s) in format START-END, e.g. 00:01:00-00:01:30, -00:00:40, 00:05:00-")
    parser.add_argument( '-f', '--fight', action="append", default=[],
                                                help="Fight interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-")
    parser.add_argument( '-fa', '--fall',  action="append", default=[],
                                                help="Fall interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-")
    # parser.add_argument("--if_usual", type=bool, default=False, help="True if there is a folder with usual life")
    # parser.add_argument("--videos_folder", type=Path, help="folder of usual life")

    args = parser.parse_args()
    process_video(video_path=args.video,
                  model_path=args.model,
                  out_json=args.out,
                  step=args.step,
                  conf_thresh=args.conf,
                  # if_usual=args.if_usual,
                  # videos_folder=args.videos_folder,
                  # out_jsons_folder=args.out_jsons_folder,
                  default_group__tag= args.group_ann,
                  default_individual_tag=args.individual_ann,
                  tension_intervals=args.tension,
                  fight_intervals=args.fight,
                  fall_intervals=args.fall
                  )


if __name__ == "__main__":
    main()
