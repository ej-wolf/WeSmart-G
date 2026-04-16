"""CLI wrapper for the timestamp-aware video-to-JSON converter."""

import argparse
from pathlib import Path

from video_to_json_bb_keypoints_folder_timestamp_truth_qa import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to per-frame JSON with QA metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Path to one video or a directory of videos")
    parser.add_argument("-m", "--model", type=Path, help="Path to a YOLO pose model")
    parser.add_argument("-o", "--out", type=Path, help="Output JSON file or output directory")
    parser.add_argument(
        "-s",
        "--sample-rate",
        type=float,
        help="Target saved frames per second (Hz), e.g. 5 keeps about 5 JSON frames/sec",
    )
    parser.add_argument("-c", "--conf", type=float, help="YOLO detection confidence threshold")
    parser.add_argument("-g", "--group-ann", type=int, nargs="+", help="Default group event tag(s)")
    parser.add_argument("-a", "--ann-file", type=Path, help="Optional shared annotation file")
    parser.add_argument(
        "-t",
        "--tension",
        action="append",
        help="Tension interval(s), e.g. 00:01:00-00:01:30, -00:00:40, 00:05:00-",
    )
    parser.add_argument(
        "-f",
        "--fight",
        action="append",
        help="Fight interval(s), e.g. 00:02:10-00:02:40, 00:03:00-",
    )
    parser.add_argument(
        "-fa",
        "--fall",
        action="append",
        help="Fall interval(s), e.g. 00:02:10-00:02:40, 00:03:00-",
    )
    parser.add_argument("--show", action="store_true", help="Show video during processing")
    parser.add_argument(
        "--time-source",
        choices=("fps", "ffprobe"),
        default="ffprobe",
        help="How to derive frame timestamps: metadata fps or ffprobe per-frame timestamps",
    )
    parser.add_argument(
        "--only-with-txt-ann",
        action="store_true",
        help="When VIDEO is a directory, process only videos that have a sibling <stem>.txt annotation",
    )
    parser.add_argument(
        "--name-from-video-path",
        action="store_true",
        help="Build JSON names from path parts after the last 'video'/'videos' folder",
    )

    args = parser.parse_args()

    process_kwargs = {
        "input_path": args.video,
        "output_path": args.out,
        "ann_file": args.ann_file,
        "default_group_tag": args.group_ann,
        "tension_intervals": args.tension,
        "fight_intervals": args.fight,
        "fall_intervals": args.fall,
        "model_path": args.model,
        "show": args.show,
        "time_source": args.time_source,
        "only_with_txt_ann": args.only_with_txt_ann,
        "name_from_video_path": args.name_from_video_path,
    }
    if args.sample_rate is not None:
        process_kwargs["sample_rate"] = args.sample_rate
    if args.conf is not None:
        process_kwargs["conf_thresh"] = args.conf

    process_video(**process_kwargs)


if __name__ == "__main__":
    main()
