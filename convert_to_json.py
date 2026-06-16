""" CLI interface for video_to_stream_data
    Converts one or more videos into structured JSON dict.
    YOLO detector used for people detection and extracting
    bound-boxes and key-points at frame level.
    If needed, time-based labeling at frame level can be performed

    Usage:
    >> convert_to_json.py [-h] VIDEO [-m MODEL] [-o OUT] [-s SAMPLE_RATE] [-c CONF]
                              [-g GROUP_ANN [GROUP_ANN ...]] [-a ANN_FILE]
    Options:
      -h, --help                  show this help message and exit
      VIDEO                       Path to input video or dir of videos (.mp4, .mkv, ...)
      -m/--model MODEL            Path to YOLO___.pt model
      -o/--out OUT                Path to output JSON file or output directory
      -s/--sample-rate RATE       Sampling rate in Hz
      -c/--conf CONF              YOLO detection confidence threshold
      -g/--group-ann [GROUP_ANN ...]  Default annotation for group event
      -a/--ann-file ANN_FILE      Optional annotation text file
"""
import argparse
from pathlib import Path
from video_to_stream_data import process_video
from common.my_local_utils import print_color

def main():
    parser = argparse.ArgumentParser(description="Convert video to JSON data file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', type=Path, help="Path to input video or dir of videos ( .mp4, .mkv, ...)")
    parser.add_argument( '-m', '--model',type=Path, help="Path to YOLO___.pt model")
    parser.add_argument( '-o', '--out',  type=Path, help="Path to output JSON file or output directory")
    parser.add_argument( '-s', '--sample-rate', type=float, help="Sampling rate in Hz")
    parser.add_argument( '-c', '--conf', type=float, help="YOLO detection confidence threshold")
    parser.add_argument( '-g', '--group-ann', type=int, nargs='+', help="Default annotation for group event")
    parser.add_argument( '-a', '--ann-file', type=Path, help="Optional annotation text file")
    parser.add_argument( '-sw', '--show', action='store_true', help='Show video during processing')
    parser.add_argument( '-z', '--zip', action='store_true', help='save JSONs as zip file')

    args = parser.parse_args()
    if args.conf is not None:
        print_color(args.conf)

    process_kwargs = dict(input_path=args.video,
                          output_path=args.out,
                          ann_file=args.ann_file,
                          default_grp_tag=args.group_ann,
                          model_path=args.model,
                          zip_output=args.zip,)
                          # show=args.show)
    if args.sample_rate is not None:
        process_kwargs['sample_rate'] = args.sample_rate
    if args.conf is not None:
        process_kwargs['yolo_threshold'] = args.conf
    if args.show is not None:
        process_kwargs['show'] = args.show

    process_video(**process_kwargs)

if __name__ == "__main__":
    main()
