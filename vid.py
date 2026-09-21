""" CLI script.
    *** play ***
    Play one or more videos with optional synchronized annotation overlays.
    usage:
    >> vid.py play video_path [annotation_path] [-sd SPEED] [-s START]
                              [-o {name,date,size}] [-r] [-z SIZE] [--hold]
    * positional arguments:
      video_path                    : video, directory, or playlist file
      annotation_path               : optional annotation file or directory
    * options:
      -h/--help                    : show help message and exit
      -sd/--speed                  : playback speed factor
      -s/--start                   : start time in seconds
      -o/--order                   : playback order: name, date, or size
      -r/--reverse                 : reverse playback order
      -z/--size                    : window size: org, max, or a scale factor
      --hold                       : keep the player open until Esc is pressed

    *** fps ***
    Measure video FPS and save a report, or load and print an existing JSON report.
    usage:
    >> vid.py fps input_path [-o OUTPUT] [-th MF_THRESHOLD] [-n]
                      [-s SORT] [-or ORDER] [-r ROWS] [-t] [-c WIDTH]
    * positional arguments:
      input_path                   : one video, directory, or JSON report
    * measurement options:
      -o/--output                  : report output file or directory
      -th/--mf-threshold           : meaningful-frame motion threshold
      -n/--no-recursive            : do not scan subdirectories
    * report options:
      -s/--sort                   : video, encoded, measured, or ratio
      -or/--order                 : ascending or descending
      -r/--rows                   : maximum number of video rows to display
      -t/--total-only             : print summary statistics only
      -c/--col-width              : width mode, number, or comma-separated widths
"""
import argparse
import math
from pathlib import Path


def _warn(message):
    from common.my_local_utils import cli_warning
    cli_warning(message)

def _non_negative(value):
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not a number') from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('number must be finite and non-negative')
    return number


def _positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('expected a positive integer') from None
    if number <= 0:
        raise argparse.ArgumentTypeError('expected a positive integer')
    return number


def _column_width(value):
    if value in ('auto', 'data-opt', 'smart-opt'):
        return value
    widths = [_non_negative(part) for part in value.split(',')]
    return widths if ',' in value else widths[0]


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    play = commands.add_parser('play', help='Play videos with annotation overlays',
                                             description='Play video(s) with synced annotation event overlays.')
    play.add_argument('video_path', type=Path, help='video, directory, or playlist file')
    play.add_argument('annotation_path', type=Path, nargs='?', default=None, help='annotation file or dir; defaults to sibling annotations')
    play.add_argument('-sd', '--speed', type=float, default=1.0, help='playback speed factor')
    play.add_argument('-s', '--start', type=float, default=0.0, help='start time in seconds')
    play.add_argument('-o', '--order', choices=('name', 'date', 'size'), default=None, help='optional playback order')
    play.add_argument('-r', '--reverse', action='store_true', help='reverse playback order')
    play.add_argument('-z', '--size', default='org', help="window size: 'org', 'max', or a scale factor")
    play.add_argument('--hold', action='store_true', help='keep the player open until Esc is pressed')

    fps = commands.add_parser('fps', help='Measure video FPS or print a saved JSON report',
                                           description='Measure and save FPS for a video/directory, or print an existing JSON report.')
    fps.add_argument('input_path', type=Path, help='one video, directory, or JSON report')
    fps.add_argument('-o',  '--output', type=Path, help='file/directory for results report; default: measured_fps.json')
    fps.add_argument('-th', '--mf-threshold', type=_non_negative, help='measurement motion threshold; default: existing measurement API default')
    fps.add_argument('-n',  '--no-recursive', action='store_true', help='measure only videos directly in the input directory')
    fps.add_argument('-s',  '--sort',  choices=('video', 'encoded', 'measured', 'ratio', 'vid', 'enc', 'msr'), help='report sort column; default: video')
    fps.add_argument('-or', '--order', choices=('ascending', 'descending', 'asc', 'dsc'), default='ascending', help='report sort order')
    fps.add_argument('-r',  '--rows', type=_positive_int, help='maximum number of video rows to display')
    fps.add_argument('-t',  '--total-only', action='store_true', help='print summary statistics only')
    fps.add_argument('-c',  '--col-width', type=_column_width, default='smart-opt', metavar='WIDTH',
                                         help='auto, data-opt, smart-opt, a number, or comma-separated column widths')
    return parser


def _run_play(args):
    from video_inspector import play_multi_vid
    play_multi_vid(args.video_path, args.annotation_path,
                   speed=args.speed, start_sec=args.start, hold_on_end=args.hold,
                   order=args.order, reverse=args.reverse, size=args.size)


def _run_fps(args):
    from video_utils import VIDEO_SUFFIXES, get_measured_fps, load_fps_report, print_fps_report

    source = args.input_path
    if not source.exists():
        raise ValueError(f'input does not exist: {source}')
    if source.is_file() and source.suffix.lower() == '.json':
        if args.output is not None or args.mf_threshold is not None or args.no_recursive:
            raise ValueError('--output, --mf-threshold, and --no-recursive apply only to measurement inputs')
        report = load_fps_report(source)
    elif source.is_dir() or (source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES):
        measurement = dict(recursive=not args.no_recursive,  print_res=False,
                           save_to=args.output if args.output is not None else True,)
        if args.mf_threshold is not None:
            measurement['mf_threshold'] = args.mf_threshold
        _, report = get_measured_fps(source, **measurement)
    else:
        raise ValueError(f'unsupported input: {source}; expected a video, directory, or JSON report')
    print_fps_report(report, sort=args.sort, order=args.order, rows=args.rows,
                     total_only=args.total_only, col_width=args.col_width)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'play':
            _run_play(args)
        elif args.command == 'fps':
            _run_fps(args)
        else:
            _warn(f'unknown command: {args.command}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        _warn(f'{args.command} failed: {type(error).__name__}: {error}')

#101(1,,1)
if __name__ == '__main__':
    main()
