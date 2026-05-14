""" CLI script.
    *** train ***
    Train one clip-level MLP classifier from cached NPZ features.
    usage:
    >> tms_trainer.py train train_cache [-h] [-v VALID_CACHE] [-t TAG] [-wd WORK_DIR]
                          [-lr LR] [-e EPOCHS] [-bs BATCH_SIZE] [-hd HIDDEN_DIM]
                          [-sr SPLIT_RATIO] [-rs RANDOM_SEED]
    * positional arguments:
      train_cache                   : train cache npz path
    * options:
      -h/ --help                    : Show help message and exit
      -v/ --valid-cache             : optional validation cache npz path
      -t/ --tag                     : run tag suffix
      -wd/--work-dir                : output run directory root
      -lr                           : learning rate
      -e/ --epochs                  : number of epochs
      -bs/--batch-size              : batch size
      -hd/--hidden-dim              : hidden layer size
      -sr/--split-ratio             : runtime train split ratio
      -rs/--random-seed             : runtime split seed

    *** test ***
    Run inference on one cached NPZ and optionally evaluate the raw results.
    By default, testing saves one unified raw NPZ that can later be analyzed
    at clip / video / stream level. Use `--pure-clips` for one minimal raw NPZ.
    usage:
    >> tms_trainer.py test test_model test_cache [-h] [-bs BATCH_SIZE] [-od OUT_DIR]
                         [-t OUTPUT_TAG] [-ec | -ev | -es] [--pure-clips]
                         [-td THRESHOLD] [-nj] [-ns] [--no-roc-csv] [--no-print]
    * positional arguments:
      test_model                    : model checkpoint path
      test_cache                    : test cache npz path
    * options:
      --pure-clips                  : save minimal clip-only raw NPZ and skip evaluation
    * evaluation options:
      -ec/--eval-clip               : run clip evaluation after saving raw predictions
      -ev/--eval-video              : run video evaluation after saving raw predictions
      -es/--eval-stream             : run stream evaluation after saving raw predictions
      -td/--threshold               : evaluation threshold
      -nj/--no-events-json          : do not save stream events json file
      -ns/--no-show-roc             : do not display ROC figure
      --no-roc-csv                  : do not save ROC CSV file
      --no-print                    : do not print test report
    * deprecated/legacy eval flags:
      --evaluate                    : old compatibility flag
      -vm/--video-mode              : old compatibility flag
      -sm/--stream-mode             : old compatibility flag
"""

from pathlib import Path
import argparse
from common.my_local_utils import print_color
from torch_clip_model import run_training, run_testing
from evaluation_tools import analyze_clip_test, analyze_video_test, analyze_stream_test


def _kwargs_from_args(args, names):
    """ Collect non-None argument values from argparse namespace."""
    out = {}
    for name in names:
        val = getattr(args, name)
        if val is not None:
            out[name] = val
    return out


def _run_train(args):
    """ Run training command."""
    kw = _kwargs_from_args(args,
                           ('work_dir', 'tag', 'lr', 'epochs', 'batch_size',
                                   'hidden_dim', 'split_ratio', 'random_seed',))
    run_dir = run_training(args.train_cache, args.valid_cache, **kw)
    print(f"Training done. Run dir: {run_dir}")


def _run_test(args):
    """ Run testing command."""
    def _warn(text):
        print_color(f"[WARN] {text}", 'o')

    new_eval_target = None
    if args.eval_stream:
        if args.eval_video:
            _warn("--eval-video was skipped because --eval-stream takes precedence")
        new_eval_target = 'stream'
    elif args.eval_video:
        new_eval_target = 'video'
    elif args.eval_clip:
        new_eval_target = 'clip'

    legacy_eval_requested = False
    legacy_eval_target = None
    if args.evaluate or args.video_mode or args.stream_mode:
        _warn("Old test-evaluation flags are deprecated; use --eval-clip / --eval-video / --eval-stream")
        if args.evaluate:
            legacy_eval_requested = True
            if args.stream_mode:
                if args.video_mode:
                    _warn("Deprecated --video-mode was skipped because --stream-mode takes precedence")
                legacy_eval_target = 'stream'
            elif args.video_mode:
                legacy_eval_target = 'video'
            else:
                legacy_eval_target = 'clip'
        else:
            _warn("Deprecated mode flags without --evaluate are ignored")

    eval_target = new_eval_target if new_eval_target is not None else legacy_eval_target

    kw = _kwargs_from_args(args, ('batch_size', 'out_dir', 'output_tag'))
    if args.pure_clips:
        kw['pure_clips'] = True
    res = run_testing(args.test_model, args.test_cache, **kw)
    if res is None:
        return

    if args.pure_clips:
        if new_eval_target is not None or legacy_eval_requested:
            _warn("--pure-clips disables immediate evaluation; eval flags were ignored")
        return

    if eval_target is None:
        return

    eval_kw = {'print': args.report, 'show_roc': args.show_roc,
               'roc_csv': args.roc_csv, 'events_json': args.events_json,
               'threshold': args.threshold}

    # TODO: Remove the legacy flag compatibility, once the new CLI is fully adopted.
    if   eval_target == 'stream':
        analyze_stream_test(res['path'], **eval_kw)
    elif eval_target == 'video':
        analyze_video_test(res['path'], **eval_kw)
    else:
        analyze_clip_test(res['path'], **eval_kw)


def main():
    """ Parse CLI args and dispatch train/test commands."""
    parser = argparse.ArgumentParser('train_cli',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='Train/test clip model on cached NPZ files',)
    sub = parser.add_subparsers(dest='cmd', required=True)

    train_p = sub.add_parser('train', help='Train model from cache npz')
    train_p.add_argument('train_cache', type=Path, help='Train cache npz path')
    train_p.add_argument('-v', '--valid-cache', type=Path, default=None, help='Optional valid cache npz path')
    train_p.add_argument('-t', '--tag', type=str, default=None, help='Run tag suffix')
    train_p.add_argument('-wd', '--work-dir', type=Path, default=None, help='Output run directory root')
    train_p.add_argument('-lr', type=float, default=None, help='Learning rate')
    train_p.add_argument('-e', '--epochs', type=int, default=None, help='Number of epochs')
    train_p.add_argument('-bs', '--batch-size', type=int, default=None, help='Batch size')
    train_p.add_argument('-hd', '--hidden-dim', type=int, default=None, help='Hidden layer size')
    train_p.add_argument('-sr', '--split-ratio', type=float, default=None, help='Runtime train split ratio')
    train_p.add_argument('-rs', '--random-seed', type=int, default=None, help='Runtime split seed')
    train_p.set_defaults(fn=_run_train)

    test_p = sub.add_parser('test', help='Test model on cache npz')
    test_p.add_argument('test_model', type=Path, help='Model.pt path')
    test_p.add_argument('test_cache', type=Path, help='Test cache npz path')
    test_p.add_argument('-bs', '--batch-size', type=int,  default=None, help='Batch size')
    test_p.add_argument('-od', '--out-dir',    type=Path, default=None, help='Output dir for predictions files')
    test_p.add_argument('-t',  '--output-tag', type=str,  default=None, help='Output tag for filename')
    #* evaluation related arguments
    test_p.add_argument('-ec', '--eval-clip',  action='store_true', help='run clip evaluation after saving raw predictions')
    test_p.add_argument('-ev', '--eval-video', action='store_true', help='run video evaluation after saving raw predictions')
    test_p.add_argument('-es', '--eval-stream',action='store_true', help='run stream evaluation after saving raw predictions')
    test_p.add_argument('--pure-clips', action='store_true', help='save minimal clip-only raw NPZ (skip evaluation)')
    test_p.add_argument('-td', '--threshold', type=float, default=None, help='Evaluation threshold')
    test_p.add_argument('-nj', '--no-events-json', dest='events_json', action='store_false', help='Do not save stream events JSON file')
    test_p.add_argument('-ns', '--no-show-roc', dest='show_roc', action='store_false', help='Do not display ROC figure')
    test_p.add_argument('--no-roc-csv', dest='roc_csv', action='store_false', help='Do not save ROC CSV file')
    test_p.add_argument('--no-print',   dest='report' ,  action='store_false', help='Do not print test report')
    #* Deprecated legacy flags
    test_p.add_argument('--evaluate', action='store_true', help='legacy flag; use --eval-clip/video/stream instead')
    test_p.add_argument('-vm', '--video-mode',  dest='video_mode', action='store_true',  help='legacy flag; use --eval-clip/video/stream instead')
    test_p.add_argument('-sm', '--stream-mode', dest='stream_mode', action='store_true', help='legacy flag; use --eval-clip/video/stream instead')
    test_p.set_defaults(report=True, show_roc=False, roc_csv=True, events_json=True,
                        video_mode=False, stream_mode=False, evaluate=False,
                        eval_clip=False, eval_video=False, eval_stream=False, pure_clips=False)
    test_p.set_defaults(fn=_run_test)

    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
