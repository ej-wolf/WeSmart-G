"""  CLI wrapper for model training/testing on cached NPZ features.
    Subcommands:
    - `train`: calls `torch_clip_model.run_training`
    - `test` : calls `torch_clip_model.run_testing`
"""

from pathlib import Path
import argparse
from torch_clip_model import run_training, run_testing
from evaluation_tools import analyze_clip_test, analyze_video_test


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
    kw = _kwargs_from_args(args, ('batch_size', 'threshold', 'out_dir', 'output_tag'))
    res = run_testing(args.test_model, args.test_cache,
                      vid_info=args.vid_info, stream_mode=args.stream_mode, **kw)
    if res is not None and args.analyze:
        if args.vid_info:
            analyze_video_test(res['path'], print=args.report, show_roc=args.show_roc)
        else:
            analyze_clip_test(res['path'], print=args.report, show_roc=args.show_roc)


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
    test_p.add_argument('-td', '--threshold',  type=float,default=None, help='Decision threshold')
    test_p.add_argument('-od', '--out-dir',    type=Path, default=None, help='Output dir for predictions files')
    test_p.add_argument('-t',  '--output-tag', type=str,  default=None, help='Output prediction filename')
    test_p.add_argument('-v',  '--video-info', dest='vid_info', action='store_true', help='add source-video info to raw predictions npz')
    test_p.add_argument('-s', '--stream_mode', dest='stream_mode', action='store_true', help='use stream-style identifiers (video_name, time_stamp)')
    test_p.add_argument('-a' , '--analyze',  dest='analyze', action='store_true', help='run further predictions analysis')
    test_p.add_argument('-ns', '--no-show-roc', dest='show_roc', action='store_false', help='Do not display ROC figure')
    test_p.add_argument('--no-report',   dest='report' ,  action='store_false', help='Do not print test report')
    test_p.set_defaults(analyze=False, report=True, show_roc=True, vid_info=False, stream_mode=False)
    test_p.set_defaults(fn=_run_test)

    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
