""" CLI script.
    '
    preprocess and extract features from given json dir.
    1) create train/val split, or load existing split from *.txt files
    2) slices JSON streams into temporal clips,
    3) extract motion features per clip
    4) Saves cached features to disk (one NPZ file per split)
    5) (optional) Prints dataset statistics for inspection

    usage
    >> precompute_clips  jsons_dir cache_dir [-h] [-s SPLIT_DIR] [-e]
                        [-rs RANDOM_SEED] [-ns] [-r VALID_RATIO]
    positional arguments:
      jsons_dir                     : dir containing JSONs
      cache_dir                     : path for the cached NPZ feature files
    options:
      -h/ --help                    : Show help message and exit
      -s/ --split-dir [path]        : Path for the train/val list files
      -e/ --allow`preP-empty        : Allow empty (None) labeling (default: True)
      -cn/--cache-name              : Name for the created npz
      -rs/--random-seed RANDOM_SEED : set the random seed (42)
      -ns/--new-split               : Force New split (default: False)
      -r/--valid-ratio VALID_RATIO  : Validation split ratio (default: 0.2)

    *** `info`:  Inspect an existing cache NPZ and print dataset/video statistics.
    usage:
    >> precompute_clips.py info npz_path [-h] [--list] [--details] [--sort SORT]
                                        [--sample SAMPLE] [-rs RANDOM_SEED]
    positional arguments:
      npz_path                          : dir containing JSONs
    options:
      -h/ --help                        : Show help message and exit
      -l/ --list                        : print video names.
      -d/ --details                     : print per-video table.
      -sr/--sort                        : options [duration, duration_rev]
      -sm/--smaple                      : sample size for list/details.
      -rs/--random-seed RANDOM_SEED     : seed for random sampling (default: 42)


list (bool)    : print video names.


        sort (str|None): duration, duration_rev, clips, clips_rev.
        random_seed (int):

"""

# import json
import random, argparse, sys, numpy as np
from pathlib import Path

# ---- import your existing pipeline ----
from json_utils import load_json_data
from common.my_local_utils import print_color
from temporal_slicing_json import slice_json_stream
from analyze_json_motion import extract_motion_features, _temporal_conv_1d, _clip_pooling


#* Defaults (move to config later if needed)
VAL_RATIO = 0.2
RANDOM_SEED = 42
POOL_MODE = 'max'
TEMPORAL_SMOOTHING = True
TEMP_KERNEL = 3

#* Files formats
VIDEO_LIST = "_videos.txt"
CACHE_LIST = "_feats.npz"
DEFAULT_TYPE = 'type_2'

def split_json_ds(dir_path:str|Path, **kwargs) -> dict[str, list[Path]]:
    """ Split a directory of JSON videos into train / validation sets.
    Parameters
        dir_path : directory containing JSON files (one per video)
    Returns
        splits : dict {'train': list of Path objects
                       'val'  : list of Path objects}
    Notes
        The split is done at the VIDEO level (not clip level).
        Deterministic split (controlled by RANDOM_SEED)
    """

    dir_path = Path(dir_path)
    assert dir_path.is_dir(), f"Not a directory: {dir_path}"

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {dir_path}")

    rng = random.Random(kwargs.get('random_seed', RANDOM_SEED))
    rng.shuffle(json_files)
    # n_total = len(json_files)
    # n_val = max(1, int(kwargs.get('val_ratio',VAL_RATIO)*len(json_files)) )
    r = kwargs.get('val_ratio',VAL_RATIO)
    n_val = int(max(np.ceil(r), r*len(json_files)))

    return {'train': json_files[n_val:], 'valid': json_files[:n_val],}


def save_vid_lists(splits: dict[str, list[Path]], out_dir: str | Path):
    """ Save annotation file lists according to split keys.
        For each key in `splits`, a file named '<key>_videos.txt' is created
        containing one JSON filename per line.
    Parameters
        splits : dict returned by split_json_ds
        out_dir : directory where list files will be written
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for grp, paths in splits.items():
        list_path = out_dir/f"{grp}{VIDEO_LIST}"
        with open(list_path, 'w') as f:
            for p in paths:
                f.write(p.name + '\n')

        print(f"Written {list_path} ({len(paths)} videos)")


# --------------------------------------------------
# Step A: feature precomputation
# --------------------------------------------------

def precompute_features_cache(   #* changed
                json_dir: str | Path,
                list_file: str| Path,
                out_path : str | Path,
                allow_empty_lbl: bool = False,
                **kwargs):
    """ Precompute clip-level motion features for a dataset split.
    Parameters
        json_dir : directory with JSON files
        list_file : text file listing JSON filenames (train/val)
        out_path : output .npz file
        allow_empty_lbl : passed to slice_json_stream (False for fully annotated)
    """
    json_dir = Path(json_dir)
    list_file = Path(list_file)
    out_path = Path(out_path).with_suffix('.npz')
    #  / list_file.name.replace(VIDEO_LIST, CACHE_LIST)

    feats, labels, meta = [], [], []

    with open(list_file, 'r') as f:
        video_names = [ln.strip() for ln in f if ln.strip()]

    for vid in video_names:
        json_path = json_dir/vid

        ## clips = slice_json_stream(json_path, allow_empty_lbl=allow_empty_lbl)
        # print_color(f" file: {json_path.name} --  {json_path.is_file()}", 'b')
        json_data = load_json_data(json_path, j_type=kwargs.get('json_type', DEFAULT_TYPE))
        clips = slice_json_stream(json_data, allow_empty_lbl=allow_empty_lbl)

        for clip in clips:
            if clip['label'] is None:
                continue
            motion_seq = extract_motion_features(clip['frames'])

            if kwargs.get('temp_smooth', TEMPORAL_SMOOTHING):
                motion_seq = _temporal_conv_1d(motion_seq, TEMP_KERNEL)

            clip_feat = _clip_pooling(motion_seq, mode=POOL_MODE)

            labels.append(int(clip['label']))
            feats.append(clip_feat)
            meta.append({'video': vid, 't_start': clip['t_start'], 't_end': clip['t_end'] } )

    feats = np.stack(feats) if feats else np.zeros((0,))
    labels = np.asarray(labels, dtype=np.int64)
    print_color(feats.shape)

    print()
    np.savez_compressed(out_path, X=feats, y=labels, meta=np.asarray(meta, dtype=object),)

    print_color(f'Saved {len(labels)} clips to {out_path}', 'b')
    inspect_feature_file(out_path)


# --------------------------------------------------
# Step B: dataset inspection
# --------------------------------------------------

def inspect_feature_file(npz_path: str|Path):
    """ Inspect cached clip-level features. """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    X = data['X']
    y = data['y']

    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    print("==== Cached dataset inspection ====")
    print(f"File            : {npz_path.name}")
    print(f"#clips          : {n}")
    print(f"#positives      : {n_pos}")
    print(f"#negatives      : {n_neg}")
    print(f"Feature dim     : {X.shape[1] if X.ndim == 2 else 'N/A'}")

    if n > 0:
        print(f'Feature min/max : {X.min():.4f} / {X.max():.4f}')


def _sample_count(sample_val, n_total: int)->int:

    if   sample_val is None:
        return n_total
    elif isinstance(sample_val, int):
        n = sample_val
    elif isinstance(sample_val, float):
        if 0 <= sample_val <= 1:
            n = round(sample_val*n_total)
        elif sample_val.is_integer() and sample_val > 1:
            n = int(sample_val)
        else:
            n = 0
    if n > 0:
        return max(0, min(n, n_total))
    raise ValueError("sample has wrong value, Use whole number (e.g. 5)  or ratio in [0,1] (e.g. 0.3)")

def cache_info(path: str|Path, **kwargs):
    """ Summarize a cached NPZ and optionally print per-video list/details.
    Args:
        path: NPZ path with keys `X`, `y`, `meta`.
    kwargs:
        list (bool)    : print video names.
        details (bool): print per-video table.
        sample (int|float|None): sample size for list/details.
        sort (str|None): duration, duration_rev, clips, clips_rev.
        random_seed (int): seed for random sampling.
    """
    # def _sample_count(sample_val, n_total: int) -> int:
    #     if sample_val is None:
    #         return n_total
    #     if isinstance(sample_val, bool):
    #         raise ValueError("sample must be int/float, not bool")
    #     if isinstance(sample_val, int):
    #         return max(0, min(sample_val, n_total))
    #     if isinstance(sample_val, float):
    #         if 0 <= sample_val <= 1:
    #             return max(0, min(int(round(sample_val * n_total)), n_total))
    #         if sample_val > 1:
    #             if sample_val.is_integer():
    #                 return max(0, min(int(sample_val), n_total))
    #             raise ValueError(
    #                 "sample>1 with fraction is invalid. Use whole number (e.g. 5 or 5.0) "
    #                 "or ratio in [0,1]."
    #             )
    #     raise ValueError("sample must be int or float")

    def _sort_names(name_list):
        sort_mode = kwargs.get('sort', None)
        valid_modes = {None, 'duration', 'duration_rev', 'clips', 'clips_rev'}
        if sort_mode not in valid_modes:
            raise ValueError("sort must be one of: duration, duration_rev, clips, clips_rev")

        if sort_mode is None:
            return sorted(name_list)

        if sort_mode in {'duration', 'duration_rev'}:
            reverse = (sort_mode == 'duration')  # large -> small
            if reverse:
                return sorted(name_list,
                              key=lambda n: (-max(0.0, per_video[n]['max_t'] - per_video[n]['min_t']), n))
            else:
                return sorted(name_list,
                              key=lambda n: (max(0.0, per_video[n]['max_t'] - per_video[n]['min_t']), n))

        reverse = (sort_mode == 'clips')  # large -> small
        if reverse:
            return sorted(name_list, key=lambda n: (-per_video[n]['n_clips'], n))
        return sorted(name_list, key=lambda n: (per_video[n]['n_clips'], n))

    path = Path(path)
    data = np.load(path, allow_pickle=True)
    meta = data['meta']
    n_clips = int(len(meta))

    # Build per-video aggregates from clip-level metadata.
    per_video = {}
    clip_durations = []
    stride_values = []
    for item in meta:
        if not isinstance(item, dict):
            continue

        vid = item.get('video')
        t_start, t_end = item.get('t_start'), item.get('t_end')
        if vid is None or t_start is None or t_end is None:
            continue

        t_start, t_end = float(t_start), float(t_end)
        dur = max(0.0, t_end - t_start)
        clip_durations.append(dur)

        rec = per_video.setdefault(str(vid),
                                   {'min_t': t_start, 'max_t': t_end, 'n_clips': 0, 'clip_time': 0.0, 'starts': []},
                                   )
        rec['min_t'] = min(rec['min_t'], t_start)
        rec['max_t'] = max(rec['max_t'], t_end)
        rec['n_clips'] += 1
        rec['clip_time'] += dur
        rec['starts'].append(t_start)

    for rec in per_video.values():
        starts = sorted(set(rec['starts']))
        if len(starts) >= 2:
            stride_values.extend([b - a for a, b in zip(starts[:-1], starts[1:]) if (b - a) > 0])

    n_videos = len(per_video)
    total_video_time = sum(max(0.0, rec['max_t'] - rec['min_t']) for rec in per_video.values())
    total_clip_time = sum(clip_durations)
    avg_clip_time = (total_clip_time/n_clips) if n_clips > 0 else 0.0

    # Infer window/stride from clip durations and clip start deltas.
    window_msg = "N/A"
    if clip_durations:
        w_med = float(np.median(clip_durations))
        w_min = float(np.min(clip_durations))
        w_max = float(np.max(clip_durations))
        if abs(w_max - w_min) < 1e-9:
            window_msg = f"{w_med:.3f}s"
        else:
            window_msg = f"~{w_med:.3f}s (range {w_min:.3f}-{w_max:.3f}s)"

    stride_msg = "N/A"
    if stride_values:
        s_med = float(np.median(stride_values))
        s_min = float(np.min(stride_values))
        s_max = float(np.max(stride_values))
        if abs(s_max - s_min) < 1e-9:
            stride_msg = f"{s_med:.3f}s"
        else:
            stride_msg = f"~{s_med:.3f}s (range {s_min:.3f}-{s_max:.3f}s)"

    # print("==== Cache info ====")
    # print(f"File: {path}")
    # print(f"Number of videos: {n_videos}")
    # print(f"Number of clips: {n_clips}")
    # print(f"Total video time: {total_video_time:.2f} sec")
    # print(f"Total clip time: {total_clip_time:.2f} sec")
    # print(f"Average clip time: {avg_clip_time:.3f} sec")
    # print(f"Window / stride settings (inferred): {window_msg} / {stride_msg}")

    print(f"\n==== \"{path.stem}\" - Cache info ====")
    print(f"Full path       : {path}\n"
          f"Number of videos: {n_videos}\n"
          f"Number of clips : {n_clips}\n"
          f"Total video time: {total_video_time:.2f} s\n"
          f"Total clip time : {total_clip_time:.2f} s\n"
          f"Avg. clip time  : {avg_clip_time:.2f} s\n"
          f"Window/ stride  : {window_msg}/ {stride_msg} (inferred)")

    # Resolve sample size from int/ratio conventions.
    names = sorted(per_video.keys())
    if kwargs.get('list', False) or kwargs.get('details', False):
        sample = kwargs.get('sample', None)
        n_sample = _sample_count(sample, n_videos)
        if n_sample < n_videos:
            rng = random.Random(kwargs.get('random_seed', RANDOM_SEED))
            names = rng.sample(names, n_sample)
        names = _sort_names(names)

    if kwargs.get('list', False):
        print("\nVideos:")
        for name in names:
            print(name)

    if kwargs.get('details', False):
        print("\nDetails:")
        print(f"{'#':>3} | {'video name':40} | {'Duration (s)':^12} | {'#clips':^6} | {'Total duration(s)':>16} | {'...':>3}")
        print("-"*124)
        for i, name in enumerate(names, start=1):
            rec = per_video[name]
            v_time = max(0.0, rec['max_t'] - rec['min_t'])
            print(f"{i:>3} | {name[:40]:40} | {v_time:^12.2f} | {rec['n_clips']:^6d} | {rec['clip_time']:15.2f} | {'':>3}")
    print("\n")
    return {'path': str(path),
            'n_videos': n_videos,
            'n_clips' : n_clips,
            'total_video_time': total_video_time,
            'total_clip_time': total_clip_time,
            'avg_clip_time'  : avg_clip_time,
            'window_inferred': window_msg,
            'stride_inferred': stride_msg,
            'sample_size': len(names),
            }
#165

def _run_build(args):
    jsons_dir = args.jsons_dir
    cache_dir = args.cache_dir
    split_dir = args.split_dir or jsons_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_list = split_dir/f'train{VIDEO_LIST}'
    valid_list = split_dir/f'valid{VIDEO_LIST}'

    if train_list.exists() and valid_list.exists() and not args.new_split:
        print('[INFO] Using existing train/val split files')
    else:
        print('[INFO] Creating new train/val split')
        print('random seed : ', args.random_seed)
        splits = split_json_ds(jsons_dir, random_seed=args.random_seed, val_ratio=args.valid_ratio)
        save_vid_lists(splits, split_dir)

    npz_name = args.cache_name or jsons_dir.name
    precompute_features_cache( jsons_dir, train_list, cache_dir/f"{npz_name}_train",
                               allow_empty_lbl=args.allow_empty,
                               json_type=args.json_type,
                               temp_smooth=(not args.no_temp_smooth),
                                )
    precompute_features_cache( jsons_dir, valid_list, cache_dir/f"{npz_name}_valid",
                               allow_empty_lbl=args.allow_empty,
                               json_type=args.json_type,
                               temp_smooth=(not args.no_temp_smooth),
                               )


def _run_info(args):
    print(args)
    kwargs = vars(args).copy()
    # kwrags.pop('cmd', None)
    cache_info(args.npz_path, **kwargs)
    # cache_info(args.npz_path, list=args.list, details=args.details, sample=args.sample,
    #     sort=args.sort,random_seed=args.random_seed, )


# --------------------------------------------------
# * CLI entry point
# --------------------------------------------------
def main():

    """CLI entry point."""
    parser = argparse.ArgumentParser('precompute_clips',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Build clip-feature caches from JSON files or JSON streams',
                                    )
    sub = parser.add_subparsers(dest='cmd')

    build_p = sub.add_parser('build', help='build train/valid cache files from JSON directory')
    build_p.add_argument('jsons_dir', type=Path, help="dir containing JSONs")
    build_p.add_argument('cache_dir', type=Path, help="path for the cached NPZ feature files")
    build_p.add_argument('-s', '--split-dir', type=Path, default=None, help="path for train_videos.txt/valid_videos.txt")
    build_p.add_argument('-cn', '--cache-name', type=Path, default=None, help="base name for output npz files")
    build_p.add_argument('-e', '--allow-empty', action='store_true', help="allow empty (None) labels")
    build_p.add_argument('-rs', '--random-seed', type=int, default=RANDOM_SEED)
    build_p.add_argument('-ns', '--new-split', action='store_true', help='force new train/valid split')
    build_p.add_argument('-r', '--valid-ratio', type=float, default=VAL_RATIO, help='validation split ratio')
    build_p.add_argument('-jt', '--json-type', default=DEFAULT_TYPE, choices=['type_1', 'type_2', '1', '2'], help='input JSON format for loader')
    build_p.add_argument('--no-temp-smooth', action='store_true', help='disable temporal smoothing')

    info_p = sub.add_parser('info', help='print cache statistics from npz file')
    info_p.add_argument('npz_path', type=Path, help='path to cache npz file')
    info_p.add_argument('-l',  '--list',    action='store_true', help='print video names')
    info_p.add_argument('-d',  '--details', action='store_true', help='print details table per video')
    info_p.add_argument('-sm', '--sample',  type=float,help='sample size for list/details: ratio [0,1] or count (>1 integer)')
    info_p.add_argument('-sr', '--sort',    type=str, choices=['duration', 'duration_rev'], help='sorting mode for list/details')
    info_p.add_argument('-rs', '--random-seed', type=int,  help=f"Sampling random seed (default {RANDOM_SEED} )")

    # Backward compatibility: if user runs old build-style command without subcommand,
    # route it to `build`.
    known_cmds = {'build', 'info'}
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in known_cmds:
        sys.argv.insert(1, 'build')

    args = parser.parse_args()
    if args.cmd == 'build':
        _run_build(args)
    elif args.cmd == 'info':
        _run_info(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    pass
    main()

#444(,2,)
