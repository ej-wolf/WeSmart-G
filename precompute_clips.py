""" CLI script.    '
    preprocess and extract features from given json dir.
    1) create train/val split, or load existing split from *.txt files
    2) slices JSON streams into temporal clips,
    3) extract motion features per clip
    4) Saves cached features to disk (one NPZ file per split)
    5) (optional) Prints dataset statistics for inspection

    usage
    >> precompute_clips  jsons_dir cache_dir [-h] [-sd SPLIT_DIR] [-e]
                        [-rs RANDOM_SEED] [-ns] [-r VALID_RATIO] [-w WINDOW] [-s STRIDE]
                        [-p] [-l] [--no-temp-smooth] [-jt {type_1,type_2,1,2}]
    positional arguments:
      jsons_dir                     : dir containing JSONs
      cache_dir                     : path for the cached NPZ feature files
    options:
      -h/ --help                    : Show help message and exit
      -cn/--cache-name              : Name for the created npz
      -sd/--split-dir [path]        : Path for the train/val list files
      -e/ --allow-empty             : Allow empty (None) labeling
      -rs/--random-seed RANDOM_SEED : set the random seed (42)
      -ns/--new-split               : Force New split (default: False)
      -r/--valid-ratio VALID_RATIO  : Validation split ratio (default: 0.2)
      -w/ --window WINDOW           : Clip window in seconds
      -s/ --stride STRIDE           : Clip stride in seconds
      -p/--pure-motion              : Use only motion features, drop the static overlap block
      -l/--legacy                   : Use only the original 18 motion features
      --no-temp-smooth              : Disable temporal smoothing
      -jt/--json-type               : Used to run old (legacy) JSON formats

    ***  info - Inspect an existing cache NPZ and print dataset/video statistics.
    usage:
    >> precompute_clips.py info npz_path [-h] [--list] [--details] [--sort SORT]
                                        [--sample SAMPLE] [-rs RANDOM_SEED]
    positional arguments:
      npz_path                      : dir containing JSONs
    options:
      -h/ --help                    : Show help message and exit
      -l/ --list                    : print video names.
      -d/ --details                 : print per-video table.
      -sr/--sort SORT               : (str) options [duration, duration_rev] or None
      -sm/--sample SAMPLE           : (int/float) sample size/portion for printing
      -rs/--random-seed             : (int) seed for random sampling (default: 42)

    ***  merge - merges multiple cache npz files into one
       out_path                     : Output merged npz path
       npz_paths                    : Cache npz files for merging
"""

import random, argparse, sys, numpy as np
from pathlib import Path
#* ---- local imports  ----
from json_utils import load_json_data
from common.my_local_utils import print_color
from temporal_slicing_json import slice_json_stream, WINDOW_SEC, STRIDE_SEC
from analyze_json_motion import extract_motion_features, _temporal_conv_1d, _clip_pooling


#* Defaults (ToDo config later if needed)
VAL_RATIO = 0.2
RANDOM_SEED = 42
POOL_MODE = 'max'
TEMPORAL_SMOOTHING = True
TEMP_KERNEL = 3
MIN_ERROR = 1e-7

#* Files formats
SPLIT_LISTS = {'train': "train_videos.txt",
               'valid': "valid_videos.txt",
               'test' : "test_videos.txt"}
VIDEO_LIST = "_videos.txt"
# CACHE_LIST = "_feats.npz"
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

def precompute_features_cache(json_dir :str|Path,
                              list_file:str|Path,
                              out_path :str|Path,
                              allow_empty_lbl:bool=False,
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

    feats, labels, meta = [], [], []

    with open(list_file, 'r') as f:
        video_names = [ln.strip() for ln in f if ln.strip()]

    for vid in video_names:
        json_path = json_dir/vid
        # clips = slice_json_stream(json_path, allow_empty_lbl=allow_empty_lbl)
        # print_color(f" file: {json_path.name} --  {json_path.is_file()}", 'b')
        json_data = load_json_data(json_path, j_type=kwargs.get('json_type', DEFAULT_TYPE))
        clips = slice_json_stream(
            json_data,
            window_sec=kwargs.get('window', WINDOW_SEC),
            stride_sec=kwargs.get('stride', STRIDE_SEC),
            allow_empty_lbl=allow_empty_lbl,
        )

        for clip in clips:
            if clip['label'] is None:
                continue
            motion_seq = extract_motion_features(
                         clip['frames'],
                         pure_motion=kwargs.get('pure_motion', False),
                         legacy=kwargs.get('legacy', False),
                         )

            if kwargs.get('temp_smooth', TEMPORAL_SMOOTHING):
                motion_seq = _temporal_conv_1d(motion_seq, TEMP_KERNEL)

            clip_feat = _clip_pooling(motion_seq, mode=POOL_MODE)

            labels.append(int(clip['label']))
            feats.append(clip_feat)
            meta.append({'video': vid, 't_start': clip['t_start'], 't_end': clip['t_end'] } )

    feats = np.stack(feats) if feats else np.zeros((0,))
    labels = np.asarray(labels, dtype=np.int64)
    #print_color(feats.shape)
    np.savez_compressed(out_path, X=feats, y=labels, meta=np.asarray(meta, dtype=object),)

    print()
    inspect_feature_file(out_path)
    print_color(f'Saved {len(labels)} clips to {out_path}', 'b')


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


def merge_cache_npz(npz_paths, out_path:str|Path):
    """Merge multiple cache NPZ files (X/y[/meta]) into one NPZ."""
    in_paths = [Path(p) for p in npz_paths]
    if len(in_paths) == 0:
        raise ValueError("No input NPZ files were provided")

    out_path = Path(out_path).with_suffix('.npz')

    X_parts, y_parts, meta_parts = [], [], []
    has_meta_flags = []
    feat_dim = None

    for p in in_paths:
        data = np.load(p, allow_pickle=True)
        if 'X' not in data.files or 'y' not in data.files:
            raise KeyError(f"{p} is missing required keys 'X'/'y'")

        X = data['X']
        y = data['y']

        if X.ndim == 1 and X.size == 0:
            if feat_dim is None:
                raise ValueError(f"{p} has empty 1D X and feature-dim cannot be inferred yet")
            X = np.zeros((0, feat_dim), dtype=np.float32)
        elif X.ndim == 2:
            if feat_dim is None:
                feat_dim = X.shape[1]
            elif X.shape[1] != feat_dim:
                raise ValueError(f"Feature dim mismatch in {p}: {X.shape[1]} != {feat_dim}")
        else:
            raise ValueError(f"Unsupported X shape in {p}: {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D in {p}, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y length mismatch in {p}: {X.shape[0]} vs {y.shape[0]}")

        X_parts.append(X.astype(np.float32))
        y_parts.append(y.astype(np.int64))

        has_meta = 'meta' in data.files
        has_meta_flags.append(has_meta)
        if has_meta:
            m = data['meta']
            if len(m) != len(y):
                raise ValueError(f"meta/y length mismatch in {p}: {len(m)} vs {len(y)}")
            meta_parts.append(np.asarray(m, dtype=object))

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    if all(has_meta_flags):
        meta_all = np.concatenate(meta_parts, axis=0)
        np.savez_compressed(out_path, X=X_all, y=y_all, meta=meta_all)
    else:
        if any(has_meta_flags):
            print("[WARN] Not all inputs contain 'meta'; merged file will skip meta.")
        np.savez_compressed(out_path, X=X_all, y=y_all)

    print_color(f"Merged {len(in_paths)} files -> {out_path}", 'b')
    inspect_feature_file(out_path)
    return out_path


def cache_info(path:str|Path, **kwargs):
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

    def _sample_count(sample_val, n_total: int) -> int:

        if sample_val is None:
            return n_total
        elif isinstance(sample_val, int):
            n = sample_val
        elif isinstance(sample_val, float):
            if 0 <= sample_val <= 1:
                n = round(sample_val * n_total)
            elif sample_val.is_integer() and sample_val > 1:
                n = int(sample_val)
            else:
                raise ValueError("sample has wrong value, Use whole number (e.g. 5)  or ratio in [0,1] (e.g. 0.3)")
        return max(0, min(n, n_total))

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
                              key=lambda n: (-max(0.0, video_info[n]['max_t'] - video_info[n]['min_t']), n))
            else:
                return sorted(name_list,
                              key=lambda n: (max(0.0, video_info[n]['max_t'] - video_info[n]['min_t']), n))

        reverse = (sort_mode == 'clips')  # large -> small
        if reverse:
            return sorted(name_list, key=lambda n: (-video_info[n]['n_clips'], n))
        return sorted(name_list, key=lambda n: (video_info[n]['n_clips'], n))

    path = Path(path)
    data = np.load(path, allow_pickle=True)
    meta = data['meta']
    y:np.ndarray = data['y']
    support = [len(y)-y.sum(), y.sum()]  #* support[0]= counts of 0, sup[1]= counts of 1
    n_clips = len(meta)

    if n_clips == 0:
        print_color(f"Empty cache file:  {path}")
        return {'path': str(path), 'n_videos': 0}

    # Build per-video aggregates from clip-level metadata.
    video_info = {}
    clip_durations = []
    stride_values = []
    valid_rows = 0

    for item in meta:
        if not isinstance(item, dict):
            continue

        vid = item.get('video')
        t_start, t_end = item.get('t_start'), item.get('t_end')
        if vid is None or t_start is None or t_end is None:
            continue

        try:
            t_start, t_end = float(t_start), float(t_end)
        except (TypeError, ValueError):
            continue

        if t_end < t_start:
            continue

        dur = t_end - t_start
        clip_durations.append(dur)
        valid_rows += 1

        rec = video_info.setdefault(str(vid),
                                   {'min_t': t_start, 'max_t': t_end, 'n_clips': 0, 'clip_time': 0.0, 'starts': []},)
        rec['min_t'] = min(rec['min_t'], t_start)
        rec['max_t'] = max(rec['max_t'], t_end)
        rec['n_clips'] += 1
        rec['clip_time'] += dur
        rec['starts'].append(t_start)

    for rec in video_info.values():
        starts = sorted(set(rec['starts']))
        if len(starts) >= 2:
            stride_values.extend([b - a for a, b in zip(starts[:-1], starts[1:]) if (b - a) > 0])

    n_videos = len(video_info)
    if n_videos == 0:
        print_color(f"[ERROR] Malformed meta data")
        return {'path': str(path), 'n_videos': 0, 'n_clips': 0, }

    total_video_time = sum(max(0.0, rec['max_t'] - rec['min_t']) for rec in video_info.values())
    avg_video_time  = (total_video_time/n_videos)
    total_clip_time = sum(clip_durations)
    # avg_clip_time = (total_clip_time/n_clips)

    # Infer window/stride from clip durations and clip start deltas.
    w_mean = np.mean(clip_durations)
    w_std  = np.std(clip_durations)
    if w_std < MIN_ERROR :
        window_msg = f"{w_mean:.2f}s"
    else:
        window_msg = f"{w_mean:.2f}s ({w_std:.3f})"

    s_mean = np.median(stride_values)
    s_std  = np.std(stride_values)
    if s_std < MIN_ERROR:
        stride_msg = f"{s_mean:.2f}s"
    else:
        stride_msg = f"~{s_mean:.2f}s ({s_std:.2f}"

    if valid_rows < n_clips:
        print_color(f"[WARN] Malformed meta: {n_clips - valid_rows} rows skipped", 'r')

    print(f"\n==== \"{path.stem}\" - Cache info ==== new !!!")
    print(f"Full path       : {path}\n"
          f"Number of videos: {n_videos}\n"
          f"Number of clips : {n_clips }\n"
          f"Total video time: {total_video_time:.2f} s\n"
          f"Avg. video Time : {avg_video_time:.2f} s\n"
          f"Total clip time : {total_clip_time:.2f} s\n"
          # f"Avg. clip time  : {avg_clip_time:.2f} s\n"
          f"Window / stride : {window_msg} / {stride_msg} (inferred)\n"
          f"GT counts 1/0   : {support[1]} / {support[0]}\n"
          f"Features number : {data['X'].shape[1] }"
          )

    #* Resolve sample size from int/ratio conventions.
    names = sorted(video_info.keys())
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
            rec = video_info[name]
            v_time = max(0.0, rec['max_t'] - rec['min_t'])
            print(f"{i:>3} | {name[:40]:40} | {v_time:^12.2f} | {rec['n_clips']:^6d} | {rec['clip_time']:15.2f} | {'':>3}")
    print()
    return {'path': str(path),
            'n_videos': n_videos,'n_clips' : n_clips,
            'total_video_time': total_video_time,
            'avg_video_time'  : avg_video_time,
            'total_clip_time': total_clip_time,
            'window': w_mean, 'window_std': w_std,
            'stride': s_mean, 'stride_std':s_std,
            'sample_size': len(names), 'support':support}
#165

def _run_build(args):
    jsons_dir:Path = args.jsons_dir
    cache_dir:Path = args.cache_dir
    split_dir:Path = args.split_dir or jsons_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    # train_list = split_dir/f"train{VIDEO_LIST}"
    # valid_list = split_dir/f"valid{VIDEO_LIST}"
    train_list = split_dir/SPLIT_LISTS['train']
    valid_list = split_dir/SPLIT_LISTS['valid']
    test_list  = split_dir/SPLIT_LISTS['test']
    #* ToDo: change valid to test

    # if train_list.exists() and valid_list.exists() and not args.new_split:
    if not args.new_split and train_list.exists() and (valid_list.exists() or test_list.exists()):
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
                               window=args.window,
                               stride=args.stride,
                               pure_motion=args.pure_motion,
                               legacy=args.legacy,
                               )
    precompute_features_cache( jsons_dir, test_list, cache_dir/f"{npz_name}_test",   # f"{npz_name}_valid",
                               allow_empty_lbl=args.allow_empty,
                               json_type=args.json_type,
                               temp_smooth=(not args.no_temp_smooth),
                               window=args.window,
                               stride=args.stride,
                               pure_motion=args.pure_motion,
                               legacy=args.legacy,
                               )


def _run_info(args):
    cache_info(args.npz_path, **vars(args))
    # print(args)
    # kwargs = vars(args).copy();  # kwargs.pop('cmd', None);  # cache_info(args.npz_path, **kwargs)


def _run_merge(args):
    merge_cache_npz(args.npz_paths, args.out_path)


# --------------------------------------------------
# * CLI entry point
# --------------------------------------------------
def main():
    """ CLI entry point."""
    parser = argparse.ArgumentParser('precompute_clips',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Build clip-feature caches from JSON files or JSON streams',
                                     )
    sub = parser.add_subparsers(dest='cmd')

    build_p = sub.add_parser('build', help='build train/valid cache files from JSON directory')
    build_p.add_argument('jsons_dir', type=Path, help="dataset path/ dir containing JSONs")
    build_p.add_argument('cache_dir', type=Path, help="path for the cached NPZ feature files")
    build_p.add_argument('-cn', '--cache-name', type=Path, default=None, help="base name for output npz files")
    build_p.add_argument('-sd', '--split-dir' , type=Path, default=None, help="path for train_videos.txt/valid_videos.txt")
    build_p.add_argument('-ns', '--new-split', action='store_true', help='force new train/valid split')
    build_p.add_argument('-r',  '--valid-ratio', type=float, default=VAL_RATIO, help='validation split ratio')
    build_p.add_argument('-rs', '--random-seed', type=int, default=RANDOM_SEED)
    build_p.add_argument('-w',  '--window', type=float, default=WINDOW_SEC, help='clip window in seconds')
    build_p.add_argument('-s',  '--stride', type=float, default=STRIDE_SEC, help='clip stride in seconds')
    build_p.add_argument('-e',  '--allow-empty', action='store_true', help="allow empty (None) labels")
    build_p.add_argument('-p',  '--pure-motion', action='store_true', help='use only motion features, without the static overlap block')
    build_p.add_argument('-l',  '--legacy',  action='store_true', help='use only the original 18 motion features')
    build_p.add_argument('--no-temp-smooth', action='store_true', help='disable temporal smoothing')
    build_p.add_argument('--json-type', default=DEFAULT_TYPE, choices=['type_1', 'type_2', '1', '2'], help='input JSON format for loader')

    info_p = sub.add_parser('info', help='print cache statistics from npz file')
    info_p.add_argument('npz_path', type=Path, help='path to cache npz file')
    info_p.add_argument('-l',  '--list',    action='store_true', help='print video names')
    info_p.add_argument('-d',  '--details', action='store_true', help='print details table per video')
    info_p.add_argument('-sm', '--sample',  type=float,help='sample size for list/details: ratio [0,1] or count (>1 integer)')
    info_p.add_argument('-sr', '--sort',    type=str, choices=['duration', 'duration_rev'], help='sorting mode for list/details')
    info_p.add_argument('-rs', '--random-seed', type=int,  help=f"Sampling random seed (default {RANDOM_SEED} )")

    merge_p = sub.add_parser('merge', help='merge multiple cache npz files into one')
    merge_p.add_argument('out_path', type=Path, help='output merged npz path')
    merge_p.add_argument('npz_paths', nargs='+', type=Path, help='input cache npz files to merge')

    #* if subcommand is missing, route it to `build' for Backward compatibility
    known_cmds = {'build', 'info', 'merge'}
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in known_cmds:
        sys.argv.insert(1, 'build')  #* set default

    args = parser.parse_args()
    if args.cmd == 'build':
        _run_build(args)
    elif args.cmd == 'info':
        _run_info(args)
    elif args.cmd == 'merge':
        _run_merge(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    pass
    main()
    # cache_info(path="data/cache/Joint_RWFLV_test.npz")

#444(,2,) -> 482(1,4,4)-
#555(1,1,4)
