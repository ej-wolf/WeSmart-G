""" CLI script.    '
    *** Build (default) ****
    preprocess and extract features from given json dir.
    1) create train/test split, or load existing split from *.txt files
    2) slices JSON streams into temporal clips,
    3) extract motion features per clip
    4) Saves cached features to disk (one NPZ file per split)
    5) (optional) Prints dataset statistics for inspection

    usage:
    >> precompute_clips.py  build  jsons_dir cache_dir [-h] [-t CACHE_TAG] [-sd SPLIT_DIR]
                            [-ns] [-r TEST_RATIO] [-rs RANDOM_SEED] [-w WINDOW] [-s STRIDE]
                            [-e] [-p] [-l] [--no-temp-smooth] [--json-type {type_1,type_2,1,2}]
    positional arguments:
      jsons_dir                     : dir containing JSONs
      cache_dir                     : path for the cached NPZ feature files
    dataset options:
      -cn/--cache-name              : legacy name for cache-tag
      -sd/--split-dir [path]        : Path for the train/test list files
      -rs/--random-seed RANDOM_SEED : set the random seed (42)
      -ns/--new-split               : Force New split (default: False)
      -r/--test-ratio TEST_RATIO    : Test split ratio (default: 0.2)

    *** stream ***
    preprocess and extract features from one long JSON stream into a single cache.
    usage:
    >> precompute_clips.py  stream json_path cache_dir [-h] [-t CACHE_TAG] [-w WINDOW]
                            [-s STRIDE] [-e] [-p] [-l] [--no-temp-smooth] [--json-type ...}]
    positional arguments:
      json_path                     : path to one long JSON stream file
      cache_dir                     : output directory for the cached NPZ feature file

    *** common options for dataset (build) and stream commands : ***
      -h/ --help                    : Show help message and exit
      -t/ --cache-tag               : base tag for the output cache npz
      -w/ --window WINDOW           : Clip window in seconds
      -s/ --stride STRIDE           : Clip stride in seconds
      -e/ --allow-empty             : Allow empty (None) labeling
      -p/ --pure-motion             : Use only motion features, drop the static overlap block
      -l/ --legacy                  : Use only the original 18 motion features
      --no-temp-smooth              : Disable temporal smoothing
      --json-type                   : input JSON format for loader

    ***  merge ***
    merges multiple cache npz files into one
    usage:      precompute_clips merge [-h] out_path npz_paths [npz_paths ...]
    positional arguments:
       out_path                     : path with name of the output merged npz path
       npz_paths                    : Cache npz files for merge
    options:
       -h/--help                    :show this help message and exit

    ***  info ***
    Inspect an existing cache NPZ and print dataset/video statistics.
    usage:
    >> precompute_clips.py info npz_path [-h] [--list] [--details] [--mode {auto,dataset,stream}] [--sort SORT]
                                        [--sample SAMPLE] [-rs RANDOM_SEED]
    positional arguments:
      npz_path                      : dir containing JSONs
    options:
      -h/ --help                    : Show help message and exit
      -l/ --list                    : print video names.
      -d/ --details                 : print per-video table.
      -m/ --mode MODE               : inspection wording mode
      -sr/--sort SORT               : (str) options [duration, duration_rev] or None
      -sm/--sample SAMPLE           : (int/float) sample size/portion for printing
      -rs/--random-seed             : (int) seed for random sampling (default: 42)
"""

import random, argparse, sys, glob, numpy as np
from pathlib import Path
#* ---- local imports  ----
from json_utils import load_json_data, list_json_sources
from common.my_local_utils import print_color, as_collection
from temporal_slicing_json import slice_json_stream, WINDOW_SEC, STRIDE_SEC
from motion_feature_schema import (DEFAULT_POOL_MODE, DEFAULT_TOP_K_MIN, DEFAULT_TOP_K_RATIO,
                                   DEFAULT_TEMP_KERNEL, DEFAULT_TEMP_SMOOTHING,
                                   FEATURE_SCHEMA_KEY, SOURCE_CACHES_KEY, TEMPORAL_SCHEMA_KEY,
                                   assert_feature_schema_match, build_cache_record, get_clip_features_vec, pack_json_value,
                                   build_feature_schema, load_cache_contract, load_cache_contract_compact, build_temporal_schema,
                                   MOTION_FPS_MAX, MOTION_FPS_MIN, MOTION_FPS_REF)


#* Defaults (ToDo config later if needed)
TEST_RATIO = 0.2
RANDOM_SEED = 42
POOL_MODE = DEFAULT_POOL_MODE
POOL_TOP_K_RATIO = DEFAULT_TOP_K_RATIO
POOL_TOP_K_MIN = DEFAULT_TOP_K_MIN
TEMPORAL_SMOOTHING = DEFAULT_TEMP_SMOOTHING
TEMP_KERNEL = DEFAULT_TEMP_KERNEL
MIN_ERROR = 1e-7

#* Files formats
SPLIT_LISTS = {'train': "train_videos.txt",
               'valid': "valid_videos.txt",
               'test' : "test_videos.txt"}
VIDEO_LIST = "_videos.txt"
# CACHE_LIST = "_feats.npz"
DEFAULT_TYPE = 'type_2'

def split_json_ds(dir_path:str|Path, **kwargs) -> dict[str, list[Path]]:
    """ Split a directory of JSON videos into train / test sets.
    Parameters
        dir_path : directory containing JSON files (one per video)
    Returns
        splits : dict {'train': list of Path objects
                       'test' : list of Path objects}
    Notes
        The split is done at the VIDEO level (not clip level).
        Deterministic split (controlled by RANDOM_SEED)
    """

    dir_path = Path(dir_path)
    assert dir_path.is_dir(), f"Not a directory: {dir_path}"

    json_files = sorted(list_json_sources(dir_path))
    if not json_files:
        raise RuntimeError(f"No JSON or zipped JSON files found in {dir_path}")

    rng = random.Random(kwargs.get('random_seed', RANDOM_SEED))
    rng.shuffle(json_files)
    # n_total = len(json_files)
    # n_test = max(1, int(kwargs.get('test_ratio', TEST_RATIO)*len(json_files)) )
    r = kwargs.get('test_ratio', kwargs.get('val_ratio', TEST_RATIO))
    n_test = int(max(np.ceil(r), r*len(json_files)))

    return {'train': json_files[n_test:], 'test': json_files[:n_test],}


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
# Step A: feature extraction
# --------------------------------------------------

def build_cache_from_json(json_paths, out_path:str|Path, **kwargs): #158->70
    """ Precompute clip-level motion features from one or more JSON paths.
        (This is the core implementation for both dataset-based and stream build.
        TODO: later add optional time-based train/test slicing for one long stream.)
    Parameters
    json_paths : JSON path or an iterable of JSON file paths
    out_path  : output .npz file
    allow_empty_lbl, window, stride: arguments passed forward to slice_json_stream
    """
    out_path = Path(out_path).with_suffix('.npz')
    json_paths = [Path(p) for p in as_collection(json_paths)]
    if not json_paths:
        raise ValueError("No JSON paths were provided")

    feature_schema = build_feature_schema(
                     pure_motion=kwargs.get('pure_motion', False),
                     legacy=kwargs.get('legacy', False),
                     temp_smooth=kwargs.get('temp_smooth', TEMPORAL_SMOOTHING),
                     temp_kernel=kwargs.get('temp_kernel', TEMP_KERNEL),
                     pool_mode=kwargs.get('pool_mode', POOL_MODE),
                     top_k_ratio=kwargs.get('top_k_ratio', kwargs.get('pool_top_k_ratio', DEFAULT_TOP_K_RATIO)),
                     top_k_min=kwargs.get('top_k_min', kwargs.get('pool_top_k_min', DEFAULT_TOP_K_MIN)),
                     motion_fps_ref=kwargs['motion_fps_ref'] if 'motion_fps_ref' in kwargs else MOTION_FPS_REF,
                     motion_fps_min=kwargs.get('motion_fps_min', MOTION_FPS_MIN),
                     motion_fps_max=kwargs.get('motion_fps_max', MOTION_FPS_MAX),
                 )
    temporal_schema = build_temporal_schema(kwargs.get('window', WINDOW_SEC), kwargs.get('stride', STRIDE_SEC),)

    feats, labels, meta = [], [], []
    for json_path in json_paths:
        json_data = load_json_data(json_path, j_type=kwargs.get('json_type', DEFAULT_TYPE))
        clips = slice_json_stream(json_data,
                                  window_sec=kwargs.get('window', WINDOW_SEC),
                                  stride_sec=kwargs.get('stride', STRIDE_SEC),
                                  allow_empty_lbl=kwargs.get('allow_empty_lbl', False),)
        for clip in clips:
            if clip['label'] is None:
                continue
            #* extract all the features and compose them into features vector
            clip_feat = get_clip_features_vec( clip['frames'],
                                 pure_motion=bool(feature_schema['pure_motion']),
                                 legacy=bool(feature_schema['legacy']),
                                 temp_smooth=bool(feature_schema['temp_smooth']),
                                 temp_kernel=int(feature_schema['temp_kernel']),
                                 pool_mode=str(feature_schema['pool_mode']),
                                 top_k_ratio=float(feature_schema['top_k_ratio']),
                                 top_k_min=int(feature_schema['top_k_min']),
                                 j_version=float(feature_schema['extractor_version']),
                                 motion_fps_ref=feature_schema['motion_fps_ref'],
                                 motion_fps_min=feature_schema['motion_fps_min'],
                                 motion_fps_max=feature_schema['motion_fps_max'],
                                 )
            labels.append(int(clip['label']))
            feats.append(clip_feat)
            meta.append({'video': json_path.name,
                         't_start': clip['t_start'],
                         't_end': clip['t_end'],
                         'n_frames': len(clip['frames']), } )

    feats = np.stack(feats) if feats else np.zeros((0, int(feature_schema['feature_dim'])), dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    #print_color(feats.shape)
    source_caches = [build_cache_record(out_path, feature_schema, temporal_schema)]
    np.savez_compressed(out_path, X=feats, y=labels, meta=np.asarray(meta, dtype=object),
                        **{ FEATURE_SCHEMA_KEY: pack_json_value(feature_schema),
                            TEMPORAL_SCHEMA_KEY: pack_json_value(temporal_schema),
                            SOURCE_CACHES_KEY: pack_json_value(source_caches),},
                        )

    print_color(f'Saved {len(labels)} clips to {out_path}', 'b')
    return out_path


def _json_paths_from_list(json_dir:str|Path, list_file:str|Path) -> list[Path]:
    """ Resolve JSON filenames from a text list into full paths under `json_dir`."""
    # json_dir = Path(json_dir)
    # list_file = Path(list_file)
    with open(Path(list_file), 'r') as f:
        video_names = [ln.strip() for ln in f if ln.strip()]
    return [Path(json_dir)/vid for vid in video_names]


# --------------------------------------------------
# Step B: cache inspection
# --------------------------------------------------

# Legacy simple inspector kept for reference; replaced by cache_info(mode=...).
# def inspect_feature_file(npz_path: str|Path):
#     """ Inspect cached clip-level features. """
#     npz_path = Path(npz_path)
#     data = np.load(npz_path, allow_pickle=True)
#
#     X , y = data['X'], data['y']
#     n = len(y)
#     n_pos = int((y == 1).sum())
#     n_neg = int((y == 0).sum())
#
#     print("==== Cached dataset inspection ====")
#     print(f"File            : {npz_path.name}")
#     print(f"#clips          : {n}")
#     print(f"#positives      : {n_pos}")
#     print(f"#negatives      : {n_neg}")
#     print(f"Feature dim     : {X.shape[1] if X.ndim == 2 else 'N/A'}")
#
#     if n > 0:
#         print(f'Feature min/max : {X.min():.4f} / {X.max():.4f}')


def merge_cache_npz(npz_paths, out_path:str|Path):
    """Merge multiple cache NPZ files (X/y[/meta]) into one NPZ."""
    in_paths = [Path(p) for p in npz_paths]
    if len(in_paths) == 0:
        raise ValueError("No input NPZ files were provided")

    out_path = Path(out_path).with_suffix('.npz')

    X_parts, y_parts, meta_parts = [], [], []
    has_meta_flags = []
    feat_dim = None
    canonical_feature_schema = None
    canonical_temporal_schema = None
    source_caches: list[dict[str, object]] = []

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

        contract = load_cache_contract(p)
        feature_schema = dict(contract['feature_schema'])
        if canonical_feature_schema is None:
            canonical_feature_schema = feature_schema
            canonical_temporal_schema = dict(contract['temporal_schema'])
        else:
            assert_feature_schema_match(canonical_feature_schema, feature_schema, context=str(p))
        source_caches.extend(dict(item) for item in contract['source_caches'])

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    save_payload = {
        'X': X_all,
        'y': y_all,
        FEATURE_SCHEMA_KEY: pack_json_value(canonical_feature_schema),
        TEMPORAL_SCHEMA_KEY: pack_json_value(canonical_temporal_schema),
        SOURCE_CACHES_KEY: pack_json_value(source_caches),
    }

    if all(has_meta_flags):
        meta_all = np.concatenate(meta_parts, axis=0)
        save_payload['meta'] = meta_all
    else:
        if any(has_meta_flags):
            print("[WARN] Not all inputs contain 'meta'; merged file will skip meta.")
    np.savez_compressed(out_path, **save_payload)

    print_color(f"Merged {len(in_paths)} files -> {out_path}", 'b')
    cache_info(out_path, mode='dataset')
    return out_path


def _resolve_input_paths(npz_inputs) -> list[Path]:
    """ Expand merge inputs where each item may be a file, dir, or glob mask."""
    resolved = []
    seen = set()

    for item in as_collection(npz_inputs):
        item = Path(item)
        item_str = str(item)

        if any(ch in item_str for ch in '*?[]'):
            matches = [Path(p) for p in glob.glob(item_str)]
        elif item.is_dir():
            matches = sorted(p for p in item.iterdir() if p.is_file() and p.suffix == '.npz')
        elif item.is_file():
            matches = [item]
        else:
            matches = []

        if not matches:
            print_color(f"[WARN] No NPZ files matched: {item}", 'y')
            continue

        for path in matches:
            if path.suffix != '.npz':
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)

    return resolved


def cache_info(path:str|Path, **kwargs):
    """ Summarize a cached NPZ and optionally print per-videolist/details.
    Args:
        path: NPZ path with keys `X`, `y`, `meta`.
    kwargs:
        mode (str): auto, dataset, stream
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
    cache_contract, used_legacy_contract = load_cache_contract_compact(path)
    feature_schema = dict(cache_contract['feature_schema'])
    temporal_schema = dict(cache_contract['temporal_schema'])
    source_caches = list(cache_contract['source_caches'])
    meta = data['meta']
    y:np.ndarray = data['y']
    support = [len(y)-y.sum(), y.sum()]  #* support[0]= counts of 0, sup[1]= counts of 1
    n_clips = len(meta)

    if n_clips == 0:
        print_color(f"Empty cache file:  {path}")
        return {'path': str(path), 'n_videos': 0}

    # Build per-video aggregates from clip-level metadata.
    clip_durations, stride_values = [],  []
    video_info = {}
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

    mode = kwargs.get('mode', 'auto')
    if mode == 'auto':
        mode = 'stream' if n_videos == 1 else 'dataset'
    if mode not in {'dataset', 'stream'}:
        raise ValueError("mode must be one of: auto, dataset, stream")

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

    if mode == 'stream':
        title = 'Stream cache info'
        count_line = f"Number of streams: {n_videos}"
        clip_line = f"Number of windows: {n_clips}"
        total_line = f"Total stream time: {total_video_time:.2f} s"
        avg_line = f"Avg. stream time : {avg_video_time:.2f} s"
        clip_time_line = f"Total window time: {total_clip_time:.2f} s"
        support_line = f"Window labels 1/0: {support[1]} / {support[0]}"
        list_label = "Streams"
        detail_name = "stream name"
    else:
        title = 'Dataset cache info'
        count_line = f"Number of videos: {n_videos}"
        clip_line = f"Number of clips : {n_clips}"
        total_line = f"Total video time: {total_video_time:.2f} s"
        avg_line = f"Avg. video Time : {avg_video_time:.2f} s"
        clip_time_line = f"Total clip time : {total_clip_time:.2f} s"
        support_line = f"GT counts 1/0   : {support[1]} / {support[0]}"
        list_label = "Videos"
        detail_name = "video name"

    feature_lines = []
    if used_legacy_contract:
        temporal_msg = f"{window_msg} / {stride_msg} (inferred)"
        feature_lines.append(f"legacy cache metadata unavailable; feature_dim={feature_schema.get('feature_dim', 'N/A')}")
        source_msg = "N/A"
    else:
        temporal_msg = f"{temporal_schema.get('window', 'N/A')}s / {temporal_schema.get('stride', 'N/A')}s (saved)"
        feature_lines.extend([# f"extractor    : {feature_schema.get('extractor', 'N/A')}",
                              f"version      : {feature_schema.get('extractor_version', 'N/A')}",
                              f"dim          : {feature_schema.get('feature_dim', 'N/A')}",
                              f"pool         : {feature_schema.get('pool_mode', 'N/A')}",
                              f"smooth       : {feature_schema.get('temp_smooth', 'N/A')}",
                              f"kernel       : {feature_schema.get('temp_kernel', 'N/A')}",
                              f"pure_motion  : {feature_schema.get('pure_motion', 'N/A')}",
                              f"legacy       : {feature_schema.get('legacy', 'N/A')}",
                            ])
        source_msg = str(len(source_caches))
    feature_block = "\n".join(f"             {line}" for line in feature_lines)

    print(f"\n==== \"{path.stem}\" - {title} ====")
    print(f"Full path       : {path}\n"
          f"{count_line}\n"
          f"{clip_line}\n"
          f"{total_line}\n"
          f"{avg_line}\n"
          f"{clip_time_line}\n"
          f"Window / stride : {temporal_msg}\n"
          f"Source caches   : {source_msg}\n"
          f"{support_line}\n"          
          f"Features number : {data['X'].shape[1]}\n"
          f"Feature extraction\n{feature_block}\n"
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
        print(f"\n{list_label}:")
        for name in names:
            print(name)

    if kwargs.get('details', False):
        print("\nDetails:")
        print(f"{'#':>3} | {detail_name:40} | {'Duration (s)':^12} | {'#clips':^6} | {'Total duration(s)':>16} | {'...':>3}")
        print("-"*124)
        for i, name in enumerate(names, start=1):
            rec = video_info[name]
            v_time = max(0.0, rec['max_t'] - rec['min_t'])
            print(f"{i:>3} | {name[:40]:40} | {v_time:^12.2f} | {rec['n_clips']:^6d} | {rec['clip_time']:15.2f} | {'':>3}")
    print()
    return {'path': str(path),
            'mode': mode,
            'n_videos': n_videos,'n_clips' : n_clips,
            'total_video_time': total_video_time,
            'avg_video_time'  : avg_video_time,
            'total_clip_time': total_clip_time,
            'window': w_mean, 'window_std': w_std,
            'stride': s_mean, 'stride_std':s_std,
            'temporal_schema': temporal_schema,
            'feature_schema': feature_schema,
            'source_caches_num': len(source_caches),
            'sample_size': len(names), 'support':support}
#165

def _run_build_cache_ds(args):
    jsons_dir:Path = args.jsons_dir
    cache_dir:Path = args.cache_dir
    split_dir:Path = args.split_dir or jsons_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_list = split_dir/SPLIT_LISTS['train']
    valid_list = split_dir/SPLIT_LISTS['valid']
    test_list  = split_dir/SPLIT_LISTS['test']

    # if train_list.exists() and valid_list.exists() and not args.new_split:
    if not args.new_split and train_list.exists() and (valid_list.exists() or test_list.exists()):
        print_color('[INFO] Using existing train/test split files','b')
    else:
        print_color(f'[INFO] Creating new train/test split\nrandom seed: {args.random_seed}','g')
        splits = split_json_ds(jsons_dir, random_seed=args.random_seed, test_ratio=args.test_ratio)
        save_vid_lists(splits, split_dir)

    eval_list = test_list if test_list.exists() else valid_list
    npz_name = args.cache_tag or jsons_dir.name
    train_cache = build_cache_from_json(_json_paths_from_list(jsons_dir, train_list), cache_dir/f"{npz_name}_train",
                                        allow_empty_lbl=args.allow_empty,
                                        json_type=args.json_type,
                                        temp_smooth=(not args.no_temp_smooth),
                                        window=args.window,
                                        stride=args.stride,
                                        pure_motion=args.pure_motion,
                                        legacy=args.legacy,
                                        pool_mode=args.pool_mode,
                                        top_k_ratio=args.top_k_ratio,
                                        top_k_min=args.top_k_min,
                                        )
    cache_info(train_cache, mode='dataset')
    test_cache = build_cache_from_json(_json_paths_from_list(jsons_dir, eval_list), cache_dir/f"{npz_name}_test",
                                       allow_empty_lbl=args.allow_empty,
                                       json_type=args.json_type,
                                       temp_smooth=(not args.no_temp_smooth),
                                       window=args.window,
                                       stride=args.stride,
                                       pure_motion=args.pure_motion,
                                       legacy=args.legacy,
                                       pool_mode=args.pool_mode,
                                       top_k_ratio=args.top_k_ratio,
                                       top_k_min=args.top_k_min,
                                       )
    cache_info(test_cache, mode='dataset')


def _run_info(args):
    npz_inputs = _resolve_input_paths(args.npz_paths)
    if not npz_inputs:
        raise ValueError("No input NPZ files were resolved for info")

    info_kwargs = vars(args).copy()
    info_kwargs.pop('cmd', None)
    info_kwargs.pop('npz_paths', None)

    for npz_path in npz_inputs:
        cache_info(npz_path, **info_kwargs)


def _run_merge(args):
    merge_inputs = _resolve_input_paths(args.npz_paths)
    if not merge_inputs:
        raise ValueError("No input NPZ files were resolved for merge")
    merge_cache_npz(merge_inputs, args.out_path)


def _run_stream_cache(args):
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = args.cache_tag or args.json_path.stem
    cache_path = build_cache_from_json(args.json_path, cache_dir/cache_tag,
                                       allow_empty_lbl=args.allow_empty,
                                       json_type=args.json_type,
                                       temp_smooth=(not args.no_temp_smooth),
                                       window=args.window,
                                       stride=args.stride,
                                       pure_motion=args.pure_motion,
                                       legacy=args.legacy,
                                       pool_mode=args.pool_mode,
                                       top_k_ratio=args.top_k_ratio,
                                       top_k_min=args.top_k_min,
                                       )
    cache_info(cache_path, mode='stream')


# --------------------------------------------------
# * CLI entry point
# --------------------------------------------------
def main():
    """ CLI entry point."""
    def _add_cache_build_args(prs):
        """Add shared cache-building options to a subparser."""
        prs.add_argument('-w', '--window', type=float, default=WINDOW_SEC, help='clip window in seconds')
        prs.add_argument('-s', '--stride', type=float, default=STRIDE_SEC, help='clip stride in seconds')
        prs.add_argument('-e', '--allow-empty', action='store_true', help="allow empty (None) labels")
        prs.add_argument('-p', '--pure-motion', action='store_true', help='use only motion features, without the static overlap block')
        prs.add_argument('-l', '--legacy', action='store_true', help='use only the original 18 motion features')
        prs.add_argument('-pm', '--pool-mode', default=POOL_MODE,
                         choices=['max', 'mean', 'lse', 'top_k', 'mean_max', 'mean_std_max', 'mm', 'msm'], help='clip pooling mode')
        prs.add_argument('-kr', '--top-k-ratio', dest='top_k_ratio', type=float, default=POOL_TOP_K_RATIO, help='ratio for top_k pooling')
        prs.add_argument('-k', '--top-k-min', dest='top_k_min', type=int, default=POOL_TOP_K_MIN, help='minimum pooled k for top_k')
        prs.add_argument('--no-temp-smooth', action='store_true', help='disable temporal smoothing')
        prs.add_argument('--json-type', default=DEFAULT_TYPE, choices=['type_1', 'type_2', '1', '2'], help='input JSON format for loader')

    parser = argparse.ArgumentParser('precompute_clips',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Build clip-feature caches from JSON files or JSON streams',)

    sub = parser.add_subparsers(dest='cmd')
    ds_bld_p = sub.add_parser('build', help='build train/test cache files from JSON directory')
    ds_bld_p.add_argument('jsons_dir', type=Path, help="dataset path/ dir containing JSONs")
    ds_bld_p.add_argument('cache_dir', type=Path, help="path for the cached NPZ feature files")
    ds_bld_p.add_argument('-t', '--cache-tag', '-cn','--cache-name',  # cn - legacy name for tag
                                      dest= 'cache_tag', type=Path, default=None, help="base tag for output npz files")
    ds_bld_p.add_argument('-sd', '--split-dir', type=Path, default=None, help="path for train_videos.txt/test_videos.txt")
    ds_bld_p.add_argument('-ns', '--new-split', action='store_true', help='force new train/test split')
    ds_bld_p.add_argument('-r',   '--test-ratio', '--valid-ratio', dest='test_ratio', type=float, default=TEST_RATIO, help='test split ratio')
    ds_bld_p.add_argument('-rs', '--random-seed', type=int, default=RANDOM_SEED)
    _add_cache_build_args(ds_bld_p)

    stream_p = sub.add_parser('stream', help='build one cache npz from a single long JSON stream')
    stream_p.add_argument('json_path', type=Path, help='path to one long JSON stream file')
    stream_p.add_argument('cache_dir', type=Path, help='output directory for the cached NPZ feature file')
    stream_p.add_argument('-t', '--cache-tag', type=Path, default=None, help='base tag for the output cache npz')
    _add_cache_build_args(stream_p)

    info_p = sub.add_parser('info', help='print cache statistics from NPZ sources')
    info_p.add_argument('npz_paths', nargs='+', type=Path, help='NPZ sources: may be a file, directory, or glob mask')
    info_p.add_argument('-l',  '--list',    action='store_true', help='print video names')
    info_p.add_argument('-d',  '--details', action='store_true', help='print details table per video')
    info_p.add_argument('-m',  '--mode',  type=str, choices=['auto', 'dataset', 'stream'],  default='auto', help='inspection wording mode')
    info_p.add_argument('-sm', '--sample',  type=float,help='sample size for list/details: ratio [0,1] or count (>1 integer)')
    info_p.add_argument('-sr', '--sort',    type=str, choices=['duration', 'duration_rev'], help='sorting mode for list/details')
    info_p.add_argument('-rs', '--random-seed', type=int,  help=f"Sampling random seed (default {RANDOM_SEED} )")

    merge_p = sub.add_parser('merge', help='merge multiple cache npz files into one')
    merge_p.add_argument('out_path', type=Path, help='output merged npz path')
    merge_p.add_argument('npz_paths', nargs='+', type=Path, help='NPZ sources: may be a file, directory, or glob mask')

    #* if subcommand is missing, route it to `build' for Backward compatibility
    known_cmds = {'build', 'stream', 'info', 'merge'}
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in known_cmds:
        sys.argv.insert(1, 'build')  #* set default

    args = parser.parse_args()
    if args.cmd == 'build':
        _run_build_cache_ds(args)
    elif args.cmd == 'stream':
        _run_stream_cache(args)
    elif args.cmd == 'info':
        _run_info(args)
    elif args.cmd == 'merge':
        _run_merge(args)
    else:
        parser.print_help()

#627(1,1,) / #636(1,1,1)
#775(1,1,1)
if __name__ == '__main__':
    pass
    main()
#444(,2,) -> 482(1,4,4)- #555(1,1,4)
