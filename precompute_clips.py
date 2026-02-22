""" CLI script.
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
      -h, --help                    : Show help message and exit
      -s/--split-dir [path]         : Path for the train/val list files
      -e/--allow-empty              : Allow empty (None) labeling (default: True)
      -rs/--random-seed RANDOM_SEED : set the random seed (42)
      -ns/--new-split               : Force New split (default: False)
      -r/--valid-ratio VALID_RATIO  : Validation split ratio (default: 0.2)
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path

# ---- import your existing pipeline ----
from json_utils import  load_json_data
from my_local_utils import print_color
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
DEFAULT_TYPE = 'type_1'

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
                out_dir : str | Path,
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
    out_path = Path(out_dir)/list_file.name.replace(VIDEO_LIST, CACHE_LIST)

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

    np.savez_compressed(out_path, X=feats, y=labels, meta=np.asarray(meta, dtype=object),)

    print(f'Saved {len(labels)} clips to {out_path}')
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

# --------------------------------------------------
# * CLI entry point
# --------------------------------------------------
def main():
    """ CLI entry point. """

    parser = argparse.ArgumentParser('precompute_clips',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('jsons_dir', type=Path, help="dir containing JSONs")
    parser.add_argument('cache_dir', type=Path, help="path for the cached NPZ feature files")
    parser.add_argument('-s', '--split-dir',   type=Path, default=None, help="path for the train_videos.txt / val_videos.txt ")
    parser.add_argument('-e', '--allow-empty', type=Path, default=True, help="'Allow empty (None) labeling")
    parser.add_argument('-rs','--random-seed', type=int,  default=RANDOM_SEED)
    parser.add_argument('-ns', '--new-split', action='store_true', help='Force New split')
    parser.add_argument('-r',  '--valid-ratio', type=float, default=VAL_RATIO, help='Validation split ratio')
    args = parser.parse_args()

    jsons_dir = args.jsons_.dir
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

    precompute_features_cache(jsons_dir, train_list, cache_dir, allow_empty_lbl=False)
    precompute_features_cache(jsons_dir, valid_list, cache_dir, allow_empty_lbl=False)



if __name__ == '__main__':
    pass
    main()

#235(,,1) -> 250(,,) ->  213
