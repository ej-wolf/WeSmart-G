"""
Step A & B: Offline clip-level feature precomputation and inspection

This script:
1) Loads train/val video lists
2) Slices JSON streams into clips
3) Computes motion features per clip
4) Saves cached features to disk (one file per split)
5) Prints dataset statistics for inspection

Assumes the following modules already exist:
- Temporal_slicing: slice_json_stream
- Json_Motion_features: compute_motion_sequence, temporal_conv_1d, clip_pooling
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path

# ---- import your existing pipeline ----
from temporal_slicing_json import slice_json_stream
from analyze_json_motion import compute_motion_sequence, temporal_conv_1d, clip_pooling

# --------------------------------------------------
# Defaults (move to config later if needed)
# --------------------------------------------------
VAL_RATIO = 0.2
RANDOM_SEED = 42

POOL_MODE = 'max'
APPLY_TEMPORAL_SMOOTH = True
TEMP_KERNEL = 3

# --------------------------------------------------
# * Split utilities
# --------------------------------------------------
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
    n_val = max(1, int(VAL_RATIO*len(json_files)) )

    # val_files   = json_files[:n_val]
    # train_files = json_files[n_val:]

    return {'train': json_files[n_val:], 'val': json_files[:n_val],}


def save_vid_lists(splits: dict[str, list[Path]], out_dir: str | Path):
    """     Save annotation file lists according to split keys.
        For each key in `splits`, a file named '<key>_videos.txt' is created
        containing one JSON filename per line.
    Parameters
        splits : dict returned by split_json_ds
        out_dir : directory where list files will be written
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_group, paths in splits.items():
        list_path = out_dir/f"{split_group}_videos.txt"
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
                out_path: str | Path,
                allow_empty_lbl: bool = False):
    """ Precompute clip-level motion features for a dataset split.
    Parameters
        json_dir : directory with JSON files
        list_file : text file listing JSON filenames (train/val)
        out_path : output .npz file
        allow_empty_lbl : passed to slice_json_stream (False for fully annotated)
    """
    json_dir = Path(json_dir)
    list_file = Path(list_file)
    out_path = Path(out_path)

    feats, labels, meta = [], [], []

    with open(list_file, 'r') as f:
        video_names = [ln.strip() for ln in f if ln.strip()]

    for vid in video_names:
        json_path = json_dir / vid
        with open(json_path, 'r') as f:
            _ = json.load(f)

        clips = slice_json_stream(json_path, allow_empty_lbl=allow_empty_lbl)

        for clip in clips:
            if clip['label'] is None:
                continue

            motion_seq = compute_motion_sequence(clip['frames'])

            if APPLY_TEMPORAL_SMOOTH:
                motion_seq = temporal_conv_1d(motion_seq, TEMP_KERNEL)

            clip_feat = clip_pooling(motion_seq, mode=POOL_MODE)

            labels.append(int(clip['label']))
            feats.append(clip_feat)
            meta.append({'video': vid, 't_start': clip['t_start'], 't_end': clip['t_end'] } )

    feats = np.stack(feats) if feats else np.zeros((0,))
    labels = np.asarray(labels, dtype=np.int64)

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
# * Main
# --------------------------------------------------
def main():

    parser = argparse.ArgumentParser('precompute_clips')
    parser.add_argument('jsons_dir', type=Path)
    parser.add_argument('cache_dir', type=Path)
    parser.add_argument('-s', '--split-dir',   type=Path, default=None)
    parser.add_argument('-e', '--allow-empty', type=Path, default=True)
    parser.add_argument('-rs','--random-seed', type=int,  default=RANDOM_SEED)
    parser.add_argument('-ns', '--new-split', action='store_true', help='Force New split')
    args = parser.parse_args()


    jsons_dir = args.jsons_dir
    cache_dir = args.cache_dir
    split_dir = args.split_dir or jsons_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_list = split_dir/"train_videos.txt"
    valid_list = split_dir/"val_videos.txt"

    if train_list.exists() and valid_list.exists() and not args.new_split:
        print('[INFO] Using existing train/val split files')
    else:
        print('[INFO] Creating new train/val split')
        #splits = split_json_ds(jsons_dir)
        print('random seed : ', args.random_seed)
        splits = split_json_ds(jsons_dir, random_seed=args.random_seed)
        save_vid_lists(splits, split_dir)

    precompute_features_cache(jsons_dir, split_dir / 'train_videos.txt',
                              cache_dir/'train_feats.npz', allow_empty_lbl=False)
    precompute_features_cache(jsons_dir, split_dir / 'val_videos.txt',
                              cache_dir/'val_feats.npz', allow_empty_lbl=False)


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == '__main__':
    pass
    main()


    # DATA_DIR = Path("./data/json_data")
    # JSON_DIR = DATA_DIR/"full_ann_w_keys"
    # SPLIT_DIR = DATA_DIR # "./json_annotations"
    # OUT_DIR = DATA_DIR/'cache'
    #
    # #OUT_DIR = Path(OUT_DIR)
    # OUT_DIR.mkdir(exist_ok=True)
    #
    # precompute_split_features(JSON_DIR, SPLIT_DIR/"train.txt",
    #                           OUT_DIR/'train_feats.npz', allow_empty_lbl=False, )
    #
    # precompute_split_features(JSON_DIR, SPLIT_DIR/'val.txt', OUT_DIR/'val_feats.npz',allow_empty_lbl=False)
    #
    # inspect_feature_file(OUT_DIR / 'train_feats.npz')
    # inspect_feature_file(OUT_DIR / 'val_feats.npz')
