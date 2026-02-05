from pathlib import Path
# import json
import random
# from typing import Dict, List

# --------------------------------------------------
# Dataset splitting utilities (video-level)
# --------------------------------------------------

# Defaults (can be moved to config later)
VAL_RATIO = 0.2
RANDOM_SEED = 66

def split_json_ds(dir_path:str|Path) -> dict[str, list[Path]]:
    """ Split a directory of JSON video annotations into train / validation sets.
        The split is done at the VIDEO level (not clip level).
    Parameters
        dir_path : directory containing JSON files (one per video)
    Returns
        splits : dict {'train': list of Path objects
                       'val'  : list of Path objects}
    Notes
        Deterministic split (controlled by RANDOM_SEED)
    """

    dir_path = Path(dir_path)
    assert dir_path.is_dir(), f"Not a directory: {dir_path}"

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {dir_path}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(json_files)

    # n_total = len(json_files)
    n_val = max(1, int(VAL_RATIO*len(json_files)) )

    val_files   = json_files[:n_val]
    train_files = json_files[n_val:]

    return {'train': train_files, 'val': val_files,}


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

    for split_name, paths in splits.items():
        list_path = out_dir/f"{split_name}.txt"
        with open(list_path, "w") as f:
            for p in paths:
                f.write(p.name + "\n")

        print(f"Written {list_path} ({len(paths)} videos)")


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":

    json_data = Path ("./data/json_data/full_ann_w_keys")
    splits = split_json_ds(json_data)
    save_vid_lists(splits, json_data.parent)
    print("Train examples:", splits['train'][:3])
    print("Val examples  :", splits['val' ][:3])
#91(,,2)
