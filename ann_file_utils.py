from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
import random

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def save_ann_file(ann_file: str | Path, lines: Iterable[str]):
    """ Save annotation lines to a file (one per line).
    Creates the parent directory if needed and returns the output Path.
    """
    ann_file = Path(ann_file).resolve()
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    # Ensure we only iterate once and strip existing newlines
    lines_list = [ln.rstrip("\n") for ln in lines if str(ln).strip()]
    with ann_file.open("w") as f:
        for ln in lines_list:
            f.write(ln + "\n")


def load_ann_file(ann_path:str|Path) -> tuple(list[int], dict[str, int]):
    """ Load an MMAction-format annotation file.
        <rel/path> <label>
    Returns:  {video_names[str]: labels[int]}  - 'relative paths'
    """
    ann_path = Path(ann_path)
    video_ann: dict[str, int] = {}
    labels: list[int] = []

    with ann_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Use rsplit to be robust to spaces in the path
            path_str, label_str = line.rsplit(" ", 1)
            video_ann[path_str] = int(label_str)
            labels += [int(label_str)]

    return labels, video_ann



def annotate_dir(dir_path:str|Path, ann:int, rel_dir=None,
                 shuffle:bool=False, **kwargs) -> list[str]:
    """ Return annotation lines for a single class directory.
        the returned lines have the format::
        relative/path/from_dataset_root <ann>   where `dataset_root` is `dir_path.parent`.
    :param dir_path: path to a directory containing video files for ONE class.
    :param ann: integer labl to assign to all videos in this directory.
    """

    dir_path = Path(dir_path).resolve()
    rel_dir = dir_path.parent if rel_dir is None else Path(rel_dir).resolve()

    if not dir_path.is_dir():
        raise ValueError(f"dir_path is not a directory: {dir_path}")

    lines: list[str] = []
    for f in dir_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        rel_path = f.relative_to(rel_dir)
        lines.append(f"{rel_path.as_posix()} {ann}")

    if shuffle:
        random.seed(kwargs.get('seed', 42))
        random.shuffle(lines)

    return lines


def build_ann_files(root_dir:str|Path, shuffle:bool = False)-> dict[str, Path]:
    """ Build one annotation file per class subdirectory under ``root_dir``.
    Assumes the following structure::
        root_dir/
            classA/
            classB/
            ...
    For each class directory, writes an annotation file next to ``root_dir``
    (i.e. in ``root_dir.parent``) named::
        <root_name>_<class_name>.txt
    Labels are integers assigned by sorted class name order (0, 1, 2, ...).
    Returns a mapping ``{class_name: ann_file_path}``.
    """

    root_dir = Path(root_dir).resolve()
    if not root_dir.is_dir():
        raise ValueError(f"root_dir is not a directory: {root_dir}")

    dataset_root = root_dir.parent
    class_dirs: Sequence[Path] = sorted(
        [p for p in root_dir.iterdir() if p.is_dir()], key=lambda p: p.name
    )

    if not class_dirs:
        raise ValueError(f"No class subdirectories found under {root_dir}")

    label_map = {d.name: idx for idx, d in enumerate(class_dirs)}

    out_files: dict[str, Path] = {}
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        ann = label_map[cls_name]
        lines = annotate_dir(cls_dir, ann=ann, shuffle=shuffle)
        ann_path = dataset_root / f"{root_dir.name}_{cls_name}.txt"
        save_ann_file(ann_path, lines)
        out_files[cls_name] = ann_path
        print(f"Class '{cls_name}' -> label {ann}, file {ann_path}")

    print("Label mapping:")
    for cls_name, idx in label_map.items():
        print(f"  {idx}: {cls_name}")

    return out_files


def split_ann_file(ann_path:str|Path, out_path: str|Path|None = None,
                   out_tags: Iterable[str] = ("train", "test"),
                   ratio: float = 0.8,  seed: int = 42, shuffle: bool = True,
                   ) -> tuple[Path, Path]:
    """Split an annotation file into two parts.
    ann_path: path to input annotation file (``rel/path <label>`` per line).
    out_path: directory for the output files. If ``None``, uses ``ann_path.parent``.
    out_tags: names for the two splits (default: ("train", "test")).
    ratio:    fraction of lines to allocate to the *first* tag.
    Returns `(first_path, second_path)`
    """

    ann_path = Path(ann_path).resolve()
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    tags = list(out_tags)
    if len(tags) != 2:
        raise ValueError(f"out_tags must have length 2, got {len(tags)}")

    with ann_path.open("r") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]

    if not lines:
        raise ValueError(f"No lines in annotation file: {ann_path}")

    if shuffle:
        random.seed(seed)
        random.shuffle(lines)

    n_total = len(lines)
    n_first = int(ratio * n_total)
    first_lines = lines[:n_first]
    second_lines = lines[n_first:]

    base_dir = Path(out_path).resolve() if out_path is not None else ann_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    stem = ann_path.stem
    first_path = base_dir / f"{stem}_{tags[0]}.txt"
    second_path = base_dir / f"{stem}_{tags[1]}.txt"

    save_ann_file(first_path, first_lines)
    save_ann_file(second_path, second_lines)

    print(f"Total: {n_total}, {tags[0]}: {len(first_lines)}, {tags[1]}: {len(second_lines)}" )
    print(f"{tags[0]} annotation: {first_path}")
    print(f"{tags[1]} annotation: {second_path}")

    return first_path, second_path


def merge_ann_files(ann_files: Iterable[str|Path],  merged_out: str|Path,
    shuffle: bool = False) -> Path:
    """Merge multiple annotation files into a single one.
    ``ann_files``: iterable of paths to annotation files.
    ``merged_out``: output file path.
    If ``shuffle`` is True, the concatenated lines are shuffled.
    Returns the output Path.
    """

    merged_out = Path(merged_out).resolve()

    paths = [Path(p).resolve() for p in ann_files]
    if not paths:
        raise ValueError("No annotation files provided to merge")

    all_lines: list[str] = []

    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        with p.open("r") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        print(f"Loaded {len(lines)} lines from {p}")
        all_lines.extend(lines)

    if not all_lines:
        raise ValueError("No lines found across input annotation files")

    if shuffle:
        # random.seed(seed)
        random.shuffle(all_lines)

    save_ann_file(merged_out, all_lines)
    print(f"Merged {len(paths)} files into {merged_out} (shuffle={shuffle})")

    return merged_out


def proc_lib(r_dir, dir_ls, sfl=False):
    for d in dir_ls:
        ann = annotate_dir(r_dir/d, 0 if 'Non' in d else 1,
                           rel_dir=r_dir, shuffle=sfl)
        save_ann_file(r_dir/f"{Path(d).name}.txt",ann)

####**********************************

def _prefix_ann_file(in_path: Path, dataset_prefix: str) -> list[str]:
    """ Read an ann.txt and prefix paths with dataset directory.
        input line:    <rel/path> <label>  ->
        Output line:  <dataset_prefix>/<rel/path> <label>
    """

    lines_out: list[str] = []
    with in_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path_str, label_str = line.rsplit(" ", 1)
            new_path = f"{dataset_prefix}/{path_str}"
            lines_out.append(f"{new_path} {label_str}\n")
    return lines_out


def build_joint_split(root_video:Path, split_name:str, out_name:str|None=None) -> Path:
    """Build a joint annotation file for a given split.
    :param root_video: Path - The 'data/video' directory.
    :param split_name: str  - One of 'train', 'val', 'all' (whatever exists in each dataset).
    :param out_name  : str or None -  Output file name. If None, use f'joint_{split_name}.txt'.
    Return:  Path to the created joint annotation file.
    """
    if out_name is None:
        out_name = f"joint_{split_name}.txt"

    rwf_root = root_video / "RWF-2000"
    rlvs_root = root_video / "RLVS"

    rwf_ann = rwf_root / f"{split_name}.txt"
    rlvs_ann = rlvs_root / f"{split_name}.txt"

    if not rwf_ann.is_file():
        raise FileNotFoundError(f"Missing RWF ann file: {rwf_ann}")
    if not rlvs_ann.is_file():
        raise FileNotFoundError(f"Missing RLVS ann file: {rlvs_ann}")

    out_path = root_video / out_name

    # Build prefixed lines from both datasets
    lines: list[str] = []
    lines.extend(_prefix_ann_file(rwf_ann, "RWF-2000"))
    lines.extend(_prefix_ann_file(rlvs_ann, "RLVS"))

    # Optional: shuffle
    # import random
    # random.shuffle(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} entries to {out_path}")
    return out_path


# ----------------- IDE-friendly main -----------------


def main_local() -> None:
    """Run with hardcoded paths; ideal for IDE 'Run file'."""
    # Adjust this if your project root is different
    proj_root = Path(__file__).resolve().parent
    root_video = proj_root/"data"/"video"

    splits = ("train", "val","all")
    print(f"Using root_video = {root_video}")
    for split in splits:
        build_joint_split(root_video, split_name=split)

    print("Done building joint ann files.")


# Optional: keep a CLI main as well, if you ever want it.
def main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build joint RWF+RLVS ann files.")
    parser.add_argument("--root-video", type=str, default="data/video", help="Root data directory")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"],
                        help="ann/label files to merge (default: train val)" )

    args = parser.parse_args()
    root_video = Path(args.root_video)
    for split in args.splits:
        build_joint_split(root_video, split_name=split)
    print("Done.")


if __name__ == "__main__":


    RWF_root = Path("data/video/RWF-2000")
    RWF_dir_ls = ["train/Train_Fight","train/Train_NonFight",
                  "val/Val_Fight"   , "val/Val_NonFight"]

    # for d in RWF_dir_ls:
    #     tag = 0 if 'NonFight' in d else 1
    #     ann = annotate_dir(RWF_root/d, tag)
    #     save_ann_file(RWF_root/f"{Path(d).name}.txt",ann)

    RLVS_root = Path("data/video/RLVS")
    RLVS_dir_ls = ["Violence","NonViolence"]

    # proc_lib(RWF_root, RWF_dir_ls, )
    # proc_lib(RLVS_root, RLVS_dir_ls)
    main_local()

#322(0,4,2)
