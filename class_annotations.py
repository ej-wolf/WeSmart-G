from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
import random

DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_SHUFFLE = True

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}


# region Internal helpers
def _read_ann_lines(path:str|Path)-> list[str]:
    """ Read non-empty annotation lines from a text file."""
    with Path(path).open("r") as f:
        return [line.strip() for line in f if line.strip()]

# endregion

# region Public API
def save_class_ann(ann_file: str | Path, lines: Iterable[str]) -> None:
    """ Save annotation lines to a file, one per line."""

    ann_file = Path(ann_file).resolve()
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    #* Ensure we only iterate once and strip existing newlines
    lines_list = [ln.rstrip("\n") for ln in lines if str(ln).strip()]
    with ann_file.open("w") as f:
        for ln in lines_list:
            f.write(ln + "\n")


def load_class_ann(ann_path:str|Path) -> tuple[list[int], dict[str, int]]:
    """ Load an MMAction-format class-annotation file.
    Each line has the form: "<rel/path> <label>".
    Returns: (labels, video_ann), maps relative vid paths (video_ann) to integer labels.
    """
    video_ann: dict[str, int] = {}
    labels:list[int] = []

    for line in _read_ann_lines(ann_path):
        # Use rsplit to be robust to spaces in the path
        path_str, label_str = line.rsplit(" ", 1)
        video_ann[path_str] = int(label_str)
        labels += [int(label_str)]

    return labels, video_ann


def assign_class_ann(dir_path:str|Path, ann:int, rel_dir=None, shuffle:bool=False, **kwargs)-> list[str]:
    """ Build class-annotation lines for all videos under one directory.
    The returned lines have the form "relative/path/from/rel_dir <ann>".
    If rel_dir is not provided, make paths relative to dir_path.parent.
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


def build_class_ann_files(root_dir: str | Path, shuffle: bool = False) -> dict[str, Path]:
    """ Build one class-annotation file per class subdirectory under root_dir.
    Assumes the following structure:    root_dir/
                                            classA/
                                            classB/
                                            ...
    For each class directory, writes an annotation file root_dir.parent
    named:
            <root_name>_<class_name>.txt
    Labels are integers assigned by sorted class name order (0, 1, 2, ...).
    Returns a mapping {class_name: ann_file_path}.
    """

    root_dir = Path(root_dir).resolve()
    if not root_dir.is_dir():
        raise ValueError(f"root_dir is not a directory: {root_dir}")

    dataset_root = root_dir.parent
    class_dirs: Sequence[Path] = sorted( [p for p in root_dir.iterdir() if p.is_dir()], key=lambda p: p.name)

    if not class_dirs:
        raise ValueError(f"No class subdirectories found under {root_dir}")

    label_map = {d.name: idx for idx, d in enumerate(class_dirs)}

    out_files: dict[str, Path] = {}
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        ann = label_map[cls_name]
        lines = assign_class_ann(cls_dir, ann=ann, shuffle=shuffle)
        ann_path = dataset_root / f"{root_dir.name}_{cls_name}.txt"
        save_class_ann(ann_path, lines)
        out_files[cls_name] = ann_path
        print(f"Class '{cls_name}' -> label {ann}, file {ann_path}")

    print("Label mapping:")
    for cls_name, idx in label_map.items():
        print(f"  {idx}: {cls_name}")

    return out_files


# noinspection PyIncorrectDocstring
def split_class_ann(ann_path:str|Path, out_path:str|Path|None=None,
                    out_tags:Iterable[str] = ("train", "test"),
                    ratio:float = DEFAULT_SPLIT_RATIO,
                    seed: int = DEFAULT_SPLIT_SEED,
                    shuffle:bool = DEFAULT_SPLIT_SHUFFLE,
                    ) -> tuple[Path, Path]:
    """ Split an annotation file into two parts.
    :param ann_path: path to input annotation file (rel/path <label> per line).
    :param out_path: directory for the output files. If None, uses ann_path.parent.
    :param out_tags: names for the two splits (default: ("train", "test")).
    :param ratio, seed, shuffle:  params of the random split
    :return tuple(first_path, second_path)
    """

    ann_path = Path(ann_path).resolve()
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    tags = list(out_tags)
    if len(tags) != 2:
        raise ValueError(f"out_tags must have length 2, got {len(tags)}")

    lines = _read_ann_lines(ann_path)

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

    save_class_ann(first_path, first_lines)
    save_class_ann(second_path, second_lines)

    print(f"Total: {n_total}, {tags[0]}: {len(first_lines)}, {tags[1]}: {len(second_lines)}" )
    print(f"{tags[0]} annotation: {first_path}")
    print(f"{tags[1]} annotation: {second_path}")

    return first_path, second_path


def merge_class_ann(ann_files: Iterable[str | Path], output_dir: str | Path, out_name: str = "merged_class_ann.txt",
                    prefix_ref: str | Path | None = None, shuffle: bool = False) -> Path:
    """Merge class-annotation files and rewrite paths relative to one shared reference directory.

    Each input line must have the form "<path> <label>".
    The path in each line is resolved relative to the annotation file's parent directory,
    then rewritten relative to prefix_ref. If prefix_ref is not provided, Path.cwd() is used.
    """
    paths = [Path(p).resolve() for p in ann_files]
    if not paths:
        raise ValueError("No annotation files provided to merge")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_out = output_dir / out_name
    prefix_root = Path.cwd().resolve() if prefix_ref is None else Path(prefix_ref).resolve()
    all_lines: list[str] = []

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        lines_out: list[str] = []
        for line in _read_ann_lines(path):
            path_str, label_str = line.rsplit(" ", 1)
            abs_path = (path.parent / path_str).resolve()
            try:
                rel_path = abs_path.relative_to(prefix_root)
            except ValueError as exc:
                raise ValueError(f"{abs_path} is not under prefix_ref {prefix_root}") from exc
            lines_out.append(f"{rel_path.as_posix()} {label_str}")

        print(f"Loaded {len(lines_out)} lines from {path}")
        all_lines.extend(lines_out)

    if not all_lines:
        raise ValueError("No lines found across input annotation files")

    if shuffle:
        random.shuffle(all_lines)

    save_class_ann(merged_out, all_lines)
    print(f"Merged {len(paths)} files into {merged_out} (shuffle={shuffle})")
    return merged_out
# endregion


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
        merge_splits(root_video, split_name=split)
    print("Done.")

# ----------------- IDE-friendly main -----------------

def main_local() -> None:
    """Run with hardcoded paths; ideal for IDE 'Run file'."""
    # Adjust this if your project root is different
    proj_root = Path(__file__).resolve().parent
    root_video = proj_root/"data"/"video"

    splits = ("train", "val","all")
    print(f"Using root_video = {root_video}")
    for split in splits:
        merge_splits(root_video, split_name=split)

    print("Done building joint ann files.")


def merge_splits(root_dir: Path, split_name: str, out_name: str | None = None) -> Path:
    """Merge matching dataset split annotation files into one file."""
    if out_name is None:
        out_name = f"joint_{split_name}.txt"

    rwf_ann = root_dir/"RWF-2000"/f"{split_name}.txt"
    rlvs_ann = root_dir/"RLVS"/f"{split_name}.txt"

    if not rwf_ann.is_file():
        raise FileNotFoundError(f"Missing RWF ann file: {rwf_ann}")
    if not rlvs_ann.is_file():
        raise FileNotFoundError(f"Missing RLVS ann file: {rlvs_ann}")

    lines_out: list[str] = []
    for ann_file, prefix in ((rwf_ann, "RWF-2000"), (rlvs_ann, "RLVS")):
        for line in _read_ann_lines(ann_file):
            path_str, label_str = line.rsplit(" ", 1)
            lines_out.append(f"{prefix}/{path_str} {label_str}")

    out_path = (root_dir / out_name).resolve()
    save_class_ann(out_path, lines_out)
    print(f"Wrote {len(lines_out)} entries to {out_path}")
    return out_path


#322(0,4,2); 309(2,4,2) -> 288(,,1)-> 277
if __name__ == "__main__":


    RWF_root = Path("data/video/RWF-2000")
    RWF_dir_ls = ["train/Train_Fight","train/Train_NonFight",
                  "val/Val_Fight"   , "val/Val_NonFight"]

    # for d in RWF_dir_ls:
    #     tag = 0 if 'NonFight' in d else 1
    #     ann = assign_class_ann(RWF_root/d, tag)
    #     save_class_ann(RWF_root/f"{Path(d).name}.txt",ann)

    RLVS_root = Path("data/video/RLVS")
    RLVS_dir_ls = ["Violence","NonViolence"]

    # main_local()
