import shutil, re
from pathlib import Path
import json


from colorama import Fore, Style
# B, U, R = '\033[1m', '\033[4m', '\033[0m'
# RED, GREEN, BLUE = Fore.RED, Fore.GREEN, Fore.BLUE
# RESET = Style.RESET_ALL

def print_color(msg:str, clr=Fore.RED):
    if   clr in ['RED',' Red', 'red', 'r']      :  clr = Fore.RED
    elif clr in ['BLUE', 'Blue', 'blue', 'b']   :  clr = Fore.BLUE
    elif clr in ['YELLOW', 'yellow', 'y']       :  clr = Fore.YELLOW
    elif clr in ['BLUE', 'Blue', 'blue', 'r']   :  clr = Fore.BLUE
    elif clr in ['GREEN', 'Green', 'green', 'g']:  clr = Fore.GREEN
    print( f"{clr}{msg}{Style.RESET_ALL}")

# ----------------------------------------------------------------------------
# Files and Paths Utils
# -----------------------------------------------------------------------------

def _make_unique_dir(root, base_name, **kwargs):
    """  Create a unique subdir under root for base_name, adding (2), (3), ... if needed.
    :param root : str|Path,  Parent directory.
    :param base_name : str, Desired subdirectory name.
    no_space : bool, optional (via kwargs)  If True, use base_name-(2) style; otherwise base_name (2).
    """
    no_space = kwargs.get("no_space", False)
    sep = "" if no_space else " "

    root = Path(root)
    clip_name = base_name
    i = 2
    clip_dir = root / clip_name

    while clip_dir.exists():
        clip_name = f"{base_name}{sep}({i})"
        clip_dir = root / clip_name
        i += 1

    clip_dir.mkdir(parents=True, exist_ok=True)
    return clip_dir, clip_name


def clear_dir(path, missing_ok:bool = False) -> None:
    """  Delete all files and subdirectories inside `path`, but keep `path` itself.
    Any error (nonexistent path, not a directory, permissions, etc.)
    is printed and swallowed, so it won't stop the program.
    ----------
    :param path : str|Path  Directory whose contents will be removed.
    :param missing_ok : bool, defaut False
            If False, raise FileNotFoundError when path doesn't exist.
            If True, silently do nothing when path doesn't exist.
    """
    try:
        p = Path(path)
    except Exception as e:
        print(f"clear_dir: invalid path {path!r}: {e}")
        return

    try:
        if not p.exists():
            print(f"clear_dir: path does not exist: {p}")
            return

        if not p.is_dir():
            print(f"clear_dir: not a directory: {p}")
            return
    except Exception as e:
        print(f"clear_dir: failed to inspect path {p!r}: {e}")
        return

    for child in p.iterdir():
        try:
            # Don't follow symlinks, just remove the link itself
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                # Weird/unknown type – try to unlink anyway
                child.unlink(missing_ok=True)
        except Exception as e:
            print(f"clear_dir: failed to remove {child!r}: {e}")


def correct_path(path:str|Path, project_root: str|Path|None=None):
    """  Try to resolve a path defined with relative prefixes (e.g. ../../a/b/c)
    by matching only its *tail* (a/b/c) somewhere under the project root.
    Rules:
    - Ignores leading ../ components
    - Searches recursively under project_root
    - If exactly one match is found → return corrected Path
    - If zero or multiple matches → print one-line warning and return None
    """
    if path is None:
        return None

    p = Path(path)
    tail = Path(*[x for x in p.parts if x not in ('..', '.')])

    root = Path(project_root) if project_root is not None else Path.cwd()
    matches = list(root.rglob(tail.as_posix()))

    if len(matches) == 1:
        # return matches[0]
        return str(matches[0]) if isinstance(path,str) else matches[0]
    elif len(matches) == 0:
        print(f"[correct_path] NOT FOUND: {tail}")
    else:
        print(f"[correct_path] AMBIGUOUS ({len(matches)} matches): {tail}")
    return None


def get_unique_name(file_name:str|Path, n:int=3) -> Path:
    """ Return a unique file name.
    Rules:  If file does not exist → return as is.
    If exists:  my_file.txt      -> my_file_001.txt  (padding = n)
                my_file_01.txt   -> my_file_02.txt   (padding preserved = 2)
                my_file_45.txt   -> my_file_46.txt   (padding preserved = 2)
    """

    file_path = Path(file_name)
    if not file_path.exists():
        return file_path

    parent, stem, suffix = file_path.parent, file_path.stem, file_path.suffix

    match = re.search(r'_(\d+)$', stem)

    if match:
        base = stem[:match.start()]
        num_str = match.group(1)
        counter = int(num_str) + 1
        padding = len(num_str)  # preserve existing padding
    else:
        base = stem
        counter = 1
        padding = n  # use default padding

    while True:
        new_name = f"{base}_{counter:0{padding}d}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


# ***** json handling ***** #x`1
# def load_json_frames(json_path: str|Path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     # frames =
#     return data['frames'] # frames


def json_equal(f1, f2, keys=None):
# def json_equal(f1, f2, ignore_header=False):
    with open(f1) as a, open(f2) as b:
        # if ignore_header:
        #     return json.load(a).get('frames') == json.load(b).get('frames')
        if keys:
            j1, j2 = json.load(a), json.load(b)
            found = [j1.get(k) == j2.get(k)  for k in collection(keys)]
            return  all (found)
            return json.load(a).get('frames') == json.load(b).get('frames')
        return json.load(a) == json.load(b)

# ***** Basic Log handling ***** #
def _save_log(lines, log_name, log_type: str | None = None):
    """Save log lines to a file; log_type reserved for future formatting tweaks."""
    if not lines:   return

    # Default: plain text, one line per entry
    if log_type is None or log_type == "default":
        log_path = Path(log_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            for line in lines:
                lf.write(line + "\n")
    else:
        # Placeholder for future formats
        log_path = Path(log_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            for line in lines:
                lf.write(line + "\n")

def _load_log_lines(log_source):
    """Normalize log source (path or list) into list of stripped lines."""
    if isinstance(log_source, (str, Path)):
        lp = Path(log_source)
        if not lp.is_file():
            return []
        with lp.open("r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    elif isinstance(log_source, list):
        return [str(ln).strip() for ln in log_source if str(ln).strip()]
    return []


# ***** Collection casting ***** #
def as_collection(x):
    """  If x is a collection (list/tuple/set/dict/range/numpy array/torch tensor/etc.)
    return it as-is. Otherwise, wrap it in a single-element list.
    (*) Strings/bytes are treated as scalars (wrapped).
    (*) Multi-dim numpy / torch arrays are returned as-is (no special handling).
    """
    from collections.abc import Iterable

    if isinstance(x, (str, bytes, bytearray)):
        return [x]    #* treat strings/bytes as scalars, not collections

    #* optional numpy/torch support without hard dependency
    np_types = ()
    torch_types = ()
    try:
        import numpy as np
        np_types = (np.ndarray,)
    except Exception:
        pass
    try:
        import torch
        torch_types = (torch.Tensor,)
    except Exception:
        pass

    collection_types = (list, tuple, set, frozenset, range, dict) + np_types + torch_types

    #* common concrete collection types
    if isinstance(x, collection_types):
        return x

    #* other iterables (e.g. generators); treat as collections and return as-is
    if isinstance(x, Iterable):
        return x

    #* scalar fallback
    return [x]

collection = as_collection


if __name__ == "__main__":
    pass
    # tst_get_pth("/mnt/local-data/Python/Projects/weSmart/work_dirs/tsm_r50_bbfrm")
    f1 = Path("/home/ejwolf/clouds/erez.wolfson/wesmart-share/data/json/RWF-2000/train_pos/cy1gi3ZJb_c_4.json")
    f2 = Path("/home/ejwolf/clouds/erez.wolfson/wesmart-share/data/json/test/pos_004.json")
    print(f2.parent.is_dir())
    print(json_equal(f1, f2))
    print(json_equal(f1, f2, keys=['frames','fps', 'step']))

#277(2,3,2)-> 260
