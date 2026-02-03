import shutil
from pathlib import Path
import json

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


def clear_dir(path, missing_ok: bool = False) -> None:
    """  Delete all files and subdirectories inside `path`, but keep `path` itself.
    Any error (nonexistent path, not a directory, permissions, etc.)
    is printed and swallowed, so it won't stop the program.
    ----------
    :param path : str|Path  Directory whose contents will be removed.
    :param missing_ok : bool, default False
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


# ***** json handling ***** #x`1
def load_json_frames(json_path: str | Path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # frames =
    return data['frames'] # frames

# ***** pth handling ***** #

#*86->63
def get_epoch_pth(dir_path:str|Path, epoch:int|str|None='best') -> str|None:
    """ Return path to desired checkpoint (pth file) in dir_path.
    :param epoch:- int : desired epoch; if exact file not found, pick the closest
                         *later* epoch_XX.pth, or the last available if none later.
                 - 'best' (default): return a 'best' checkpoint if present
                         (file name containing 'best'). If none, fall back to last
                       epoch_XX.pth and print a warning.
                 - 'last': return the last epoch_XX.pth (highest epoch number).
    """
    #* Helper: extract epoch number from names like 'epoch_25.pth'
    def _epoch_of(name: str) -> int | None:
        if name.startswith('epoch_') and name.endswith('.pth'):
            num_part = name[len('epoch_'):-len('.pth')]
            try:
                return int(num_part)
            except ValueError:
                return None

    def all_pairs():
        epc_pairs = [(p.name, _epoch_of(p.name)) for p in pth_files]
        return [(p, e) for p, e in epc_pairs if e is not None]

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {dir_path}')

    pth_files = sorted(dir_path.glob('*.pth'))
    if not pth_files:
        # raise FileNotFoundError(f'No .pth files found in {dir_path}')
        print(f"[Error] No pth files found in {dir_path}")
        return None

    #* --- 'best' case ---
    if (isinstance(epoch, str) and epoch == 'best') or epoch is None:
        pth = [p for p in pth_files if 'best' in p.name]
        if pth:  # If several, pick the newest by mtime
            best_pth = max(pth, key=lambda p: p.stat().st_mtime)
            return str(best_pth.name)
        print("[WARN] No best checkpoint found, falling back to last epoch.")
        epoch = 'last'  # fallthrough

    #* --- 'last' case ---
    if isinstance(epoch, str) and epoch == 'last':
        epoch_pairs = all_pairs()
        if not epoch_pairs:
            # Only weird filenames – fall back to newest
            latest = max(pth_files, key=lambda p: p.stat().st_mtime)
            return str(latest)
        last_pth, _ = max(epoch_pairs, key=lambda pe: pe[1])
        return last_pth

    #* --- numeric (or numeric-string) epoch request ---
    if not isinstance(epoch, int):
        print(f"Unsupported epoch spec: {epoch!r}")

    #* Exact match first
    pth = f'epoch_{epoch}.pth'
    if (dir_path/pth).is_file():
        return pth
    #* Otherwise, search for closest later epoch_XX.pth
    epoch_pairs = all_pairs()
    closest_pth, _ = min(epoch_pairs,key=lambda pe: (abs(pe[1] - epoch), -pe[1]), )
    return str(closest_pth)


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


# ----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
def tst_get_pth(tst_path:str=None, **kwargs):

    tst_ls = kwargs.get('tst_ls', [None, 'best', 'last', 10, 25, 17, 51, 81, -1])
    d = Path(tst_path)
    for t in tst_ls:
        print(get_epoch_pth(d, t))
    d = d.parent
    print(get_epoch_pth(d)),  print(get_epoch_pth(d, epoch=10))


if __name__ == "__main__":
    pass
    # tst_get_pth("/mnt/local-data/Python/Projects/weSmart/work_dirs/tsm_r50_bbfrm")

#277(2,3,2)-> 260
