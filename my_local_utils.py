from pathlib import Path
import shutil

# def _make_unique_dir(root, base_name):
#     """Create a unique subdir under root for base_name, adding (2), (3), ... if needed."""
#
#     root = Path(root)
#     clip_name = base_name
#     i = 2
#     clip_dir = root/clip_name
#     while clip_dir.exists():
#         clip_name = f"{base_name} ({i})"
#         clip_dir = root / clip_name
#         i += 1
#     clip_dir.mkdir(parents=True, exist_ok=True)
#     return clip_dir, clip_name


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


" *** Basic Log handling *** "
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

" *** Collection casting ***"
def as_collection(x):
    """  If x is a collection (list/tuple/set/dict/range/numpy array/torch tensor/etc.)
    return it as-is. Otherwise, wrap it in a single-element list.

    Strings/bytes are treated as scalars (wrapped).
    Multi-dim numpy / torch arrays are returned as-is (no special handling).
    """
    from collections.abc import Iterable
    # treat strings/bytes as scalars, not collections
    if isinstance(x, (str, bytes, bytearray)):
        return [x]

    # optional numpy / torch support without hard dependency
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
