import shutil, re, zipfile, fnmatch
import numpy as np, torch, json
from pathlib import Path


from colorama import Fore, Style
# B, U, R = '\033[1m', '\033[4m', '\033[0m'
# RED, GREEN, BLUE = Fore.RED, Fore.GREEN, Fore.BLUE
# RESET = Style.RESET_ALL

def rgb(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def print_color(msg:str, clr=Fore.RED):
    if   clr in ['RED',' Red', 'red', 'r']      :  clr = Fore.RED
    elif clr in ['BLUE', 'Blue', 'blue', 'b']   :  clr = Fore.BLUE
    elif clr in ['YELLOW', 'yellow', 'y']       :  clr = Fore.YELLOW
    elif clr in ['BLUE', 'Blue', 'blue', 'r']   :  clr = Fore.BLUE
    elif clr in ['GREEN', 'Green', 'green', 'g']:  clr = Fore.GREEN
    elif clr in ['ORANGE', 'orange', 'o']       :  clr = rgb(255, 165, 0)
    elif clr in ['dark_orange', 'do']           :  clr = rgb(210, 100, 30)

    print( f"{clr}{msg}{Style.RESET_ALL}")


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
        # import numpy as np
        np_types = (np.ndarray,)
    except Exception:
        pass
    try:
        #import torch
        torch_types = (torch.Tensor,)
    except Exception:
        pass

    # * common concrete collection types
    collection_types = (list, tuple, set, frozenset, range, dict) + np_types + torch_types
    if isinstance(x, collection_types):
        return x
    #* other iterables (e.g. generators); treat as collections and return as-is
    if isinstance(x, Iterable):
        return x
    #* scalar fallback
    return [x]

collection = as_collection

# ***** General Files/ Paths sys Utils ***************************************#
# -----------------------------------------------------------------------------

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


def resolve_output_path(src_path, output_name, out_path=None, suffix=None):
    """ Resolve an output path from explicit path or source file location.
    If `suffix` is given, return the resolved path with the given suffix,
    otherwise return  with stem only.
    """
    out_path = out_path or (src_path.parent if src_path else None)
    if out_path is None:
        return None
    out_path = Path(out_path)
    if out_path.suffix == '':
        resolved = out_path/output_name
    else:
        resolved = out_path

    if suffix in [None, '']:
        return resolved
    suffix = str(suffix)
    if not suffix.startswith('.'):
        suffix = f'.{suffix}'
    return resolved.with_suffix(suffix)


def clear_dir(path, missing_ok:bool = False) -> None:
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


#* region ***** Compressing Utils  ****************************************************#

def zip_dir(target_dir:Path|str, method='file', protocol='zip', rm_policy='ask', mask=None):
    """ Compress a directory either child-by-child or as one archive.
    method:     'file'  -> zip each direct child of `dir` separately
                'dir'   -> zip the whole directory into one archive
    protocol:   'zip' implemented now,  ('7z' / 'rar' reserved for later)
    rm_policy:  'keep'  -> keep originals
                'remove'-> remove originals after successful compression
                'ask'   -> ask once at the end whether to remove originals
    """
    def _matches_mask(path_obj):
        if mask in [None, '']:
            return True
        rel_name = str(path_obj.relative_to(target_dir)) if path_obj != target_dir else path_obj.name
        return fnmatch.fnmatch(path_obj.name, mask) or fnmatch.fnmatch(rel_name, mask)

    target_dir = Path(target_dir)
    if not target_dir.is_dir():
        raise NotADirectoryError(target_dir)
    if method not in {'file', 'dir'}:
        raise ValueError(f"Unknown zip_dir method: {method}")
    archived, originals = [], []
    if method == 'file':
        for child in sorted(target_dir.iterdir()):
            if not _matches_mask(child):
                continue
            archive_path = zip_one_path(child, protocol=protocol)
            archived.append(archive_path)
            originals.append(child)
    else:
        members = [child for child in sorted(target_dir.rglob('*')) if child.is_file() and _matches_mask(child)]
        archive_path = zip_one_path(target_dir, protocol=protocol, members=members)
        archived.append(archive_path)
        originals.append(target_dir)

    if _archive_rm_decision(rm_policy, len(originals), "Remove original sources after archiving"):
        for src in originals:
            if src.is_dir():
                shutil.rmtree(src)
            elif src.exists():
                src.unlink()

    return archived


def unzip_dir(z: Path | str, rm_policy='ask'):
    """ Extract per-file ZIPs from a directory, or unpack a directory ZIP and then extract nested JSON ZIPs.
    If `z` is a directory:   extract every direct `*.zip` child into that directory.
    If `z` is a `.zip` file: unpack it first, then extract nested `*.zip` files so the final outputs are plain files like `foo.json`.
    """
    z = Path(z)
    extracted = []
    archives_to_remove = []

    if z.is_file():
        if z.suffix.lower() != '.zip':
            raise ValueError(f"Expected a directory or .zip file, got: {z}")
        direct_files = [Path(name) for name in zipfile.ZipFile(z, 'r').namelist() if not name.endswith('/')]
        root_parts = {name.parts[0] for name in direct_files if name.parts}
        if len(root_parts) == 1 and next(iter(root_parts)) == z.stem:
            root_dir = z.parent / z.stem
            _extract_zip_file(z, out_dir=z.parent)
        else:
            root_dir = z.parent / z.stem
            _extract_zip_file(z, out_dir=root_dir)
        archives_to_remove.append(z)
    elif z.is_dir():
        root_dir = z
    else:
        raise FileNotFoundError(z)

    nested_archives = sorted(root_dir.rglob('*.zip'))
    for archive in nested_archives:
        extracted.extend(_extract_zip_file(archive, out_dir=archive.parent))
        archives_to_remove.append(archive)

    if _archive_rm_decision(rm_policy, len(archives_to_remove), "Remove ZIP archives after extracting"):
        for archive in archives_to_remove:
            if archive.exists():
                archive.unlink()

    return extracted

#* compressing helpers
def zip_one_path(path: str | Path, protocol='zip', members=None):
    """ Archive one file or directory and return the created archive path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    archive_path = path.with_suffix('.zip') if path.is_file() else path.parent / f"{path.name}.zip"
    if archive_path.exists():
        archive_path = get_unique_name(archive_path)
    if protocol != 'zip':
        # TODO: add 7z / rar support once dependency policy is settled.
        raise NotImplementedError(f"archive protocol '{protocol}' is not implemented yet")

    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        if path.is_file():
            zf.write(path, arcname=path.name)
        else:
            children = members if members is not None else sorted(path.rglob('*'))
            for child in children:
                child = Path(child)
                if child.is_file():
                    zf.write(child, arcname=str(child.relative_to(path.parent)))
    return archive_path


def _archive_rm_decision(rm_policy, n_items, action):
    """ Resolve whether archived/source items should be removed."""
    if rm_policy not in {'keep', 'remove', 'ask'}:
        raise ValueError(f"Unknown rm_policy: {rm_policy}")
    if rm_policy == 'remove':
        return True
    if rm_policy == 'keep':
        return False
    try:
        ans = input(f"{action} {n_items} item(s)? [Y/N]: ").strip().lower()
    except EOFError:
        ans = ''
    return ans in {'y','Y', 'Yes', 'yes'}


def _extract_zip_file(zip_path: str | Path, out_dir: str | Path | None = None):
    """ Extract one ZIP archive and return the created file paths."""

    zip_path = Path(zip_path)
    if zip_path.suffix.lower() != '.zip':
        raise ValueError(f"Not a ZIP file: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [Path(name) for name in zf.namelist() if not name.endswith('/')]
        root_parts = {name.parts[0] for name in names if name.parts}

        if out_dir is None:
            if len(root_parts) == 1 and next(iter(root_parts)) == zip_path.stem:
                out_dir = zip_path.parent
            else:
                out_dir = zip_path.parent
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(out_dir)

    return [out_dir / name for name in names]

# endregion

# ***** JSON  Utils  *********************************************************#
def serialize_json_data(value):
    """Recursively convert numpy scalars/arrays into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): serialize_json_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_json_data(v) for v in value]
    if isinstance(value, tuple):
        return [serialize_json_data(v) for v in value]
    if isinstance(value, np.ndarray):
        return [serialize_json_data(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ***** json handling ***** #x`1
def json_equal(jf1, jf2, keys):
    """ Compare selected JSON values by dotted key paths.
    Path format: 'key.subkey.other_key'
    """
    def _get_by_path(obj, path):
        cur = obj
        # Walk nested dicts using a dotted path like 'header.fps'.
        for part in str(path).split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return None, False
            cur = cur[part]
        return cur, True
    with open(jf1) as a, open(jf2) as b:
        j1, j2 = json.load(a), json.load(b)
    for path in collection(keys):
        v1, ok1 = _get_by_path(j1, path)
        v2, ok2 = _get_by_path(j2, path)
        if ok1 != ok2 or v1 != v2:
            return False
    return True


# ***** Basic log utils ******************************************************#
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

#277(2,3,2)-> 260. 435(1,9,4)->
if __name__ == "__main__":
    pass
    f1 = Path("/home/ejwolf/clouds/erez.wolfson/wesmart-share/data/json/RWF-2000/train_pos/cy1gi3ZJb_c_4.json")
    f2 = Path("/home/ejwolf/clouds/erez.wolfson/wesmart-share/data/json/test/pos_004.json")
    # print(f2.parent.is_dir())
    # print(json_equal(f1, f2))
    # print(json_equal(f1, f2, keys=['frames','fps', 'step']))
    unzip_dir("/mnt/local-data/Projects/Wesmart/Video-datasets/HMC/events.zip")
