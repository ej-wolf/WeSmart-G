"""json_utils
    Load project JSON sources and normalize them into the internal data structures.
    Public API:
        resolve_json_source(file) -> Path
        resolve_json_files(file_ls, root_dir) -> list[Path]
        list_json_sources(dir_path) -> list[Path]
        load_json_raw(file) -> dict
        save_json_raw(jdata, j_path, compression='zip', **json_kwargs) -> Path
        load_json_data(file, j_type='type_1') -> dict
"""

import gzip
import json, zipfile
from pathlib import Path
from typing import Any

# Local imports.
from common.my_local_utils import print_color

STREAM_FILE_TYPES = ('.json.zip', '.json.gz', '.json', '.zip', '.gz')

# region Public API
def is_json(file: str | Path) -> bool:
    """Return whether a path uses one of the supported JSON file suffixes."""
    return str(file).lower().endswith(STREAM_FILE_TYPES)


def _json_base(file: str | Path) -> Path:
    """Return the path without JSON/archive suffixes."""
    file = Path(file)
    name = str(file)
    for suffix in STREAM_FILE_TYPES:
        if name.endswith(suffix):
            return Path(name[:-len(suffix)])
    return file


def resolve_json_source(file:str|Path):
    """ Resolve a logical JSON path to an existing plain or archived source."""
    file = Path(file)
    base = _json_base(file)
    candidates = [file, base.with_suffix('.json'),
                        Path(str(base) + '.json.zip'), base.with_suffix('.zip'),
                        Path(str(base) + '.json.gz') , base.with_suffix('.gz')]

    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(file)


def resolve_json_files(file_ls: str|Path|list, root_dir:str|Path|None=None) -> list[Path]:
    """ Resolve JSON paths from a list file or an in-memory path list.
    Relative entries are searched below ``root_dir`` or the current working
    directory. Alternate supported suffixes are accepted, so a listed
    ``name.json.gz`` can resolve to an existing ``name.zip``.
    """
    root = Path.cwd() if root_dir is None else Path(root_dir)
    if isinstance(file_ls, (str, Path)):
        list_path = Path(file_ls)
        if list_path.is_file() and list_path.suffix.lower() == '.txt':
            entries = [Path(line.strip())
                       for line in list_path.read_text(encoding='utf-8').splitlines()
                       if line.strip() and not line.strip().startswith('#')]
        else:
            entries = [list_path]
    else:
        entries = [Path(entry) for entry in file_ls]

    resolved = []
    for entry in entries:
        if not is_json(entry):
            print(f'Skipping non-JSON entry: {entry}')
            continue
        candidate = entry if entry.is_absolute() else root / entry
        try:
            resolved.append(resolve_json_source(candidate))
        except FileNotFoundError:
            print(f'Skipping missing JSON: {candidate}')
    return resolved


def list_json_sources(dir_path: str | Path):
    """ List logical JSON dataset entries from plain or archived JSON files."""
    dir_path = Path(dir_path)
    entries = {}
    priorities = {'.json': 0, '.json.zip': 1, '.zip': 2, '.json.gz': 3, '.gz': 4}

    def source_info(path: Path):
        name = path.name
        if name.endswith('.json'):
            return path, '.json'
        if name.endswith('.json.zip'):
            return path.with_suffix(''), '.json.zip'
        if name.endswith('.zip'):
            return path.with_suffix('.json'), '.zip'
        if name.endswith('.json.gz'):
            return path.with_suffix(''), '.json.gz'
        if name.endswith('.gz'):
            return path.with_suffix('.json'), '.gz'
        return None, None

    for path in sorted(dir_path.iterdir()):
        if not path.is_file():
            continue
        logical, kind = source_info(path)
        if logical is None:
            continue

        prev = entries.get(logical.name)
        if prev is None or priorities[kind] < prev[1]:
            entries[logical.name] = (logical, priorities[kind])
    return [item[0] for item in entries.values()]


def load_json_raw(file: str|Path):
    """ Load a raw JSON dict from a plain `.json`, ZIP, or GZIP source."""
    src = resolve_json_source(file)
    if str(src).endswith('.json.gz') or src.suffix.lower() == '.gz':
        with gzip.open(src, 'rt', encoding='utf-8') as f:
            return json.load(f)

    if src.suffix.lower() != '.zip':
        with src.open('r', encoding='utf-8') as f:
            return json.load(f)

    logical_name = Path(file).name
    with zipfile.ZipFile(src, 'r') as zf:
        members = [name for name in zf.namelist() if not name.endswith('/')]
        json_members = [name for name in members if name.lower().endswith('.json')]
        target = next((name for name in json_members if Path(name).name == logical_name), None)
        if target is None:
            if len(json_members) == 1:
                target = json_members[0]
            else:
                raise ValueError(f"Ambiguous JSON archive: {src}")
        with zf.open(target, 'r') as f:
            return json.load(f)


def save_json_raw(jdata:dict, j_path:str|Path, compression='zip', **json_kwargs) -> Path:
    """Save one JSON document as plain JSON, ZIP-compressed JSON, or GZIP-compressed JSON."""
    compression = str(compression).lower()
    if compression not in {'none', 'zip', 'gz'}:
        raise ValueError(f"Unknown JSON compression: {compression}")

    j_path = Path(j_path)
    base = _json_base(j_path)
    json_kwargs = {'ensure_ascii': False, 'indent': 2, **json_kwargs}

    if compression == 'none':
        out_path = base.with_suffix('.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(jdata, f, **json_kwargs)
        return out_path

    if compression == 'gz':
        out_path = j_path if str(j_path).endswith(('.json.gz', '.gz')) else Path(str(base) + '.json.gz')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, 'wt', encoding='utf-8') as f:
            json.dump(jdata, f, **json_kwargs)
        return out_path

    out_path = j_path if str(j_path).endswith(('.json.zip', '.zip')) else Path(str(base) + '.json.zip')
    json_name = base.with_suffix('.json').name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(json_name, json.dumps(jdata, **json_kwargs))
    return out_path


def load_json_data(file:str|Path, j_type='type_1'):
    """Load one JSON file and normalize it into the internal data structure."""
    def _header(raw, version: str) -> dict[str, Any]:
        return {'video_file': raw.get('video'), 'fps': raw.get('fps'), 'sampling': raw.get('step'), 'version': version}

    def _type_1_detections(frame: dict[str, Any]) -> list[dict[str, Any]]:
        detections = []
        for bb in frame.get('bbs_list_of_keypoints', []):
            detections.append({'class': bb[0], 'conf': bb[1], 'bbox': bb[2:6],  'key_pts': bb[6]})
        return detections

    def _type_2_detections(frame: dict[str, Any]) -> list[dict[str, Any]]:
        detections = []
        for det in frame.get('detection_list', []):
            key_pts = det.get('key_points', [])
            # Pose keypoints are expected in flattened triplets: x, y, conf.
            if key_pts and len(key_pts) % 3 != 0:
                print_color(f"[WARN] key_points length not divisible by 3 in frame {frame.get('f')}", 'y')

            detections.append({'class': det['class'], 'conf': det['conf'], 'bbox': det.get('bbox', []),
                               'key_pts': key_pts,})   # Preserve the flattened format for existing consumers.
        return detections

    def _normalize_frames(raw, get_detections) -> list[dict[str, Any]]:
        frames_out = []
        for frame in raw.get('frames', []):
            frames_out.append({'f': frame.get('f'),'t': frame.get('t'),
                               'group_events': frame.get('group_events', []),
                               'detections_list': get_detections(frame)})
        return frames_out

    file = Path(file)
    raw = load_json_raw(file)

    try:
        if j_type == 'type_1':
            return {'header': _header(raw, '1.0'),
                    'frames': _normalize_frames(raw, _type_1_detections)}
        elif j_type in ['type_2', '2', 2]:
            return {'header': _header(raw, '2.0'),
                    'frames': _normalize_frames(raw, _type_2_detections)}
        else:
            print_color(f"Warning: Unknown Json format: {j_type}", 'y')
            return None
    except Exception as exc:
        raise ValueError(f"Error: Failed to load {file.name}; format: {j_type}") from exc
# endregion
