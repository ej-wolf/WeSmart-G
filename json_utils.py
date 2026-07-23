"""json_utils
    Load project JSON sources and normalize them into the internal data structures.
    Public API:
        resolve_json_source(file) -> Path
        list_json_sources(dir_path) -> list[Path]
        load_json_raw(file) -> dict
        load_json_data(file, j_type='type_1') -> dict
"""

import json, zipfile
from pathlib import Path
from typing import Any

# Local imports.
from common.my_local_utils import print_color

# region Public API
def resolve_json_source(file:str|Path):
    """ Resolve a logical JSON path to an existing plain or archived source."""
    file = Path(file)
    candidates = [file]
    if file.suffix.lower() == '.json':
        candidates += [file.with_suffix('.zip'), Path(str(file) + '.zip')]
    else:
        candidates += [file.with_suffix('.json'), file.with_suffix('.zip')]

    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(file)


def list_json_sources(dir_path: str | Path):
    """List logical JSON dataset entries from plain or archived JSON files."""
    dir_path = Path(dir_path)
    entries = {}
    for path in sorted(dir_path.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith('.json'):
            logical = path
        elif name.endswith('.json.zip'):
            logical = path.with_suffix('')
        elif name.endswith('.zip'):
            logical = path.with_suffix('.json')
        else:
            continue

        prev = entries.get(logical.name)
        if prev is None or prev.suffix != '.json':
            entries[logical.name] = logical
    return list(entries.values())


def load_json_raw(file: str|Path):
    """ Load a raw JSON dict from a plain `.json` or supported ZIP source."""
    src = resolve_json_source(file)
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
