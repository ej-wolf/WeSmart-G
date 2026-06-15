"""json_utils
    Load project JSON sources and compare stream JSON outputs semantically.
    Public API:
        resolve_json_source(file) -> Path
        list_json_sources(dir_path) -> list[Path]
        load_json_raw(file) -> dict
        load_json_data(file, j_type='type_1') -> dict
        compare_stream_json(j1, j2, *, tolerances=None, ignore_path_fields=True) -> (ok, report)
"""

import json, zipfile
from pathlib import Path
from typing import Any

# Local imports.
from common.my_local_utils import print_color

DEFAULT_STREAM_JSON_TOLERANCES = {'avg_abs': 1e-4, 'max_abs': 1e-3}

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


def compare_stream_json(j1, j2, *, tolerances=None, ignore_path_fields=True) -> tuple[bool, dict[str, Any]]:
    """Compare two stream JSONs by metadata, frame layout, annotations, and numeric payload."""
    tolerances = {**DEFAULT_STREAM_JSON_TOLERANCES, **(tolerances or {})}
    data_1 = j1 if isinstance(j1, dict) else load_json_raw(j1)
    data_2 = j2 if isinstance(j2, dict) else load_json_raw(j2)

    metadata = _compare_stream_metadata(data_1, data_2, ignore_path_fields=ignore_path_fields)
    frame_structure = _compare_stream_frame_structure(data_1.get('frames', []), data_2.get('frames', []))
    numeric = _compare_stream_numeric_content(data_1.get('frames', []), data_2.get('frames', []), tolerances)
    annotations = _compare_stream_annotations(data_1, data_2)

    ok = (not metadata['unequal'] and
          not (frame_structure['frame_count'] or
               frame_structure['missing_frame_indices'] or
               frame_structure['extra_frame_indices'] or
               frame_structure['timestamp_mismatches'] or
               frame_structure['detection_count_mismatches']) and
          annotations['event_intervals_equal'] and
          not annotations['frame_annotation_mismatches'] and
          numeric['within_tolerance'])

    report = {'ok': ok,
              'metadata': metadata,
              'frame_structure': frame_structure,
              'numeric': numeric,
              'annotations': annotations}
    return ok, report

# endregion


# region Internal helpers
def _normalize_event_intervals(value):
    """Normalize one-item-list interval payloads to the dict shape used by the comparator."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return value


def _compare_stream_metadata(data_1: dict[str, Any], data_2: dict[str, Any], *, ignore_path_fields: bool) -> dict[str, Any]:
    ignored = {'frames', 'event_intervals'}
    if ignore_path_fields:
        ignored.add('video')

    unequal = {}
    keys = (set(data_1) | set(data_2)) - ignored
    for key in sorted(keys):
        val_1 = data_1.get(key, '<MISSING>')
        val_2 = data_2.get(key, '<MISSING>')
        if val_1 != val_2:
            unequal[key] = {'j1': val_1, 'j2': val_2}
    return {'unequal': unequal}


def _frame_map(frames: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Index frames by frame number for stable cross-file comparison."""
    return {frame.get('f'): frame for frame in frames}


def _compare_stream_frame_structure(frames_1: list[dict[str, Any]], frames_2: list[dict[str, Any]]) -> dict[str, Any]:
    map_1, map_2 = _frame_map(frames_1), _frame_map(frames_2)
    frame_count = None
    if len(frames_1) != len(frames_2):
        frame_count = {'j1': len(frames_1), 'j2': len(frames_2)}

    missing = sorted(set(map_1) - set(map_2))
    extra = sorted(set(map_2) - set(map_1))
    timestamp_mismatches = []
    detection_count_mismatches = []

    for frame_idx in sorted(set(map_1) & set(map_2)):
        frame_1, frame_2 = map_1[frame_idx], map_2[frame_idx]
        if frame_1.get('t') != frame_2.get('t'):
            timestamp_mismatches.append({'frame': frame_idx, 'j1': frame_1.get('t'), 'j2': frame_2.get('t')})

        det_count_1 = len(frame_1.get('detection_list') or [])
        det_count_2 = len(frame_2.get('detection_list') or [])
        if det_count_1 != det_count_2:
            detection_count_mismatches.append({'frame': frame_idx, 'j1': det_count_1, 'j2': det_count_2})

    return {'frame_count': frame_count,
            'missing_frame_indices': missing,
            'extra_frame_indices': extra,
            'timestamp_mismatches': timestamp_mismatches,
            'detection_count_mismatches': detection_count_mismatches}


def _bbox_iou(box_1, box_2) -> float:
    """Return IoU for two normalized XYXY boxes."""
    if len(box_1) != 4 or len(box_2) != 4:
        return 0.0
    x1 = max(float(box_1[0]), float(box_2[0]))
    y1 = max(float(box_1[1]), float(box_2[1]))
    x2 = min(float(box_1[2]), float(box_2[2]))
    y2 = min(float(box_1[3]), float(box_2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_1 = max(0.0, float(box_1[2]) - float(box_1[0])) * max(0.0, float(box_1[3]) - float(box_1[1]))
    area_2 = max(0.0, float(box_2[2]) - float(box_2[0])) * max(0.0, float(box_2[3]) - float(box_2[1]))
    union = area_1 + area_2 - inter
    return 0.0 if union <= 0.0 else inter / union


def _match_detections(dets_1: list[dict[str, Any]], dets_2: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Greedily align detections by class first, then by bbox IoU."""
    candidates = []
    for idx_1, det_1 in enumerate(dets_1):
        cls_1 = det_1.get('class')
        box_1 = det_1.get('bbox', [])
        for idx_2, det_2 in enumerate(dets_2):
            cls_2 = det_2.get('class')
            box_2 = det_2.get('bbox', [])
            candidates.append((int(cls_1 == cls_2), _bbox_iou(box_1, box_2), -idx_1, -idx_2, idx_1, idx_2))

    used_1, used_2 = set(), set()
    matches = []
    for _, _, _, _, idx_1, idx_2 in sorted(candidates, reverse=True):
        if idx_1 in used_1 or idx_2 in used_2:
            continue
        used_1.add(idx_1)
        used_2.add(idx_2)
        matches.append((idx_1, idx_2))
        if len(matches) == min(len(dets_1), len(dets_2)):
            break
    return sorted(matches)


def _compare_stream_numeric_content(frames_1: list[dict[str, Any]], frames_2: list[dict[str, Any]], tolerances: dict[str, float])\
                                    -> dict[str, Any]:
    """Compare comparable numeric fields after aligning detections by class-aware IoU."""
    map_1, map_2 = _frame_map(frames_1), _frame_map(frames_2)
    total_abs, count = 0.0, 0
    max_abs, max_path = 0.0, None

    def add_delta(path: str, val_1, val_2) -> None:
        nonlocal total_abs, count, max_abs, max_path
        delta = abs(float(val_1) - float(val_2))
        total_abs += delta
        count += 1
        if delta > max_abs:
            max_abs, max_path = delta, path

    for frame_idx in sorted(set(map_1) & set(map_2)):
        dets_1 = map_1[frame_idx].get('detection_list') or []
        dets_2 = map_2[frame_idx].get('detection_list') or []
        if len(dets_1) != len(dets_2):
            continue

        for det_idx_1, det_idx_2 in _match_detections(dets_1, dets_2):
            det_1, det_2 = dets_1[det_idx_1], dets_2[det_idx_2]
            if _is_number(det_1.get('conf')) and _is_number(det_2.get('conf')):
                add_delta(f'frames[{frame_idx}].detection_list[{det_idx_1}].conf', det_1['conf'], det_2['conf'])
            _compare_number_lists(det_1.get('bbox', []), det_2.get('bbox', []),
                                  f'frames[{frame_idx}].detection_list[{det_idx_1}].bbox', add_delta)
            _compare_number_lists(det_1.get('key_points', []), det_2.get('key_points', []),
                                  f'frames[{frame_idx}].detection_list[{det_idx_1}].key_points', add_delta)

    avg_abs = total_abs/count if count else 0.0
    return {'count': count,
            'avg_abs': avg_abs,
            'max_abs': max_abs,
            'max_path': max_path,
            'within_tolerance': (avg_abs <= float(tolerances['avg_abs']) and max_abs <= float(tolerances['max_abs']))}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_number_lists(values_1, values_2, path: str, add_delta) -> None:
    if len(values_1) != len(values_2):
        return
    for idx, (val_1, val_2) in enumerate(zip(values_1, values_2)):
        if _is_number(val_1) and _is_number(val_2):
            add_delta(f'{path}[{idx}]', val_1, val_2)


def _compare_stream_annotations(data_1: dict[str, Any], data_2: dict[str, Any]) -> dict[str, Any]:
    """Compare top-level event intervals and per-frame annotation payloads."""
    intervals_1 = _normalize_event_intervals(data_1.get('event_intervals'))
    intervals_2 = _normalize_event_intervals(data_2.get('event_intervals'))
    map_1, map_2 = _frame_map(data_1.get('frames', [])), _frame_map(data_2.get('frames', []))
    mismatches = []

    for frame_idx in sorted(set(map_1) & set(map_2)):
        ann_1 = {'group_events': map_1[frame_idx].get('group_events', []),
                 'individual_events': map_1[frame_idx].get('individual_events', [])}
        ann_2 = {'group_events': map_2[frame_idx].get('group_events', []),
                 'individual_events': map_2[frame_idx].get('individual_events', [])}
        if ann_1 != ann_2:
            mismatches.append({'frame': frame_idx, 'j1': ann_1, 'j2': ann_2})

    return {'event_intervals_equal': intervals_1 == intervals_2,
            'event_intervals': {'j1': intervals_1, 'j2': intervals_2} if intervals_1 != intervals_2 else None,
            'frame_annotation_mismatches': mismatches}
# endregion

#536(4!0,3,2)-> 373 (0,1,0) -> 336(,2,)
