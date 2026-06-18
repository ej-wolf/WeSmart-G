"""Stream JSON utilities for inspection, comparison, and pair conversion."""
from __future__ import annotations
import argparse
import json, csv
from pathlib import Path
from typing import Any
import numpy as np
#* project import
from common.my_local_utils import print_color, get_unique_name
from json_utils import list_json_sources, load_json_raw, resolve_json_source

JSON_SUFFIX = '.json'
CSV_SUFFIX = '.csv'
NPZ_SUFFIX = '.npz'
MINIMAL_DETECTOR = {'model': 'npz_import', 'version': None, 'source': 'out_alex_pair'}
STREAM_INFO_REPORT = 'stream_json_info.json'
STREAM_INFO_SUMMARY = 'stream_json_info.csv'
FPS_TOLERANCES = 0.2
SJ_EVENT_BUCKETS = {'empty': None, 'norm': 0, 'abnormal': 1, 'tension': 3, 'fight': 4}
DEFAULT_STREAM_NUMERIC_TOLERANCES = {'avg_abs': 0.05, 'max_abs': 0.05}
META_IGNORED = {'frames', 'event_intervals', 'detector', 'detection_threshold'}


#* region Stream JSON info *****************************#
def _frame_delta_stats(frames: list[dict[str, Any]]) -> tuple[list[float], float]:
    times = [float(frame.get('t', 0.0)) for frame in frames]
    deltas = [max(0.0, times[idx + 1] - times[idx]) for idx in range(len(times) - 1)]
    positive = [delta for delta in deltas if delta > 0.0]
    tail_dt = float(np.median(positive)) if positive else 0.0
    return times, tail_dt


def _merged_bucket_durations(frames: list[dict[str, Any]], bucket: int | None) -> list[float]:
    if not frames:
        return []

    times, tail_dt = _frame_delta_stats(frames)
    durations = []
    start_idx = None
    prev_idx = None

    for idx, frame in enumerate(frames):
        grp_evn = frame.get('group_events') or []
        active = (len(grp_evn) == 0) if bucket is None else (bucket in grp_evn)
        if active and start_idx is None:
            start_idx = idx
        if active:
            prev_idx = idx
            continue
        if start_idx is not None and prev_idx is not None:
            durations.append(max(0.0, times[prev_idx] - times[start_idx]) + tail_dt)
            start_idx, prev_idx = None, None

    if start_idx is not None and prev_idx is not None:
        durations.append(max(0.0, times[prev_idx] - times[start_idx]) + tail_dt)
    return durations


def print_stream_json_info(report: dict[str, Any], **kwargs) -> None:
    """Print one compact SJ inspection report."""
    if not kwargs.get('print_cli', True):
        return

    fps_info = report['fps']
    print(f"\n======== Stream JSON Info ======================")
    print(f"Count             : {report['sj_count']}")
    print(f"Total duration    : {report['duration']['total']:.2f} s")
    print(f"Avg/std duration  : {report['duration']['avg']:.2f}  ({report['duration']['std']:.2f}) s")
    print(f"Min Max duration  : {report['duration']['min']:.2f} - {report['duration']['max']:.2f} s")
    print(f"Total frames      : {report['frames']['total']}")
    print(f"Avg. frame count  : {report['frames']['avg']:.2f} s")
    print("\nEvent durations")
    print(f"{'tag':10} | {'total(s)':>10} | {'avg seg(s)':>10} | {'segments':>8}")
    print("-" * 48)
    for tag in tuple(SJ_EVENT_BUCKETS):
        row = report['group_events'][tag]
        print(f"{tag:10} | {row['total']:10.2f} | {row['avg']:10.2f} | {row['count']:8d}")


def stream_json_info(sj_path, op_path=None, **kwargs) -> dict[str, Any]:
    """Inspect one SJ directory or list of SJ files and return aggregated info."""

    def _resolve_output_path(op_path: str | Path | None, save_format: str) -> Path | None:
        if op_path is None:
            return None
        op_path = Path(op_path)
        default_name = STREAM_INFO_SUMMARY if save_format == 'csv' else STREAM_INFO_REPORT
        default_suffix = CSV_SUFFIX if save_format == 'csv' else JSON_SUFFIX
        if op_path.is_dir():
            name = op_path / default_name
        elif op_path.suffix.lower() in {JSON_SUFFIX, CSV_SUFFIX}:
            name = op_path
        else:
            name = op_path.with_suffix(default_suffix)
        return get_unique_name(name)

    def _summary_rows(report: dict[str, Any]) -> list[list[Any]]:
        fps = report['fps']
        rows = [['metric', 'value', 'value_2', 'unit'],
                ['Count', report['sj_count'], '', ''],
                ['Total duration', f"{report['duration']['total']:.2f}", '', 's'],
                ['Avg/std duration', f"{report['duration']['avg']:.2f}", f"{report['duration']['std']:.2f}", 's'],
                ['Min/max duration', f"{report['duration']['min']:.2f}", f"{report['duration']['max']:.2f}", 's'],
                ['Total frames', report['frames']['total'], '', ''],
                ['Avg. frame count', f"{report['frames']['avg']:.2f}", '', ''],
                ['Avg/min/max fps', f"{fps['avg']:.3f}", f"{fps['min']:.3f} / {fps['max']:.3f}", ''],
                ['FPS within eps', fps['fps_within_epsilon'], fps['fps_epsilon'], ''],
                [],
                ['tag', 'total(s)', 'avg seg(s)', 'segments']]
        for tag in SJ_EVENT_BUCKETS:
            row = report['group_events'][tag]
            rows.append([tag, f"{row['total']:.2f}", f"{row['avg']:.2f}", row['count']])
        return rows

    def _save_report() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print_color(out_path, 'g')
        if save_format == 'json':
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        elif save_format == 'csv':
            with out_path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(_summary_rows(report))

    def _load_sj_list(sj_path) -> list[Path]:
        if isinstance(sj_path, (list, tuple, set)):
            paths = [resolve_json_source(p) for p in sj_path]
            if not paths:
                raise ValueError('No stream JSON files were provided')
            return paths

        sj_path = Path(sj_path)
        if sj_path.is_dir():
            paths = list_json_sources(sj_path)
            if not paths:
                raise FileNotFoundError(f'No stream JSON files found in {sj_path}')
            return paths
        try:
            return [resolve_json_source(sj_path)]
        except FileNotFoundError:
            pass
        raise FileNotFoundError(sj_path)

    def _stream_duration(frames: list[dict[str, Any]]) -> float:
        if len(frames) < 2:
            return 0.0
        return max(0.0, float(frames[-1].get('t', 0.0)) - float(frames[0].get('t', 0.0)))

    def _ensure_stream_json(data: dict[str, Any], src: Path) -> list[dict[str, Any]]:
        frms = data.get('frames')
        if not isinstance(frms, list):
            raise ValueError(f'missing frames list in {src}')
        return frms

    def _duration_stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {'total': 0.0, 'avg': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        return {'total': float(arr.sum()), 'avg': float(arr.mean()), 'std': float(arr.std()),
                'max': float(arr.max()), 'min': float(arr.min())}

    sj_files = _load_sj_list(sj_path)
    fps_epsilon = float(kwargs.get('fps_epsilon', FPS_TOLERANCES))

    durations, frame_counts, fps_values = [], [], []
    bucket_segments = {name: [] for name in SJ_EVENT_BUCKETS}
    valid_files, bad_files = [], []

    for path in sj_files:
        try:
            data = load_json_raw(path)
            frames = _ensure_stream_json(data, path)
            durations.append(_stream_duration(frames))
            frame_counts.append(len(frames))
            fps_values.append(float(data.get('fps', 0.0) or 0.0))
            for tag_name, bucket in SJ_EVENT_BUCKETS.items():
                bucket_segments[tag_name].extend(_merged_bucket_durations(frames, bucket))
            valid_files.append(str(path))
        except Exception as exc:
            print_color(f"[ERROR] stream_json_info skipped {path}: {type(exc).__name__}: {exc}", 'r')
            bad_files.append({'file': str(path), 'error': f'{type(exc).__name__}: {exc}'})

    duration = _duration_stats(durations)
    frm_arr = np.asarray(frame_counts, dtype=np.float64)
    fps_arr = np.asarray(fps_values, dtype=np.float64)
    group_events = {}
    for tag_name, seg_durations in bucket_segments.items():
        stats = _duration_stats(seg_durations)
        group_events[tag_name] = {'total': stats['total'],
                                  'avg': stats['avg'],
                                  'count': len(seg_durations)}

    report = {'sj_count': len(valid_files),
              'files': valid_files,
              'bad_files': bad_files,
              'duration': duration,
              'frames': {'total': int(frm_arr.sum()) if frm_arr.size else 0,
                         'avg': float(frm_arr.mean()) if frm_arr.size else 0.0},
              'fps': {'avg': float(fps_arr.mean()) if fps_arr.size else 0.0,
                      'min': float(fps_arr.min()) if fps_arr.size else 0.0,
                      'max': float(fps_arr.max()) if fps_arr.size else 0.0,
                      'fps_epsilon': fps_epsilon,
                      'fps_within_epsilon': (fps_arr.max() - fps_arr.min()) <= fps_epsilon if fps_arr.size else True},
              'group_events': group_events}

    save_format = kwargs.get('save_format', 'csv')
    if save_format.lower() not in {'csv', 'json'}:
        raise ValueError(f"Unsupported stream_json_info save_format: {save_format}")

    out_path = _resolve_output_path(op_path, save_format)
    if out_path is not None:
        _save_report()
    print_stream_json_info(report, **kwargs)
    return report

#* endregion *#


#* region Stream JSON compare **************************#
def compare_stream_json(j1, j2, *, tolerances=None, ignore_path_fields=True) -> tuple[bool, dict[str, Any]]:
    """Compare two stream JSONs by metadata, frame layout, annotations, and numeric payload."""
    if tolerances is None:
        tolerances = dict(DEFAULT_STREAM_NUMERIC_TOLERANCES)
    elif isinstance(tolerances, dict):
        tolerances = {**DEFAULT_STREAM_NUMERIC_TOLERANCES, **tolerances}
    else:
        raise TypeError("compare_stream_json tolerances must be None or a concrete dict")
    data_1 = j1 if isinstance(j1, dict) else load_json_raw(j1)
    data_2 = j2 if isinstance(j2, dict) else load_json_raw(j2)

    metadata = _compare_stream_metadata(data_1, data_2, ignore_path_fields=ignore_path_fields)
    frame_structure = _cmpr_frm_structure(data_1.get('frames', []), data_2.get('frames', []))
    numeric = _cmp_numeric_content(data_1.get('frames', []), data_2.get('frames', []), tolerances)
    annotations = _compare_annotations(data_1, data_2)

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


def _normalize_event_intervals(value):
    """Normalize event intervals to meaningful non-empty sec-only values."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        return {}

    intervals = {}
    for key, payload in value.items():
        if isinstance(payload, dict):
            sec_intervals = payload.get('sec', [])
        elif isinstance(payload, list):
            sec_intervals = payload
        else:
            sec_intervals = []
        if sec_intervals:
            intervals[key] = sec_intervals
    return intervals


def _compare_stream_metadata(data_1: dict[str, Any], data_2: dict[str, Any], *, ignore_path_fields: bool) -> dict[str, Any]:
    ignored = set(META_IGNORED)
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


def _cmpr_frm_structure(frames_1: list[dict[str, Any]], frames_2: list[dict[str, Any]]) -> dict[str, Any]:
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


def _cmp_numeric_content(frames_1: list[dict[str, Any]], frames_2: list[dict[str, Any]], tolerances: dict[str, float]) -> dict[str, Any]:
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

    avg_abs = total_abs / count if count else 0.0
    return {'count': count,
            'avg_abs': avg_abs,
            'max_abs': max_abs,
            'max_path': max_path,
            'tolerances': dict(tolerances),
            'within_tolerance': (avg_abs <= float(tolerances['avg_abs']) and max_abs <= float(tolerances['max_abs']))}


def _is_number(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _compare_number_lists(values_1, values_2, path: str, add_delta) -> None:
    if len(values_1) != len(values_2):
        return
    for idx, (val_1, val_2) in enumerate(zip(values_1, values_2)):
        if _is_number(val_1) and _is_number(val_2):
            add_delta(f'{path}[{idx}]', val_1, val_2)


def _compare_annotations(data_1: dict[str, Any], data_2: dict[str, Any]) -> dict[str, Any]:
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

#* endregion *#


#* region Convert (npz, json) pair to stream JSON  ***************#
def _scalar(value: Any):
    """Convert numpy scalar-like values into plain Python values."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(()).item()
    return value


def _step_value(meta: dict[str, Any]) -> int | None:
    timing = meta.get('timing', {}) or {}
    fps = meta.get('fps')
    rate = timing.get('sampling_rate_hz')
    if timing.get('sampling_mode') == 'union_rates':
        return None
    if fps and rate:
        return int(round(fps / rate))
    return None


def _warn_mismatch(name: str, left: Any, right: Any):
    if left != right:
        print_color(f"[WARN] pair metadata mismatch for {name}: json={left!r}, npz={right!r}", 'y')


def _validate_pair(meta: dict[str, Any], npz_data, stem: str):
    required = {'frame_indices', 'frame_times_sec', 'group_events', 'group_event_counts',
                'person_counts', 'classes', 'confidences', 'bboxes', 'keypoints'}
    missing = sorted(required - set(npz_data.files))
    if missing:
        raise KeyError(f"{stem}: missing required npz arrays: {missing}")

    n_frames = len(npz_data['frame_indices'])
    frame_arrays = ('frame_times_sec', 'group_events', 'group_event_counts', 'person_counts',
                    'classes', 'confidences', 'bboxes', 'keypoints')
    for key in frame_arrays:
        if npz_data[key].shape[0] != n_frames:
            raise ValueError(f"{stem}: {key} length mismatch: {npz_data[key].shape[0]} vs {n_frames}")

    person_slots = npz_data['classes'].shape[1]
    slot_shapes = {'confidences': npz_data['confidences'].shape[1],
                   'bboxes': npz_data['bboxes'].shape[1],
                   'keypoints': npz_data['keypoints'].shape[1]}
    for key, width in slot_shapes.items():
        if width != person_slots:
            raise ValueError(f"{stem}: {key} slot mismatch: {width} vs {person_slots}")

    if np.any(npz_data['person_counts'] > person_slots):
        raise ValueError(f"{stem}: person_counts exceeds person slot width {person_slots}")

    group_width = npz_data['group_events'].shape[1]
    if np.any(npz_data['group_event_counts'] > group_width):
        raise ValueError(f"{stem}: group_event_counts exceeds group event width {group_width}")

    _warn_mismatch('video', meta.get('video'), _scalar(npz_data['video']) if 'video' in npz_data.files else None)
    _warn_mismatch('fps', meta.get('fps'), _scalar(npz_data['fps']) if 'fps' in npz_data.files else None)
    _warn_mismatch('duration_sec', meta.get('duration_sec'),
                   _scalar(npz_data['duration_sec']) if 'duration_sec' in npz_data.files else None)
    _warn_mismatch('frame_width', meta.get('frame_width'),
                   _scalar(npz_data['frame_width']) if 'frame_width' in npz_data.files else None)
    _warn_mismatch('frame_height', meta.get('frame_height'),
                   _scalar(npz_data['frame_height']) if 'frame_height' in npz_data.files else None)


def _frame_detections(npz_data, frame_idx: int) -> list[dict[str, Any]]:
    person_count = int(npz_data['person_counts'][frame_idx])
    dets = []
    for det_idx in range(person_count):
        dets.append({'class': int(npz_data['classes'][frame_idx, det_idx]),
                     'conf': float(npz_data['confidences'][frame_idx, det_idx]),
                     'bbox': npz_data['bboxes'][frame_idx, det_idx].tolist(),
                     'key_points': npz_data['keypoints'][frame_idx, det_idx].tolist()})
    return dets


def _build_frames(npz_data) -> list[dict[str, Any]]:
    frames = []
    for row_idx, frame_no in enumerate(npz_data['frame_indices']):
        event_count = int(npz_data['group_event_counts'][row_idx])
        group_events = npz_data['group_events'][row_idx]
        group_tags = [int(v) for v in group_events[:event_count] if v != 0]
        frames.append({'f': int(frame_no),
                       't': float(npz_data['frame_times_sec'][row_idx]),
                       'individual_events': [],
                       'group_events': sorted(set(group_tags), reverse=True),
                       'detection_list': _frame_detections(npz_data, row_idx)})
    return frames


def pair_to_stream_json(npz_path, json_path, out_path=None, **kwargs) -> dict[str, Any]:
    """Convert one HMC pair into one stream-JSON dict and optionally save it."""
    json_path, npz_path = Path(json_path), Path(npz_path)

    with json_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    npz_data = np.load(npz_path, allow_pickle=True)
    stem = json_path.stem
    try:
        _validate_pair(meta, npz_data, stem)

        def _evn_intervals(event_intervals: dict[str, Any] | None) -> dict[str, dict[str, list]]:
            out = {}
            for key, payload in (event_intervals or {}).items():
                sec_intervals = payload.get('sec', []) if isinstance(payload, dict) else payload
                out[str(key)] = {'sec': list(sec_intervals or [])}
            return out

        timing = meta.get('timing', {}) or {}
        target_rate = timing.get('sampling_rates_hz')
        if target_rate is None:
            target_rate = timing.get('sampling_rate_hz')
        data = {'video': meta.get('video') or _scalar(npz_data['video']),
                'fps': meta.get('fps') if meta.get('fps') is not None else _scalar(npz_data['fps']),
                'sampling rate': {'target': target_rate,
                                  'effective': timing.get('effective_fps')},
                'step': _step_value(meta),
                'detector': dict(MINIMAL_DETECTOR),
                'event_intervals': _evn_intervals(meta.get('event_intervals')),
                'frames': _build_frames(npz_data)}
    finally:
        npz_data.close()

    dst = None
    if out_path is not None:
        out_path = Path(out_path)
        dst = out_path if out_path.suffix.lower() == JSON_SUFFIX else out_path / f"{stem}.json"

    if dst is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def convert_pair_dir(pair_dir, out_dir=None, **kwargs) -> list[Path]:
    """Convert every matched json+npz stem in one directory."""
    pair_dir = Path(pair_dir)
    json_stems = {p.stem: p for p in sorted(pair_dir.glob(f'*{JSON_SUFFIX}'))}
    npz_stems = {p.stem: p for p in sorted(pair_dir.glob(f'*{NPZ_SUFFIX}'))}
    stems = sorted(set(json_stems) & set(npz_stems))
    if not stems:
        raise FileNotFoundError(f"No matched json+npz pairs found in {pair_dir}")

    out_dir = pair_dir if out_dir is None else Path(out_dir)
    out_paths = []
    for stem in stems:
        dst = out_dir / f'{stem}.json'
        pair_to_stream_json(npz_stems[stem], json_stems[stem], out_path=dst, **kwargs)
        out_paths.append(dst)
    return out_paths

#* endregion *#


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_dir', type=Path, help='directory containing matched .json/.npz stems')
    parser.add_argument('-o', '--out-dir', type=Path, default=None,
                        help='output directory for converted stream JSON files')
    return parser

#349 -#536(4!0,3,2)-> 373 (0,1,0) -> 336(,2,)
#600(1,9,1)
if __name__ == '__main__':
    pass
