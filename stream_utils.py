"""In-memory transformations for Stream JSON dictionaries."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

FPS_MEASURED_EPS = 1e-6
DEFAULT_SJ_NUMERIC_TOLERANCES = {'avg_abs': 0.05, 'max_abs': 0.05}
META_IGNORED = {'frames', 'event_intervals', 'detector', 'detection_threshold'}


def resample_fps(stream: dict, fps_rsmp: float) -> dict:
    """Resample stream frames in place and return the modified stream."""

    frames = stream.get('frames')
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError('resample_fps requires at least two stream frames')

    times = np.asarray([frame.get('t') for frame in frames], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('resample_fps requires finite, strictly increasing frame timestamps')

    fps_rsmp = float(fps_rsmp)
    actual_fps = (len(times) - 1)/(times[-1] - times[0])
    if not 0 < fps_rsmp <= actual_fps + FPS_MEASURED_EPS:
        raise ValueError(f'resample_fps {fps_rsmp:g} must be positive and no greater than actual FPS {actual_fps:.3f}')
    if abs(fps_rsmp - actual_fps) <= FPS_MEASURED_EPS:
        return stream

    period = 1.0/fps_rsmp
    targets = np.arange(times[0], times[-1] + 0.5*period, period)
    targets = targets[targets <= times[-1]]
    right = np.searchsorted(times, targets, side='left')
    right = np.clip(right, 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    selected = np.unique(selected)
    if len(selected) < 2:
        raise ValueError('resample_fps produced fewer than two unique stream frames')

    original_count = len(frames)
    frames = [frames[idx] for idx in selected]
    effective_fps = (len(frames) - 1)/(frames[-1]['t'] - frames[0]['t'])
    stream['frames'] = frames

    sampling_key = ('sampling rate' if 'sampling rate' in stream else
                    'sampling_rate' if 'sampling_rate' in stream else 'sampling rate')
    sampling = stream.get(sampling_key)
    sampling = dict(sampling) if isinstance(sampling, dict) else {}
    if 'original_sampling_rate' not in stream:
        original_sampling = dict(sampling)
        try:
            if abs(actual_fps - float(original_sampling['effective'])) > FPS_MEASURED_EPS:
                original_sampling['measured'] = actual_fps
        except (KeyError, TypeError, ValueError):
            original_sampling['measured'] = actual_fps
        stream['original_sampling_rate'] = original_sampling
    sampling.update({'target': fps_rsmp, 'effective': effective_fps})
    stream[sampling_key] = sampling
    stream['step'] = None

    counts = stream.get('frame_count')
    counts = dict(counts) if isinstance(counts, dict) else {}
    counts.setdefault('original', original_count)
    resampled = [int(match.group(1)) for key in counts
                 if (match := re.fullmatch(r'resampled_(\d+)', str(key)))]
    counts[f'resampled_{max(resampled, default=0) + 1:02d}'] = len(frames)
    counts['current'] = len(frames)
    stream['frame_count'] = counts

    return stream


def filter_yolo(stream: dict, cutoff: float) -> dict:
    """Filter stream detections in place by a higher confidence cutoff."""
    cutoff = float(cutoff)
    if not 0 < cutoff <= 1:
        raise ValueError('YOLO cutoff must be greater than 0 and no greater than 1')

    detector = stream.get('detector')
    if detector is not None and not isinstance(detector, dict):
        raise ValueError('detector must be a dictionary')
    detector = detector if isinstance(detector, dict) else {}
    filters = stream.get('yolo_filters')
    if filters is not None and not isinstance(filters, dict):
        raise ValueError('yolo_filters must be a dictionary')
    if isinstance(filters, dict):
        filters = {int(key) if str(key).isdigit() else key: value for key, value in filters.items()}

    original = detector.get('threshold')
    if original is None and filters is not None:
        original = filters.get(0)
    if original is None:
        original = stream.get('detection_threshold')
    if original is not None:
        original = float(original)
        if not np.isfinite(original):
            raise ValueError(f'invalid original YOLO threshold: {original}')

    current = stream.get('detection_threshold')
    if current is None and filters is not None:
        indices = [key for key in filters if isinstance(key, int)]
        current = filters[max(indices)] if indices else None
    if current is None:
        current = original
    if current is not None:
        current = float(current)
        if not np.isfinite(current):
            raise ValueError(f'invalid detection_threshold: {current}')
        if cutoff <= current:
            raise ValueError(f'YOLO cutoff {cutoff:g} must be higher than active threshold {current:g}')

    source_count = removed = 0
    frames = stream.get('frames')
    if not isinstance(frames, list):
        raise ValueError('stream frames must be a list')
    filtered = []
    for frame in frames:
        det_key = ('detection_list' if 'detection_list' in frame
                   else 'detections_list' if 'detections_list' in frame else None)
        if det_key is None:
            continue
        detections = frame[det_key]
        if not isinstance(detections, list):
            raise ValueError(f"frame {frame.get('f', '?')} detections must be a list")
        kept = []
        for det in detections:
            if not isinstance(det, dict) or det.get('conf') is None:
                raise ValueError(f"frame {frame.get('f', '?')} has a detection without confidence")
            confidence = float(det['conf'])
            if not np.isfinite(confidence):
                raise ValueError(f"frame {frame.get('f', '?')} has invalid detection confidence")
            if confidence >= cutoff:
                kept.append(det)
        source_count += len(detections)
        removed += len(detections) - len(kept)
        filtered.append((frame, det_key, kept))

    if detector.get('threshold') is None and original is not None:
        detector['threshold'] = original
        stream['detector'] = detector
    if filters is None:
        filters = {0: original}
        stream['yolo_filters'] = filters
    else:
        stream['yolo_filters'] = filters
    if 0 not in filters:
        filters[0] = original
    indices = [int(key) for key in filters if str(key).isdigit()]
    filter_idx = max(indices, default=0) + 1
    filters[filter_idx] = cutoff
    for frame, det_key, kept in filtered:
        frame[det_key] = kept
    stream['detection_threshold'] = cutoff

    counts = stream.get('detection_count')
    counts = dict(counts) if isinstance(counts, dict) else {}
    counts.setdefault('original', source_count)
    counts[f'filtered_{filter_idx:02d}'] = source_count - removed
    counts['current'] = source_count - removed
    stream['detection_count'] = counts

    return stream


def stream_duration(frames: list[dict[str, Any]]) -> float:
    """Return the timestamp span covered by stream frames."""
    if len(frames) < 2:
        return 0.0
    return max(0.0, float(frames[-1].get('t', 0.0)) - float(frames[0].get('t', 0.0)))


def event_durations(frames: list[dict[str, Any]], tag:int|None = None) -> list[float]:
    """ Return merged durations for one group-event tag or empty-frame spans."""
    if not frames:
        return []

    t_ls, tail_dt = _frame_delta_stats(frames)
    durations = []
    start_i = prev_i = None
    for i, frm in enumerate(frames):
        group_events = frm.get('group_events') or []
        active = not group_events if tag is None else tag in group_events
        if active and start_i is None:
            start_i = i
        if active:
            prev_i = i
            continue
        if start_i is not None and prev_i is not None:
            durations.append(max(0.0, t_ls[prev_i] - t_ls[start_i]) + tail_dt)
            start_i = prev_i = None

    if start_i is not None and prev_i is not None:
        durations.append(max(0.0, t_ls[prev_i] - t_ls[start_i]) + tail_dt)
    return durations


def _frame_delta_stats(frames: list[dict[str, Any]]) -> tuple[list[float], float]:

    t_ls = [float(frm.get('t', 0.0)) for frm in frames]
    deltas = [max(0.0, t_ls[i + 1] - t_ls[i]) for i in range(len(t_ls) - 1)]
    positive = [d for d in deltas if d > 0.0]
    return t_ls, float(np.median(positive)) if positive else 0.0


#* region Stream comparison *****************************#

def compare_meta(stream_1: dict, stream_2: dict, *, ignore_path_fields=True) -> dict[str, Any]:
    """Compare Stream JSON header metadata excluding frame payloads."""
    ignored = set(META_IGNORED)
    if ignore_path_fields:
        ignored.add('video')

    unequal = {}
    for key in sorted((set(stream_1) | set(stream_2)) - ignored):
        value_1 = stream_1.get(key, '<MISSING>')
        value_2 = stream_2.get(key, '<MISSING>')
        if value_1 != value_2:
            unequal[key] = {'j1': value_1, 'j2': value_2}
    return {'unequal': unequal}


def _frame_map(frames: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Index frames by frame number for stable comparison."""
    return {f.get('f'): f for f in frames}


def _cmp_frame_structure(frames_1: list[dict[str, Any]], frames_2: list[dict[str, Any]]) -> dict[str, Any]:
    map_1, map_2 = _frame_map(frames_1), _frame_map(frames_2)
    frame_count = {'j1': len(frames_1), 'j2': len(frames_2)} if len(frames_1) != len(frames_2) else None
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
    x1, y1 = max(float(box_1[0]), float(box_2[0])), max(float(box_1[1]), float(box_2[1]))
    x2, y2 = min(float(box_1[2]), float(box_2[2])), min(float(box_1[3]), float(box_2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_1 = max(0.0, float(box_1[2]) - float(box_1[0])) * max(0.0, float(box_1[3]) - float(box_1[1]))
    area_2 = max(0.0, float(box_2[2]) - float(box_2[0])) * max(0.0, float(box_2[3]) - float(box_2[1]))
    union = area_1 + area_2 - inter
    return 0.0 if union <= 0.0 else inter/union


def _match_detections(dets_1: list[dict[str, Any]], dets_2: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Greedily align detections by class first, then by bbox IoU."""

    candidates = []
    for i_1, det_1 in enumerate(dets_1):
        for i_2, det_2 in enumerate(dets_2):
            candidates.append((int(det_1['class'] == det_2['class']),
                               _bbox_iou(det_1['bbox'], det_2['bbox']), -i_1, -i_2, i_1, i_2))
                               # _bbox_iou(det_1.get('bbox', []), det_2.get('bbox', [])),
                               # -i_1, -i_2, i_1, i_2))

    used_1, used_2, matches = set(), set(), []
    for _, _, _, _, i_1, i_2 in sorted(candidates, reverse=True):
        if i_1 in used_1 or i_2 in used_2:
            continue
        used_1.add(i_1)
        used_2.add(i_2)
        matches.append((i_1, i_2))
        if len(matches) == min(len(dets_1), len(dets_2)):
            break
    return sorted(matches)


def _cmp_numeric(frames_1: list[dict[str, Any]], frames_2: list[dict[str, Any]], tolerances: dict[str, float]) -> dict[str, Any]:
    """Compare numeric detection fields after class-aware IoU alignment."""

    def _is_number(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    map_1, map_2 = _frame_map(frames_1), _frame_map(frames_2)
    total_abs = count = 0
    max_abs, max_path = 0.0, None

    def add_delta(path: str, value_1, value_2) -> None:
        nonlocal total_abs, count, max_abs, max_path
        delta = abs(float(value_1) - float(value_2))
        total_abs += delta
        count += 1
        if delta > max_abs:
            max_abs, max_path = delta, path

    def cmp_numbers(values_1, values_2, path: str) -> None:
        if len(values_1) == len(values_2):
            for idx, (val_1, val_2) in enumerate(zip(values_1, values_2)):
                if _is_number(val_1) and _is_number(val_2):
                    add_delta(f'{path}[{idx}]', val_1, val_2)

    for frame_idx in sorted(set(map_1) & set(map_2)):
        dets_1 = map_1[frame_idx].get('detection_list') or []
        dets_2 = map_2[frame_idx].get('detection_list') or []
        if len(dets_1) != len(dets_2):
            continue
        for i_det1, i_det2 in _match_detections(dets_1, dets_2):
            det_1, det_2 = dets_1[i_det1], dets_2[i_det2]
            if _is_number(det_1.get('conf')) and _is_number(det_2.get('conf')):
                add_delta(f'frames[{frame_idx}].detection_list[{i_det1}].conf', det_1['conf'], det_2['conf'])
            cmp_numbers(det_1.get('bbox', []), det_2.get('bbox', []),
                        f'frames[{frame_idx}].detection_list[{i_det1}].bbox')
            cmp_numbers(det_1.get('key_points', []), det_2.get('key_points', []),
                        f'frames[{frame_idx}].detection_list[{i_det1}].key_points')

    avg_abs = total_abs/count if count else 0.0
    return {'count': count,
            'avg_abs': avg_abs,
            'max_abs': max_abs,
            'max_path': max_path,
            'tolerances': dict(tolerances),
            'within_tolerance': (avg_abs <= float(tolerances['avg_abs']) and
                                 max_abs <= float(tolerances['max_abs']))}


def _cmp_annotations(stream_1: dict, stream_2: dict) -> dict[str, Any]:
    def intervals(value):
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
        if not isinstance(value, dict):
            return {}
        normalized = {}
        for key, payload in value.items():
            if isinstance(payload, dict):
                sec_intervals = payload.get('sec', [])
            elif isinstance(payload, list):
                sec_intervals = payload
            else:
                sec_intervals = []
            if sec_intervals:
                normalized[key] = sec_intervals
        return normalized

    intervals_1 = intervals(stream_1.get('event_intervals'))
    intervals_2 = intervals(stream_2.get('event_intervals'))
    map_1 = _frame_map(stream_1.get('frames', []))
    map_2 = _frame_map(stream_2.get('frames', []))
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


def compare_stream(stream_1: dict, stream_2: dict, *, tolerances=None) -> dict[str, Any]:
    """Compare Stream JSON frame, annotation, and numeric payloads."""
    if tolerances is None:
        tolerances = dict(DEFAULT_SJ_NUMERIC_TOLERANCES)
    elif isinstance(tolerances, dict):
        tolerances = {**DEFAULT_SJ_NUMERIC_TOLERANCES, **tolerances}
    else:
        raise TypeError('compare_stream tolerances must be None or a concrete dict')

    frames_1 = stream_1.get('frames', [])
    frames_2 = stream_2.get('frames', [])
    return {'frame_structure': _cmp_frame_structure(frames_1, frames_2),
            'numeric': _cmp_numeric(frames_1, frames_2, tolerances),
            'annotations': _cmp_annotations(stream_1, stream_2)}

#* endregion *#
