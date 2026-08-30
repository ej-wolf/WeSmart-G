"""Stream JSON utilities for inspection, comparison, and pair conversion."""
from __future__ import annotations
import argparse
import copy
import io
import json, csv
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
#* project import
from common.my_local_utils import print_color, get_unique_name
from json_utils import list_json_sources, load_json_raw, resolve_json_source, save_json_raw
from stream_utils import compare_meta, compare_stream, event_durations, stream_duration

JSON_SUFFIX = '.json'
CSV_SUFFIX = '.csv'
NPZ_SUFFIX = '.npz'

TAG_NO_EVENT = 0
TAG_ABNORMAL = 1
TAG_FALL    = 2
TAG_TENSION = 3
TAG_FIGHT   = 4

SJ_EVENT_BUCKETS = {'empty': None, 'norm': TAG_NO_EVENT, 'abnormal': TAG_ABNORMAL,
                    'tension': TAG_TENSION, 'fight': TAG_FIGHT}
SJ_DRAW_RANDOM_SEED = 66


#* region Stream JSON info *****************************#

FPS_TOLERANCES = 0.2
SJ_INFO_REPORT  = 'stream_json_info.json'
SJ_INFO_SUMMARY = 'stream_json_info.csv'
SJ_META_INFO = 'stream_meta_info.json'
DEFAULT_STREAM_META = Path.cwd() / 'work_dirs' / SJ_META_INFO


def stream_stem(path: str | Path) -> str:
    """Return the dot-free canonical identity of a stream-related file."""
    return Path(path).name.split('.', 1)[0]


def load_stream_inputs(inputs) -> tuple[list[tuple[str, dict]], str, list[dict]]:
    """Load homogeneous stream dictionaries or JSON sources for stream testing."""
    if inputs is None:
        return [], 'streams', []
    items = list(inputs) if isinstance(inputs, (list, tuple, set, frozenset)) else [inputs]
    has_dict = [isinstance(item, dict) for item in items]
    if any(has_dict) and not all(has_dict):
        raise ValueError("stream inputs cannot mix stream dictionaries and paths")

    if all(has_dict):
        streams = []
        for index, data in enumerate(items):
            name = Path(str(data.get('video') or f"stream_{index + 1}")).name
            streams.append((name, copy.deepcopy(data)))
        return streams, 'streams', []

    paths = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(list_json_sources(path))
        else:
            paths.append(path)

    streams, failures = [], []
    for path in paths:
        try:
            streams.append((path.name, load_json_raw(path)))
        except Exception as exc:
            failures.append({'stream': path.name, 'reason': 'bad data',
                             'error': f'{type(exc).__name__}: {exc}'})
    if len(items) == 1 and Path(items[0]).is_dir():
        source_name = Path(items[0]).name
    elif len(paths) == 1:
        source_name = paths[0].stem
    else:
        source_name = 'streams'
    return streams, source_name, failures


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

    def _resolve_output_path(op_path: str | Path | None, save_format: str)-> Path|None:
        if op_path is None:
            return None
        op_path = Path(op_path)
        default_name = SJ_INFO_SUMMARY if save_format == 'csv' else SJ_INFO_REPORT
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
                # ['FPS within eps', fps['fps_within_epsilon'], fps['fps_epsilon'], ''],
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
            durations.append(stream_duration(frames))
            frame_counts.append(len(frames))
            fps_values.append(float(data.get('fps', 0.0) or 0.0))
            for tag_name, bucket in SJ_EVENT_BUCKETS.items():
                bucket_segments[tag_name].extend(event_durations(frames, bucket))
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
    # return report


def collect_stream_meta(sj_path, op_path=None) -> dict[str, Any]:
    """Collect extendable per-stream metadata from plain or zipped SJ files."""
    def _resolve_eff_fps(data: dict[str, Any]) -> float:
        def valid(val) -> float | None:
            return val if np.isfinite(val) and val > 0.0 else None

        #* common format
        sampling = data.get('sampling rate', data.get('sampling_rate'))
        if isinstance(sampling, dict):
            value = valid(sampling.get('effective'))
            if value is not None:
                return value

        timing = data.get('timing')
        value = valid(timing.get('sampling_rate_hz')) if isinstance(timing, dict) else None
        if value is not None:
            return value
        #* legacy format
        video_fps = valid(data.get('fps'))
        step = valid(data.get('step'))
        if video_fps is not None and step is not None:
            return video_fps / step
        raise ValueError('missing valid effective FPS')

    def _entry_id(stem:str, fps:float| None, ylth:float|None) -> tuple:
        return (stem,
                None if fps is None else float(fps),
                None if ylth is None else float(ylth))

    if isinstance(sj_path, (list, tuple, set)):
        sj_files = [resolve_json_source(path) for path in sj_path]
        if not sj_files:
            raise ValueError('No stream JSON files were provided')
    else:
        sj_path = Path(sj_path)
        if sj_path.is_dir():
            sj_files = list_json_sources(sj_path)
            if not sj_files:
                raise FileNotFoundError(f'No stream JSON files found in {sj_path}')
        else:
            sj_files = [resolve_json_source(sj_path)]

    records, existing_ids = [], set()
    records_changed = False
    output_path = Path(op_path) if op_path is not None else DEFAULT_STREAM_META
    output_path = (output_path / SJ_META_INFO
                   if output_path.is_dir() or output_path.suffix == '' else output_path)
    if output_path.is_file():
        with output_path.open('r', encoding='utf-8') as file:
            saved = json.load(file)
        saved_records = saved.get('streams', []) if isinstance(saved, dict) else []
        if not isinstance(saved_records, list):
            raise ValueError(f'Invalid streams list in {output_path}')
        for record in saved_records:
            if not isinstance(record, dict):
                continue
            record = dict(record)
            record.pop('key', None)
            if record.get('stem') is None:
                records.append(record)
                continue
            canonical_stem = stream_stem(record['stem'])
            records_changed = records_changed or canonical_stem != record['stem']
            record['stem'] = canonical_stem
            entry_id = _entry_id(record['stem'], record.get('fps'),
                                 record.get('yolo_threshold'))
            if entry_id not in existing_ids:
                records.append(record)
                existing_ids.add(entry_id)
            else:
                records_changed = True

    added, skipped, bad_files = [], [], []
    for path in sj_files:
        try:
            data = load_json_raw(path)
            frames = data.get('frames')
            if not isinstance(frames, list):
                raise ValueError('missing frames list')

            stem = stream_stem(path)
            fps = _resolve_eff_fps(data)
            ylth = data.get('detection_threshold')
            if ylth is None:
                detector = data.get('detector')
                ylth = detector.get('threshold') if isinstance(detector, dict) else None
            ylth = float(ylth) if ylth is not None else None

            counts = []
            for frame in frames:
                detections = frame.get('bbs_list_of_keypoints')
                if detections is None:
                    detections = frame.get('detection_list', frame.get('detections_list', []))
                counts.append(len(detections) if isinstance(detections, (list, tuple)) else 0)

            entry_id = _entry_id(stem, fps, ylth)
            if entry_id in existing_ids:
                skipped.append({'stem': stem, 'fps': fps, 'yolo_threshold': ylth})
                continue

            longest_run = current_run = 0
            for count in counts:
                current_run = current_run + 1 if count > 0 else 0
                longest_run = max(longest_run, current_run)

            duration = (max(0.0, float(frames[-1].get('t', 0.0))
                           - float(frames[0].get('t', 0.0))) if len(frames) > 1 else 0.0)
            record = {'stem': stem, 'fps': fps, 'yolo_threshold': ylth,
                      'duration': duration,
                      'frames': len(frames),
                      'person_dets': sum(counts), 'max_dets_frame': max(counts, default=0),
                      'consecutive_det_frames': longest_run}
            records.append(record)
            existing_ids.add(entry_id)
            added.append(record)
        except Exception as error:
            print_color(f'[WARN] collect_stream_meta skipped {path}: '
                        f'{type(error).__name__}: {error}', 'y')
            bad_files.append({'file': str(path),
                              'error': f'{type(error).__name__}: {error}'})

    if output_path is not None and (added or records_changed):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'streams': records}
        with output_path.open('w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    return {'path': str(output_path) if output_path is not None else None,
            'added': added,
            'skipped': skipped,
            'bad_files': bad_files,
            'count': len(records)}


def draw_json_streams(json_path, criteria, cutoff: float, **kwargs) -> tuple[list[Path], dict[str, float]]:
    """ Draw random SJs until the selected duration criteria reaches
        cutoff within tolerance."""

    def _load_sj_list(p) -> list[Path]:
        if isinstance(p, (list, tuple, set)):
            paths = [resolve_json_source(p) for p in p]
            if not paths:
                raise ValueError('No stream JSON files were provided')
            return paths
        p = Path(p)
        if p.is_dir():
            paths = list_json_sources(p)
            if not paths:
                raise FileNotFoundError(f'No stream JSON files found in {p}')
            return paths
        try:
            return [resolve_json_source(p)]
        except FileNotFoundError:
            pass
        raise FileNotFoundError(p)

    def _normalize_criteria(crt) -> tuple[list[str | int], list[Any]]:
        crt_ls = list(crt) if isinstance(crt, (list, tuple, set)) else [crt]
        crt_ls = ['t_total' if crit == 'total' else crit for crit in crt_ls]
        valid = {'t_total', 'empty', TAG_NO_EVENT, TAG_ABNORMAL, TAG_FALL, TAG_TENSION, TAG_FIGHT}
        return [crit for crit in crt_ls if crit in valid], [crit for crit in crt_ls if crit not in valid]

    def _file_times(path: Path) -> dict[str, Any]:
        path = resolve_json_source(path)
        data = load_json_raw(path)
        frames = data.get('frames')
        if not isinstance(frames, list):
            raise ValueError('missing frames list')

        parts = {'t_total': stream_duration(frames),
                 'empty': sum(event_durations(frames, None)),
                 TAG_NO_EVENT: sum(event_durations(frames, TAG_NO_EVENT)),
                 TAG_ABNORMAL: sum(event_durations(frames, TAG_ABNORMAL)),
                 TAG_FALL: sum(event_durations(frames, TAG_FALL)),
                 TAG_TENSION: sum(event_durations(frames, TAG_TENSION)),
                 TAG_FIGHT: sum(event_durations(frames, TAG_FIGHT))}
        return {'path': path,
                'time': sum(parts[cr] for cr in criteria_ls),
                'parts': parts}

    def _save_file_list(files: list[Path]) -> None:
        list_path = kwargs.get('list_path')
        if list_path is None:
            return

        path_format = kwargs.get('path_format', 'name')
        if path_format not in {'name', 'full', 'cwd'}:
            raise ValueError(f"Unsupported draw_json_streams path_format: {path_format}")

        out_path = Path(list_path)
        if out_path.is_dir():
            out_path = out_path/'stream_ls.txt'
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def fmt_path(path: Path) -> str:
            if path_format == 'name':
                return path.name
            if path_format == 'full':
                return str(path.resolve())
            return os.path.relpath(path.resolve(), Path.cwd().resolve())

        lines = [fmt_path(path) for path in files]
        if kwargs.get('sort_list', True):
            lines = sorted(lines)

        with out_path.open('w', encoding='utf-8') as f:
            for line in lines:
                f.write(f'{line}\n')

    def _resolve_collect_source(path: Path, list_file: Path | None = None) -> Path:
        candidates = [path]
        if list_file is not None and not path.is_absolute():
            candidates.append(list_file.parent / path)
        json_root = Path(json_path) if not isinstance(json_path, (list, tuple, set)) else None
        if json_root is not None and json_root.is_dir() and not path.is_absolute():
            candidates.append(json_root / path)

        for candidate in candidates:
            try:
                return resolve_json_source(candidate)
            except FileNotFoundError:
                pass
        return resolve_json_source(path)

    def _load_collect_paths() -> list[Path]:
        collect = kwargs.get('json_collect')
        if collect is None:
            return []
        if isinstance(collect, (str, Path)):
            collect = Path(collect)
            if collect.is_file():
                paths = []
                for line in collect.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if line:
                        paths.append(_resolve_collect_source(Path(line), collect))
                return paths
            return [_resolve_collect_source(collect)]
        return [_resolve_collect_source(Path(path)) for path in collect]

    tolerance = kwargs.get('tolerance', 0.05)

    if cutoff <= 0.0:
        raise ValueError('draw_json_streams cut off must be positive')
    if tolerance < 0.0:
        raise ValueError('draw_json_streams tolerance must be non-negative')

    criteria_ls, _ = _normalize_criteria(criteria)
    allowed_max = cutoff + tolerance if tolerance > 1.0 else cutoff * (1.0 + tolerance)
    rng = random.Random(kwargs.get('seed', SJ_DRAW_RANDOM_SEED))

    collect_paths = _load_collect_paths()
    collect_keys = {path.resolve() for path in collect_paths}
    selected = []
    for path in collect_paths:
        try:
            row = _file_times(path)
            selected.append(row)
        except Exception as exc:
            print_color(f"[ERROR] draw_json_streams skipped {path}: {type(exc).__name__}: {exc}", 'r')

    candidates = []
    for path in _load_sj_list(json_path):
        try:
            if resolve_json_source(path).resolve() in collect_keys:
                continue
            row = _file_times(path)
            if row['time'] > 0.0:
                candidates.append(row)
        except Exception as exc:
            print_color(f"[ERROR] draw_json_streams skipped {path}: {type(exc).__name__}: {exc}", 'r')

    rng.shuffle(candidates)
    t_acc = sum(row['time'] for row in selected)

    while candidates and not (cutoff <= t_acc <= allowed_max):
        row = candidates.pop()
        selected.append(row)
        t_acc += row['time']

        while selected and t_acc > allowed_max:
            idx = max(range(len(selected)), key=lambda i: selected[i]['time'])
            dropped = selected.pop(idx)
            t_acc -= dropped['time']

    totals = {'t_total': 0.0, 'empty': 0.0, TAG_NO_EVENT: 0.0, TAG_ABNORMAL: 0.0,
              TAG_FALL: 0.0, TAG_TENSION: 0.0, TAG_FIGHT: 0.0}
    for row in selected:
        for key in totals:
            totals[key] += row['parts'][key]

    files_list = [row['path'] for row in selected]
    _save_file_list(files_list)
    return files_list, totals


def collect_jsons(json_ls, src_dir, trg_dir) -> list[Path]:
    """ Copy selected SJ files into trg_dir from src_dir or absolute list entries."""

    def _load_list(jls) -> list[Path]:
        if isinstance(jls, (str, Path)):
            jls = Path(jls)
            if jls.is_file():
                return [Path(l.strip()) for l in jls.read_text(encoding='utf-8').splitlines()
                        if l.strip()]
            print(f"collect_jsons expected a list file, got: {jls}")
            return []
        return [Path(path) for path in jls]

    entries = _load_list(json_ls)
    trg_dir = Path(trg_dir)
    trg_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    if entries and src_dir is None and not entries[0].is_absolute():
        print_color("[WARN] collect_jsons cannot resolve relative paths without src_dir", 'y')
        return copied
    src_dir = Path(src_dir) if src_dir is not None else None

    print('DBUG:', entries)
    for entry in entries:
        src_req = entry if entry.is_absolute() else src_dir/entry
        if entry.is_absolute():
            print(entry)
        else:
            print("not abolute")
        print(src_req)

        try:
            src = resolve_json_source(src_req)
        except FileNotFoundError:
            print(f"collect_jsons missing file: {src_req}")
            continue

        dst = trg_dir/src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied

#* endregion *#

#* region Stream JSON compare **************************#
def compare_stream_json(j1, j2, *, tolerances=None, ignore_video_path=True) -> tuple[bool, dict[str, Any]]:
    """Compare two stream JSONs by metadata, frame layout, annotations, and numeric payload."""
    if tolerances is not None and not isinstance(tolerances, dict):
        raise TypeError('compare_stream_json tolerances must be None or a concrete dict')
    data_1 = j1 if isinstance(j1, dict) else load_json_raw(j1)
    data_2 = j2 if isinstance(j2, dict) else load_json_raw(j2)

    metadata = compare_meta(data_1, data_2, ignore_video_path=ignore_video_path)
    stream = compare_stream(data_1, data_2, tolerances=tolerances)
    frame_structure = stream['frame_structure']
    annotations = stream['annotations']
    numeric = stream['numeric']

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

#* endregion *#


#* region Convert (npz, json) pair to stream JSON  ***************#
MINIMAL_DETECTOR = {'model': 'npz_import', 'version':None, 'source':'out_alex_pair'}
STREAM_JSON_PROGRESS_STEP = 2000

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


def _event_intervals(event_intervals: dict[str, Any] | None) -> dict[str, dict[str, list]]:
    out = {}
    for key, payload in (event_intervals or {}).items():
        sec_intervals = payload.get('sec', []) if isinstance(payload, dict) else payload
        out[str(key)] = {'sec': list(sec_intervals or [])}
    return out


def _stream_json_base(meta: dict[str, Any], npz_data) -> dict[str, Any]:
    timing = meta.get('timing', {}) or {}
    target_rate = timing.get('sampling_rates_hz')
    if target_rate is None:
        target_rate = timing.get('sampling_rate_hz')
    return {'video': meta.get('video') or _scalar(npz_data['video']),
            'fps': meta.get('fps') if meta.get('fps') is not None else _scalar(npz_data['fps']),
            'sampling rate': {'target': target_rate, 'effective': timing.get('effective_fps')},
            'step': _step_value(meta),
            'detector': dict(MINIMAL_DETECTOR),
            'event_intervals': _event_intervals(meta.get('event_intervals'))}


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


def _frame_detections(classes, confidences, bboxes, keypoints, person_count: int, frame_idx: int) -> list[dict[str, Any]]:
    if person_count <= 0:
        return []
    cls_ls = classes[frame_idx, :person_count].tolist()
    conf_ls = confidences[frame_idx, :person_count].tolist()
    bbox_ls = bboxes[frame_idx, :person_count].tolist()
    kp_ls = keypoints[frame_idx, :person_count].tolist()
    dets = []
    for det_idx in range(person_count):
        dets.append({'class': int(cls_ls[det_idx]),
                     'conf': float(conf_ls[det_idx]),
                     'bbox': bbox_ls[det_idx],
                     'key_points': kp_ls[det_idx]})
    return dets


def _build_frames(npz_data) -> list[dict[str, Any]]:
    frame_indices = npz_data['frame_indices']
    frame_times = npz_data['frame_times_sec']
    group_events_arr = npz_data['group_events']
    group_counts = npz_data['group_event_counts']
    person_counts = npz_data['person_counts']
    classes = npz_data['classes']
    confidences = npz_data['confidences']
    bboxes = npz_data['bboxes']
    keypoints = npz_data['keypoints']

    frames = []
    for row_idx, frame_no in enumerate(frame_indices):
        event_count = int(group_counts[row_idx])
        group_events = group_events_arr[row_idx]
        group_tags = [int(v) for v in group_events[:event_count] if v != 0]
        person_count = int(person_counts[row_idx])
        frames.append({'f': int(frame_no),
                       't': float(frame_times[row_idx]),
                       'individual_events': [],
                       'group_events': sorted(set(group_tags), reverse=True),
                       'detection_list': _frame_detections(classes, confidences, bboxes, keypoints,
                                                           person_count, row_idx)})
    return frames


def save_pair_stream_json(npz_path, json_path, out_path, **kwargs) -> Path:
    """Convert and save one json+npz pair as standard Stream JSON without building all frames in memory."""
    json_path, npz_path, out_path = Path(json_path), Path(npz_path), Path(out_path)
    if str(out_path).endswith(f'{JSON_SUFFIX}.gz') or out_path.suffix.lower() == '.gz':
        raise ValueError("Saving gzip Stream JSON is disabled; use .json.zip or .json")
    progress = kwargs.get('progress', True)
    progress_step = int(kwargs.get('progress_step', STREAM_JSON_PROGRESS_STEP))

    def write_json_body(file):
        file.write('{')
        for idx, (key, value) in enumerate(base.items()):
            if idx:
                file.write(',')
            json.dump(key, file, ensure_ascii=False, separators=(',', ':'))
            file.write(':')
            json.dump(value, file, ensure_ascii=False, separators=(',', ':'))
        file.write(',"frames":[')
        for row_idx, frame_no in enumerate(frame_indices):
            if row_idx:
                file.write(',')
            event_count = int(group_counts[row_idx])
            group_tags = [int(v) for v in group_events[row_idx, :event_count] if v != 0]
            person_count = int(person_counts[row_idx])
            frame = {'f': int(frame_no),
                     't': float(frame_times[row_idx]),
                     'individual_events': [],
                     'group_events': sorted(set(group_tags), reverse=True),
                     'detection_list': _frame_detections(classes, confidences, bboxes, keypoints,
                                                         person_count, row_idx)}
            json.dump(frame, file, ensure_ascii=False, separators=(',', ':'))
            if progress and progress_step > 0 and (row_idx + 1) % progress_step == 0:
                print(f"  converted {row_idx + 1}/{len(frame_indices)} frames")
        file.write(']}')

    with json_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    npz_data = np.load(npz_path, allow_pickle=True)
    try:
        _validate_pair(meta, npz_data, json_path.stem)
        base = _stream_json_base(meta, npz_data)
        frame_indices = npz_data['frame_indices']
        frame_times = npz_data['frame_times_sec']
        group_events = npz_data['group_events']
        group_counts = npz_data['group_event_counts']
        person_counts = npz_data['person_counts']
        classes = npz_data['classes']
        confidences = npz_data['confidences']
        bboxes = npz_data['bboxes']
        keypoints = npz_data['keypoints']

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if str(out_path).endswith(f'{JSON_SUFFIX}.zip') or out_path.suffix.lower() == '.zip':
            json_name = (out_path.name[:-4] if str(out_path).endswith(f'{JSON_SUFFIX}.zip')
                         else out_path.with_suffix(JSON_SUFFIX).name)
            with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                with zf.open(json_name, 'w') as raw:
                    with io.TextIOWrapper(raw, encoding='utf-8') as file:
                        write_json_body(file)
        else:
            with out_path.open('w', encoding='utf-8') as file:
                write_json_body(file)
    finally:
        npz_data.close()

    return out_path


def pair_to_stream_json(npz_path, json_path, out_path=None, **kwargs) -> dict[str, Any]:
    """ Convert one HMC pair into one stream-JSON dict and optionally save it."""
    json_path, npz_path = Path(json_path), Path(npz_path)

    with json_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    npz_data = np.load(npz_path, allow_pickle=True)
    stem = json_path.stem
    try:
        _validate_pair(meta, npz_data, stem)
        data = _stream_json_base(meta, npz_data)
        data['frames'] = _build_frames(npz_data)
    finally:
        npz_data.close()

    dst = None
    if out_path is not None:
        out_path = Path(out_path)
        if str(out_path).endswith(f'{JSON_SUFFIX}.gz') or out_path.suffix.lower() == '.gz':
            raise ValueError("Saving gzip Stream JSON is disabled; use .json.zip or .json")
        dst = (out_path if out_path.suffix.lower() in {JSON_SUFFIX, '.zip'}
               or str(out_path).endswith(f'{JSON_SUFFIX}.zip')
                        else out_path / f"{stem}.json")
    if dst is not None:
        compression = ('zip' if str(dst).endswith(f'{JSON_SUFFIX}.zip') or dst.suffix.lower() == '.zip'
                       else 'none')
        save_json_raw(data, dst, compression=compression)

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
        save_pair_stream_json(npz_stems[stem], json_stems[stem], out_path=dst, **kwargs)
        out_paths.append(dst)
    return out_paths

#* endregion *#


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_dir', type=Path, help='directory containing matched .json/.npz stems')
    parser.add_argument('-o', '--out-dir', type=Path, default=None,
                        help='output directory for converted stream JSON files')
    return parser

#1056(3,16,5)
# 833(3,15,5)

if __name__ == '__main__':
    pass
