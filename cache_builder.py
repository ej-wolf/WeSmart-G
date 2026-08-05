"""Orchestrate combined train/test cache construction."""
from __future__ import annotations

import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from common.my_local_utils import as_collection
from json_utils import list_json_sources, resolve_json_files
from json_stream_utils import stream_stem
from precompute_clips import build_cache_from_json
from motion_feature_schema import MOTION_FPS_REF

CACHE_TIME_ATOL = 1e-6
CACHE_COMPARE_MAX_ISSUES = 50
CACHE_FEATURE_ATOL = 1e-6
CACHE_FEATURE_RTOL = 1e-5


# region Helpers

def _pool_tag(pool_name: str) -> str:
    return {'mean_max': 'mm', 'mean_std_max': 'msm'}.get(pool_name, pool_name)


def _time_tag(window: float, stride: float) -> str:
    def fmt(value: float) -> str:
        text = str(value)
        return text[:-2] if text.endswith('.0') else text

    return f"{fmt(float(window)).replace('.', '')}-{fmt(float(stride)).replace('.', '')}"


def _parse_slice(spec: Any) -> tuple[float, float]:
    if isinstance(spec, dict):
        return float(spec['window']), float(spec['stride'])
    if isinstance(spec, (tuple, list)) and len(spec) >= 2:
        return float(spec[0]), float(spec[1])
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split(':') if part.strip()]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    raise TypeError(f"Unsupported time-slice spec: {spec}")


def _split_paths(paths: list[Path], split_ratio: float, random_seed: int) -> dict[str, list[Path]]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError('split_ratio must be between 0 and 1')
    shuffled = sorted(paths)
    random.Random(random_seed).shuffle(shuffled)
    n_test = int(max(math.ceil(split_ratio), split_ratio * len(shuffled)))
    return {'train': shuffled[n_test:], 'test': shuffled[:n_test]}


def _load_ttp(split_dir: Path, root_dir: Path) -> dict[str, list[Path]]:
    lists = {}
    for split in ('train', 'test'):
        list_file = split_dir / f'{split}_videos.txt'
        if not list_file.is_file():
            raise FileNotFoundError(f'Missing {split} TTP file: {list_file}')
        lists[split] = resolve_json_files(list_file, root_dir)
        if not lists[split]:
            raise ValueError(f'No JSON sources resolved from {list_file}')
    _validate_lists(lists)
    return lists


def _validate_lists(lists: dict[str, list[Path]]) -> None:
    """Reject duplicate streams and train/test overlap before building."""
    seen = set()
    for path in lists['train'] + lists['test']:
        key = str(path.resolve())
        if key in seen:
            raise ValueError(f'Duplicate stream in resolved TTP lists: {path}')
        seen.add(key)
    train_keys = {str(path.resolve()) for path in lists['train']}
    test_keys = {str(path.resolve()) for path in lists['test']}
    overlap = train_keys & test_keys
    if overlap:
        raise ValueError(f'A stream appears in both train and test lists: {next(iter(overlap))}')


# endregion

# region API

def resolve_lists(json_dirs, *, ttp_dir: str | Path | None = None,
                  root_dir: str | Path | None = None) -> dict[str, list[Path]]:
    """Resolve and combine train/test stream paths from source directories.

    Each source directory must contain a complete train/test TTP pair unless
    an explicit global ``ttp_dir`` is supplied.
    """
    dirs = [Path(path) for path in as_collection(json_dirs)]
    if not dirs:
        raise ValueError('No JSON directories were provided')
    for directory in dirs:
        if not directory.is_dir():
            raise NotADirectoryError(directory)

    if ttp_dir is not None:
        root = Path.cwd() if root_dir is None else Path(root_dir)
        return _load_ttp(Path(ttp_dir), root)

    combined = {'train': [], 'test': []}
    missing = []
    for directory in dirs:
        train_file = directory / 'train_videos.txt'
        test_file = directory / 'test_videos.txt'
        if not train_file.is_file() or not test_file.is_file():
            missing.append(str(directory))
            continue
        for split in combined:
            list_file = directory / f'{split}_videos.txt'
            combined[split].extend(resolve_json_files(list_file, directory))

    if missing:
        raise FileNotFoundError('Missing complete TTP pair in: ' + ', '.join(missing))
    if not combined['train'] or not combined['test']:
        raise ValueError('Resolved TTP lists are empty')

    _validate_lists(combined)
    return combined


def draw_ttp(json_dirs, *, split_ratio: float, random_seed: int,
             split_dir: str | Path | None = None,
             output_dir: str | Path | None = None,
             root_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Draw one global video-level TTP and save it without overwriting files."""
    if root_dir is None:
        raise ValueError('root_dir is required when drawing a TTP')
    if split_dir is None and output_dir is None:
        raise ValueError('split_dir or output_dir is required when drawing a TTP')
    save_dir = Path(split_dir if split_dir is not None else output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_file, test_file = save_dir / 'train_videos.txt', save_dir / 'test_videos.txt'
    if train_file.exists() or test_file.exists():
        raise FileExistsError(f'TTP files already exist in {save_dir}')

    paths = []
    for directory in as_collection(json_dirs):
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        paths.extend(list_json_sources(directory))
    if not paths:
        raise ValueError('No JSON sources found for TTP drawing')
    unique_paths = []
    seen_paths = set()
    for path in paths:
        path = path.resolve()
        if path in seen_paths:
            raise ValueError(f'Duplicate stream while drawing TTP: {path}')
        seen_paths.add(path)
        unique_paths.append(path)
    paths = unique_paths

    splits = _split_paths(paths, float(split_ratio), int(random_seed))
    root = Path(root_dir).resolve()
    values = {}
    for split, list_file in (('train', train_file), ('test', test_file)):
        entries = []
        for path in splits[split]:
            try:
                entries.append(path.resolve().relative_to(root).as_posix())
            except ValueError as exc:
                raise ValueError(f'{path} is outside root_dir {root}') from exc
        list_file.write_text('\n'.join(entries) + '\n', encoding='utf-8')
        values[split] = list_file
    return values['train'], values['test']


def build_cache(json_paths, out_path: str | Path, *, window: float,
                stride: float, pool_mode: str, **kwargs) -> Path:
    """Build one NPZ cache from an already-resolved stream list."""
    build_kwargs = dict(kwargs)
    build_kwargs.update({'window': window, 'stride': stride, 'pool_mode': pool_mode})
    return build_cache_from_json(json_paths, out_path, **build_kwargs)


def build_cache_pair(train_paths, test_paths, *, output_dir: str | Path,
                     cache_tag: str, window: float, stride: float,
                     pool_mode: str, **kwargs) -> dict[str, dict[str, Any]]:
    """Build one train/test cache pair with shared feature parameters."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(kwargs.pop('log_path', output_dir / 'caching_log.txt'))
    source_tag = kwargs.pop('stream_src', cache_tag)
    split_seed = kwargs.get('random_seed', 'N/A')
    fps_ref = kwargs.get('motion_fps_ref', MOTION_FPS_REF)
    total = len(train_paths) + len(test_paths)
    results = {}

    def write_log(row: dict[str, Any]) -> None:
        columns = (('time-stamp', 19, '<'), ('status', 6, '<'),
                   ('stream_src', 16, '<'), ('n_jsons', 7, '>'),
                   ('set', 6, '<'), ('split', 10, '>'),
                   ('fps_ref', 7, '>'), ('pool', 7, '<'),
                   ('window-stride', 14, '<'), ('t_wrk', 7, '>'),
                   ('cache_name', 0, '<'))

        def format_row(values: dict[str, Any]) -> str:
            fields = []
            for key, width, align in columns:
                text = str(values[key])
                fields.append(f'{text:{align}{width}}' if width else text)
            return '  '.join(fields)

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8') as file:
                if log_path.stat().st_size == 0:
                    file.write(format_row({key: key for key, _, _ in columns}) + '\n')
                file.write(format_row(row) + '\n')
        except Exception as exc:
            print(f'[WARN] Failed to write cache build log {log_path}: '
                  f'{type(exc).__name__}: {exc}')

    for split, paths in (('train', train_paths), ('test', test_paths)):
        out_path = output_dir / f'{cache_tag}_{split}.npz'
        started = time.time()
        try:
            print(f'building cache: {out_path.name}')
            build_cache(paths, out_path, window=window, stride=stride,
                        pool_mode=pool_mode, **kwargs)
            result = {'cache': out_path, 'ok': out_path.is_file()}
            if result['ok']:
                print(f'OK | t = {time.time() - started:.2f} | {out_path.name}')
        except Exception as exc:
            result = {'cache': out_path, 'ok': False,
                      'error': f'{type(exc).__name__}: {exc}'}
            print(f'[FAIL] {out_path}: {result["error"]}')

        elapsed = time.time() - started
        split_part = len(paths) / total if total else 0.0
        write_log({'time-stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   'status': 'OK' if result['ok'] else 'FAIL',
                   'stream_src': source_tag,
                   'n_jsons': len(paths), 'set': split,
                   'split': f'{split_part:.3f}/{split_seed}', 'fps_ref': fps_ref,
                   'pool': pool_mode, 'window-stride': f'{window:g}-{stride:g}',
                   't_wrk': f'{elapsed:.2f}', 'cache_name': out_path.name})
        results[split] = result
    return results


def compare_caches(ref_cache:str|Path, target_cache: str|Path, **kwargs) -> dict[str, Any]:
    """Compare cache stream, time, and GT records independently of row order.

    Feature values and cache schemas are ignored unless ``compare_features``
    is true. The caches are equivalent when they contain the same logical
    stream stem, clip start and end times, and GT label for every row,
    including duplicate rows.
    """
    def add_issue(issue: dict[str, Any]) -> None:
        if len(report['mismatches']) < max_issues:
            report['mismatches'].append(issue)

    def load_records(path: Path, side: str) -> tuple[list[tuple[str, float, float, int]], np.ndarray | None]:
        try:
            with np.load(path, allow_pickle=True) as data:
                missing = [key for key in ('meta', 'y') if key not in data.files]
                if missing:
                    report['errors'].append(
                        f'{side} cache is missing: {", ".join(missing)}')
                    return [], None
                meta, labels = data['meta'], data['y']
                if len(meta) != len(labels):
                    report['errors'].append(
                        f'{side} meta/y length mismatch: {len(meta)} != {len(labels)}')

                features = None
                if cmp_ftrs:
                    if 'X' not in data.files:
                        report['errors'].append(f'{side} cache is missing: X')
                    else:
                        features = data['X']
                        if features.ndim != 2:
                            report['errors'].append(f'{side} X is not a 2D matrix')
                            features = None
                        elif len(features) != len(meta):
                            report['errors'].append(
                                f'{side} X/meta length mismatch: {len(features)} != {len(meta)}')
                            features = None

                records = []
                valid_indices = []
                for row_idx, (row, label) in enumerate(zip(meta, labels)):
                    if not isinstance(row, dict):
                        report['errors'].append(f'{side} meta[{row_idx}] is not a dict')
                        continue
                    try:
                        video = row['video']
                        start = float(row['t_start'])
                        end = float(row['t_end'])
                        label = label.item() if isinstance(label, np.generic) else label
                        if isinstance(label, bool):
                            label = int(label)
                        elif isinstance(label, (int, np.integer)):
                            label = int(label)
                        elif isinstance(label, (float, np.floating)) and np.isfinite(label):
                            if not float(label).is_integer():
                                raise ValueError('GT label is not integral')
                            label = int(label)
                        else:
                            raise ValueError('GT label is not numeric')
                        if not np.isfinite(start) or not np.isfinite(end):
                            raise ValueError('clip time is not finite')
                        records.append((stream_stem(video), start, end, label))
                        valid_indices.append(row_idx)
                    except (KeyError, TypeError, ValueError) as exc:
                        report['errors'].append(f'{side} meta[{row_idx}] invalid: {exc}')
                selected_features = features[valid_indices] if features is not None else None
                return records, selected_features
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            report['errors'].append(f'{side} cache load failed: {type(exc).__name__}: {exc}')
            return [], None

    def group_records(records: list[tuple[str, float, float, int]]) -> dict[tuple[str, int], list[tuple[float, float]]]:
        grouped = {}
        for stream, start, end, label in records:
            grouped.setdefault((stream, label), []).append((start, end))
        for values in grouped.values():
            values.sort()
        return grouped

    ref_cache, target_cache = Path(ref_cache), Path(target_cache)
    time_atol = float(kwargs.get('time_atol', CACHE_TIME_ATOL))
    cmp_ftrs = bool(kwargs.get('cmp_ftrs', False))
    feature_atol = float(kwargs.get('feature_atol', CACHE_FEATURE_ATOL))
    feature_rtol = float(kwargs.get('feature_rtol', CACHE_FEATURE_RTOL))
    max_issues = int(kwargs.get('max_issues', CACHE_COMPARE_MAX_ISSUES))
    report = {
        'equivalent': False,
        'ref': str(ref_cache),
        'target': str(target_cache),
        'counts': {'ref': 0, 'target': 0, 'matched': 0, 'mismatched': 0,
                   'missing': 0, 'extra': 0},
        'mismatches': [],
        'errors': [],
        'features': {'requested': cmp_ftrs, 'equivalent': None,
                     'shape': {'ref': None, 'target': None}, 'max_abs': None},
    }

    ref_records, ref_features = load_records(ref_cache, 'ref')
    target_records, target_features = load_records(target_cache, 'target')
    report['counts']['ref'] = len(ref_records)
    report['counts']['target'] = len(target_records)

    ref_groups = group_records(ref_records)
    target_groups = group_records(target_records)
    for group in sorted(set(ref_groups) | set(target_groups)):
        stream, label = group
        ref_rows = ref_groups.get(group, [])
        target_rows = target_groups.get(group, [])
        pair_count = min(len(ref_rows), len(target_rows))

        for row_idx in range(pair_count):
            ref_start, ref_end = ref_rows[row_idx]
            target_start, target_end = target_rows[row_idx]
            if (np.isclose(ref_start, target_start, atol=time_atol, rtol=0.0)
                    and np.isclose(ref_end, target_end, atol=time_atol, rtol=0.0)):
                report['counts']['matched'] += 1
            else:
                report['counts']['mismatched'] += 1
                add_issue({'type': 'time', 'stream': stream, 'gt': label,
                           'ref': {'t_start': ref_start, 't_end': ref_end},
                           'target': {'t_start': target_start, 't_end': target_end}})

        missing = len(ref_rows) - pair_count
        extra = len(target_rows) - pair_count
        report['counts']['missing'] += missing
        report['counts']['extra'] += extra
        for start, end in ref_rows[pair_count:]:
            add_issue({'type': 'missing', 'stream': stream, 'gt': label,
                       'ref': {'t_start': start, 't_end': end}})
        for start, end in target_rows[pair_count:]:
            add_issue({'type': 'extra', 'stream': stream, 'gt': label,
                       'target': {'t_start': start, 't_end': end}})

    if cmp_ftrs:
        if ref_features is None or target_features is None:
            report['features']['equivalent'] = False
        else:
            ref_order = sorted(range(len(ref_records)), key=ref_records.__getitem__)
            target_order = sorted(range(len(target_records)), key=target_records.__getitem__)
            ref_matrix = ref_features[ref_order]
            target_matrix = target_features[target_order]
            report['features']['shape'] = {
                'ref': list(ref_matrix.shape), 'target': list(target_matrix.shape)}
            if ref_matrix.shape != target_matrix.shape:
                report['features']['equivalent'] = False
                add_issue({'type': 'feature_shape',
                           'ref': list(ref_matrix.shape),
                           'target': list(target_matrix.shape)})
            else:
                try:
                    difference = np.abs(ref_matrix - target_matrix)
                    report['features']['max_abs'] = (
                        float(np.max(difference)) if difference.size else 0.0)
                    report['features']['equivalent'] = bool(np.allclose(
                        ref_matrix, target_matrix, atol=feature_atol,
                        rtol=feature_rtol, equal_nan=True))
                    if not report['features']['equivalent']:
                        add_issue({'type': 'feature_values',
                                   'max_abs': report['features']['max_abs']})
                except (TypeError, ValueError) as exc:
                    report['features']['equivalent'] = False
                    report['errors'].append(
                        f'Feature comparison failed: {type(exc).__name__}: {exc}')

    report['equivalent'] = (not report['errors']
                            and report['counts']['ref'] == report['counts']['target']
                            and report['counts']['mismatched'] == 0
                            and report['counts']['missing'] == 0
                            and report['counts']['extra'] == 0
                            and report['features']['equivalent'] is not False)
    return report


def build_cache_batch(json_dirs, pool_methods, windows, *, output_dir: str | Path,
                      ttp_dir: str | Path | None = None,
                      root_dir: str | Path | None = None,
                      split_dir: str | Path | None = None,
                      **kwargs) -> list[dict[str, Any]]:
    """Build one combined train/test cache pair per grid configuration."""
    json_dirs = [Path(path) for path in as_collection(json_dirs)]
    split_ratio = kwargs.pop('split_ratio', None)
    random_seed = kwargs.pop('random_seed', None)
    if split_ratio is not None:
        if ttp_dir is not None:
            raise ValueError('Pass either ttp_dir or split_ratio, not both')
        if random_seed is None:
            raise ValueError('random_seed is required when split_ratio is supplied')
        draw_ttp(json_dirs, split_ratio=split_ratio, random_seed=random_seed,
                 split_dir=split_dir, output_dir=output_dir, root_dir=root_dir)
        ttp_dir = split_dir if split_dir is not None else output_dir

    lists = resolve_lists(json_dirs, ttp_dir=ttp_dir, root_dir=root_dir)
    results = []
    cache_tag = kwargs.pop('cache_tag', 'Joint')
    for window_spec in as_collection(windows):
        window, stride = _parse_slice(window_spec)
        for pool_mode in as_collection(pool_methods):
            pool_mode = str(pool_mode)
            tag = f'{cache_tag}_P-{_pool_tag(pool_mode)}_W{_time_tag(window, stride)}'
            results.append(build_cache_pair(
                lists['train'], lists['test'], output_dir=output_dir,
                cache_tag=tag, window=window, stride=stride,
                pool_mode=pool_mode, **kwargs))
    return results


build_caches = build_cache_batch

# endregion

#462(9,21,3)

if __name__ == "__main__":


    json_root = Path("data/json_files")
    out_dir = Path("data/cache/gen_03/Joint_sets-2")

    # train = resolve_json_files(json_root/"gen-2_train_videos.txt", json_root)
    # test  = resolve_json_files(json_root/"gen-2_
    # test_videos.txt", json_root)
    train = resolve_json_files(json_root/"gen-3_train_videos.txt", json_root)
    test  = resolve_json_files(json_root/"gen-3_test_videos.txt", json_root)

    pool_modes = ["max", "lse", "top_k", "mm"]
    win_slc = [(3.0, 1.0, 'W30-10'),
               (1.2, 0.6, 'W12-06')]

    for win, stride, w_tag in win_slc:
        for pool in pool_modes:
            tag = f"Joint_P-{pool}_{w_tag}"
            build_cache_pair( train, test,
                              output_dir=out_dir,
                              cache_tag=tag,
                              window=win, stride=stride,
                              pool_mode=pool,)
