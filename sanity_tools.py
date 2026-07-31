"""Numerical and structural comparison tools for sanity checks.

Public API:
    files_equal(file_1, file_2) -> bool
    compare_json(test, ref, **kwargs) -> dict
    compare_npz(test, ref, **kwargs) -> dict
    compare_csv(test, ref, **kwargs) -> dict
    compare_file(test, ref, **kwargs) -> dict
    compare_dirs(test_dir, ref_dir, **kwargs) -> dict
    compare_stream_json_dirs(test_dir, ref_dir, **kwargs) -> dict
    run_sanity_test(models, ref_dir, ds_tests=None, stm_tests=None, **kwargs) -> dict

CLI:
    python3 sanity_tools.py compare REF_PATH TARGET_PATH [options]
    python3 sanity_tools.py test MODELS [...] REF_DIR [options]
    python3 sanity_tools.py test \
        work_dirs/models \
        work_dirs/reference-run \
        --ds-tests data/cache/Joint_sets \
        --stm-tests data/json_files/testing \
        --sanity-op-dir work_dirs/sanity \
        --test-kwargs "{'threshold':[0.5, 0.6], 'test_pair':True, 'print_reports':'none'}"
    --test-kwargs may also point to a JSON file. If omitted, the command uses
    REF_DIR/test-config.json when available.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from json_utils import (STREAM_FILE_TYPES, list_json_sources, load_json_raw,
                        resolve_json_files, resolve_json_source, save_json_raw)


DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = None
DEFAULT_VARIANCE_K = 0.03
DEFAULT_MAX_ISSUES = 50
FILE_COMPARE_CHUNK_SIZE = 1024*1024
DEFAULT_OP_DIR = Path('work_dirs/sanity')
SUPPORTED_PATTERNS = tuple(f'*{suffix}' for suffix in STREAM_FILE_TYPES) + ('*.npz', '*.csv')


# region Public API
def files_equal(file_1, file_2, chunk_size=FILE_COMPARE_CHUNK_SIZE) -> bool:
    """Return whether two files contain exactly the same bytes."""
    file_1, file_2 = Path(file_1), Path(file_2)
    for path in (file_1, file_2):
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if file_1.stat().st_size != file_2.stat().st_size:
        return False

    with file_1.open('rb') as f_1, file_2.open('rb') as f_2:
        while True:
            chunk_1 = f_1.read(chunk_size)
            chunk_2 = f_2.read(chunk_size)
            if chunk_1 != chunk_2:
                return False
            if not chunk_1:
                return True


def compare_json(test, ref, **kwargs) -> dict[str, Any]:
    """Compare two JSON paths or dictionaries structurally and numerically."""
    def add_issue(issue_type, path, **details):
        structure['mismatches'] += 1
        if len(structure['issues']) < max_issues:
            structure['issues'].append({'type': issue_type, 'path': path, **details})

    def compare_values(test_val, ref_val, path=''):
        if path in ignore_paths:
            return
        structure['compared'] += 1

        if isinstance(test_val, dict) and isinstance(ref_val, dict):
            test_keys, ref_keys = set(test_val), set(ref_val)
            for key in sorted(test_keys - ref_keys):
                child = f'{path}.{key}' if path else str(key)
                if child not in ignore_paths:
                    add_issue('extra_key', child)
            for key in sorted(ref_keys - test_keys):
                child = f'{path}.{key}' if path else str(key)
                if child not in ignore_paths:
                    add_issue('missing_key', child)
            for key in sorted(test_keys & ref_keys):
                child = f'{path}.{key}' if path else str(key)
                compare_values(test_val[key], ref_val[key], child)
            return

        if isinstance(test_val, list) and isinstance(ref_val, list):
            if len(test_val) != len(ref_val):
                add_issue('list_length', path, test=len(test_val), ref=len(ref_val))
            elif all(_is_number(value) for value in test_val + ref_val):
                numeric.add(test_val, ref_val, path)
                return
            for idx, (test_item, ref_item) in enumerate(zip(test_val, ref_val)):
                compare_values(test_item, ref_item, f'{path}[{idx}]')
            return

        test_num = _is_number(test_val)
        ref_num = _is_number(ref_val)
        if test_num and ref_num:
            numeric.add(test_val, ref_val, path)
            return

        if type(test_val) is not type(ref_val):
            add_issue('type', path, test=type(test_val).__name__, ref=type(ref_val).__name__)
        elif test_val != ref_val:
            add_issue('value', path, test=test_val, ref=ref_val)

    max_issues = int(kwargs.get('max_issues', DEFAULT_MAX_ISSUES))
    ignore_paths = set(kwargs.get('ignore_paths') or ())
    numeric = _NumericComparison(kwargs.get('atol', DEFAULT_ATOL),
                                 kwargs.get('rtol', DEFAULT_RTOL), max_issues,
                                 kwargs.get('variance_k', DEFAULT_VARIANCE_K))
    test_path = test if isinstance(test, (str, Path)) else None
    ref_path = ref if isinstance(ref, (str, Path)) else None
    test_label = str(test_path) if test_path is not None else '<memory>'
    ref_label = str(ref_path) if ref_path is not None else '<memory>'

    byte_equal = None
    if test_path is not None and ref_path is not None:
        test_src, ref_src = resolve_json_source(test_path), resolve_json_source(ref_path)
        byte_equal = files_equal(test_src, ref_src)
        if byte_equal:
            return _byte_equal_report('json', test_label, ref_label)

    test_data = load_json_raw(test_path) if test_path is not None else test
    ref_data = load_json_raw(ref_path) if ref_path is not None else ref
    structure = {'ok': True, 'compared': 0, 'mismatches': 0, 'issues': []}
    compare_values(test_data, ref_data)
    structure['ok'] = structure['mismatches'] == 0
    numeric_report = numeric.report()
    return {'ok': structure['ok'] and numeric_report['ok'],
            'kind': 'json',
            'test': test_label,
            'ref': ref_label,
            'byte_equal': byte_equal,
            'structure': structure,
            'numeric': numeric_report}


def compare_npz(test, ref, **kwargs) -> dict[str, Any]:
    """Compare two NPZ files by keys, array structure, and numeric values."""
    test, ref = Path(test), Path(ref)
    if files_equal(test, ref):
        return _byte_equal_report('npz', str(test), str(ref))

    max_issues = int(kwargs.get('max_issues', DEFAULT_MAX_ISSUES))
    ignored = set(kwargs.get('ignore_keys') or ())
    check_dtype = bool(kwargs.get('check_dtype', True))
    numeric = _NumericComparison(kwargs.get('atol', DEFAULT_ATOL),
                                 kwargs.get('rtol', DEFAULT_RTOL), max_issues,
                                 kwargs.get('variance_k', DEFAULT_VARIANCE_K))
    structure = {'ok': True,
                 'keys_tested': 0,
                 'mismatches': 0,
                 'missing_keys': [],
                 'extra_keys': [],
                 'shape_mismatches': [],
                 'dtype_mismatches': [],
                 'value_mismatches': []}

    with np.load(test, allow_pickle=True) as test_data, np.load(ref, allow_pickle=True) as ref_data:
        test_keys = set(test_data.files) - ignored
        ref_keys = set(ref_data.files) - ignored
        structure['extra_keys'] = sorted(test_keys - ref_keys)
        structure['missing_keys'] = sorted(ref_keys - test_keys)

        for key in sorted(test_keys & ref_keys):
            structure['keys_tested'] += 1
            test_arr, ref_arr = test_data[key], ref_data[key]
            if test_arr.shape != ref_arr.shape:
                structure['shape_mismatches'].append(
                    {'key': key, 'test': list(test_arr.shape), 'ref': list(ref_arr.shape)})
                continue
            if check_dtype and test_arr.dtype != ref_arr.dtype:
                structure['dtype_mismatches'].append(
                    {'key': key, 'test': str(test_arr.dtype), 'ref': str(ref_arr.dtype)})

            if np.issubdtype(test_arr.dtype, np.number) and np.issubdtype(ref_arr.dtype, np.number):
                numeric.add(test_arr, ref_arr, key)
                continue

            try:
                values_equal = np.array_equal(test_arr, ref_arr, equal_nan=True)
            except (TypeError, ValueError):
                values_equal = np.array_equal(test_arr, ref_arr)
            if not values_equal:
                structure['mismatches'] += 1
                if len(structure['value_mismatches']) < max_issues:
                    structure['value_mismatches'].append({'key': key})

    structure['mismatches'] += (len(structure['missing_keys'])
                                + len(structure['extra_keys'])
                                + len(structure['shape_mismatches'])
                                + len(structure['dtype_mismatches']))
    structure['ok'] = structure['mismatches'] == 0
    numeric_report = numeric.report()
    return {'ok': structure['ok'] and numeric_report['ok'],
            'kind': 'npz',
            'test': str(test),
            'ref': str(ref),
            'byte_equal': False,
            'structure': structure,
            'numeric': numeric_report}


def compare_csv(test, ref, **kwargs) -> dict[str, Any]:
    """Compare two CSV files structurally and with tolerant numeric cells."""
    def load_rows(path):
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            sample = f.read(4096)
            f.seek(0)
            delimiter = ';' if sample.count(';') > sample.count(',') else ','
            return list(csv.reader(f, delimiter=delimiter))

    def as_number(value):
        try:
            return float(value.strip())
        except (AttributeError, ValueError):
            return None

    def add_issue(issue_type, path, **details):
        structure['mismatches'] += 1
        if len(structure['issues']) < max_issues:
            structure['issues'].append({'type': issue_type, 'path': path, **details})

    test, ref = Path(test), Path(ref)
    if files_equal(test, ref):
        return _byte_equal_report('csv', str(test), str(ref))

    max_issues = int(kwargs.get('max_issues', DEFAULT_MAX_ISSUES))
    ignore_columns = set(kwargs.get('ignore_columns') or ())
    numeric = _NumericComparison(kwargs.get('atol', DEFAULT_ATOL),
                                 kwargs.get('rtol', DEFAULT_RTOL), max_issues,
                                 kwargs.get('variance_k', DEFAULT_VARIANCE_K))
    test_rows, ref_rows = load_rows(test), load_rows(ref)
    structure = {'ok': True,
                 'rows_tested': min(len(test_rows), len(ref_rows)),
                 'row_count': {'test': len(test_rows), 'ref': len(ref_rows)},
                 'mismatches': 0,
                 'issues': []}
    if len(test_rows) != len(ref_rows):
        add_issue('row_count', '', test=len(test_rows), ref=len(ref_rows))

    header_idx, ignored_indices = None, set()
    if ignore_columns:
        for row_idx, row in enumerate(test_rows):
            found = {idx for idx, value in enumerate(row) if value.strip() in ignore_columns}
            if found:
                header_idx, ignored_indices = row_idx, found
                break

    column_rtol = {}
    for row_idx, ref_row in enumerate(ref_rows):
        for col_idx, value in enumerate(ref_row):
            if header_idx is not None and row_idx >= header_idx and col_idx in ignored_indices:
                continue
            number = as_number(value)
            if number is not None:
                column_rtol.setdefault(col_idx, []).append(number)
    column_rtol = {col_idx: numeric.resolve_rtol(values)
                   for col_idx, values in column_rtol.items()}

    for row_idx, (test_row, ref_row) in enumerate(zip(test_rows, ref_rows)):
        if len(test_row) != len(ref_row):
            add_issue('column_count', f'row[{row_idx}]', test=len(test_row), ref=len(ref_row))
        for col_idx, (test_cell, ref_cell) in enumerate(zip(test_row, ref_row)):
            if header_idx is not None and row_idx >= header_idx and col_idx in ignored_indices:
                continue
            path = f'row[{row_idx}].col[{col_idx}]'
            test_num, ref_num = as_number(test_cell), as_number(ref_cell)
            if test_num is not None and ref_num is not None:
                numeric.add(test_num, ref_num, path, rtol=column_rtol.get(col_idx))
            elif test_cell != ref_cell:
                add_issue('value', path, test=test_cell, ref=ref_cell)

    structure['ok'] = structure['mismatches'] == 0
    numeric_report = numeric.report()
    return {'ok': structure['ok'] and numeric_report['ok'],
            'kind': 'csv',
            'test': str(test),
            'ref': str(ref),
            'byte_equal': False,
            'structure': structure,
            'numeric': numeric_report}


def compare_file(test, ref, *, kind=None, **kwargs) -> dict[str, Any]:
    """Compare one supported file pair, inferring its kind when omitted."""
    kind = kind or _file_kind(test)
    if kind == 'json':
        return compare_json(test, ref, **kwargs)
    if kind == 'npz':
        return compare_npz(test, ref, **kwargs)
    if kind == 'csv':
        return compare_csv(test, ref, **kwargs)
    if kind != 'stream_json':
        raise ValueError(f"Unsupported comparison kind: {kind!r}")

    from json_stream_utils import compare_stream_json

    test_path = test if isinstance(test, (str, Path)) else None
    ref_path = ref if isinstance(ref, (str, Path)) else None
    test_label = str(test_path) if test_path is not None else '<memory>'
    ref_label = str(ref_path) if ref_path is not None else '<memory>'
    byte_equal = None
    if test_path is not None and ref_path is not None:
        byte_equal = files_equal(resolve_json_source(test_path), resolve_json_source(ref_path))
        if byte_equal:
            return _byte_equal_report(kind, test_label, ref_label)

    ok, details = compare_stream_json(
        test, ref,
        tolerances=kwargs.get('tolerances'),
        ignore_path_fields=bool(kwargs.get('ignore_path_fields', True)))
    metadata = details['metadata']
    frames = details['frame_structure']
    annotations = details['annotations']
    structure_mismatches = (
        len(metadata['unequal'])
        + int(frames['frame_count'] is not None)
        + len(frames['missing_frame_indices'])
        + len(frames['extra_frame_indices'])
        + len(frames['timestamp_mismatches'])
        + len(frames['detection_count_mismatches'])
        + int(not annotations['event_intervals_equal'])
        + len(annotations['frame_annotation_mismatches'])
    )
    structure = {'ok': structure_mismatches == 0,
                 'mismatches': structure_mismatches,
                 'metadata': metadata,
                 'frames': frames,
                 'annotations': annotations}
    raw_numeric = details['numeric']
    numeric = {'ok': raw_numeric['within_tolerance'],
               'count': raw_numeric['count'],
               'mismatches': int(not raw_numeric['within_tolerance']),
               'avg_abs': raw_numeric['avg_abs'],
               'max_abs': raw_numeric['max_abs'],
               'max_path': raw_numeric['max_path'],
               'tolerances': raw_numeric['tolerances'],
               'issues': []}
    return {'ok': ok,
            'kind': kind,
            'test': test_label,
            'ref': ref_label,
            'byte_equal': byte_equal,
            'structure': structure,
            'numeric': numeric}


def compare_dirs(test_dir, ref_dir, **kwargs) -> dict[str, Any]:
    """Compare supported files in two directories by exact relative path."""
    def collect_files(base_dir):
        def is_excluded(path):
            return any(path.match(pattern) for pattern in exclude_patterns)

        if patterns is None:
            paths = base_dir.rglob('*') if recursive else base_dir.glob('*')
            return {path.relative_to(base_dir).as_posix(): path for path in paths
                    if path.is_file() and not is_excluded(path)}

        files = {}
        for pattern in patterns:
            paths = base_dir.rglob(pattern) if recursive else base_dir.glob(pattern)
            for path in paths:
                if path.is_file() and not is_excluded(path):
                    files[path.relative_to(base_dir).as_posix()] = path
        return files

    test_dir, ref_dir = Path(test_dir), Path(ref_dir)
    for path in (test_dir, ref_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)

    recursive = bool(kwargs.get('recursive', True))
    raw_patterns = kwargs.get('patterns')
    patterns = ([raw_patterns] if isinstance(raw_patterns, str)
                else list(raw_patterns) if raw_patterns is not None else None)
    raw_excludes = kwargs.get('exclude_patterns')
    exclude_patterns = ([raw_excludes] if isinstance(raw_excludes, str)
                        else list(raw_excludes) if raw_excludes is not None else [])
    options = kwargs.get('options') or {}
    test_files, ref_files = collect_files(test_dir), collect_files(ref_dir)
    missing = sorted(set(ref_files) - set(test_files))
    extra = sorted(set(test_files) - set(ref_files))
    file_reports, errors = [], []

    for rel_path in sorted(set(test_files) & set(ref_files)):
        try:
            kind = _file_kind(test_files[rel_path])
            report = compare_file(test_files[rel_path], ref_files[rel_path],
                                  kind=kind, **options.get(kind, {}))
            report['file'] = rel_path
        except Exception as error:
            report = {'ok': False,
                      'kind': 'error',
                      'test': str(test_files[rel_path]),
                      'ref': str(ref_files[rel_path]),
                      'file': rel_path,
                      'error': f'{type(error).__name__}: {error}'}
            errors.append({'file': rel_path, 'error': report['error']})
        file_reports.append(report)

    if not test_files and not ref_files:
        errors.append({'error': 'no files found'})
    ok = not missing and not extra and not errors and all(report['ok'] for report in file_reports)
    return {'ok': ok,
            'test_dir': str(test_dir),
            'ref_dir': str(ref_dir),
            'missing': missing,
            'extra': extra,
            'files': file_reports,
            'errors': errors}


def compare_stream_json_dirs(test_dir, ref_dir, **kwargs) -> dict[str, Any]:
    """Compare logical Stream JSON files across plain and compressed sources."""
    test_dir, ref_dir = Path(test_dir), Path(ref_dir)
    for path in (test_dir, ref_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)

    test_files = {path.name: path for path in list_json_sources(test_dir)}
    ref_files = {path.name: path for path in list_json_sources(ref_dir)}
    missing = sorted(set(ref_files) - set(test_files))
    extra = sorted(set(test_files) - set(ref_files))
    file_reports, errors = [], []

    for name in sorted(set(test_files) & set(ref_files)):
        try:
            report = compare_file(test_files[name], ref_files[name], kind='stream_json',
                                  tolerances=kwargs.get('tolerances'),
                                  ignore_path_fields=kwargs.get('ignore_path_fields', True))
            report['file'] = name
        except Exception as error:
            report = {'ok': False,
                      'kind': 'stream_json',
                      'test': str(test_files[name]),
                      'ref': str(ref_files[name]),
                      'file': name,
                      'error': f'{type(error).__name__}: {error}'}
            errors.append({'file': name, 'error': report['error']})
        file_reports.append(report)

    if not test_files and not ref_files:
        errors.append({'error': 'no Stream JSON files found'})
    ok = not missing and not extra and not errors and all(report['ok'] for report in file_reports)
    return {'ok': ok,
            'test_dir': str(test_dir),
            'ref_dir': str(ref_dir),
            'missing': missing,
            'extra': extra,
            'files': file_reports,
            'errors': errors}


def run_sanity_test(models, ref_dir, ds_tests=None, stm_tests=None, **kwargs) -> dict[str, Any]:
    """Run model tests into a unique directory, then compare them with reference results."""
    from common.my_local_utils import as_collection, get_unique_name

    sanity_op_dir = Path(kwargs.pop('sanity_op_dir', DEFAULT_OP_DIR))
    root_path = kwargs.pop('root_path', None)
    report_mode = kwargs.pop('report', 'quick')
    atol = kwargs.pop('atol', DEFAULT_ATOL)
    variance_k = kwargs.pop('variance_k', DEFAULT_VARIANCE_K)
    if 'out_dir' in kwargs:
        raise ValueError("out_dir is managed by run_sanity_test; use sanity_op_dir")

    ref_dir = Path(ref_dir)
    if not ref_dir.is_dir():
        raise NotADirectoryError(ref_dir)
    model_refs = list(as_collection(models))
    if not model_refs:
        raise ValueError('models cannot be empty')
    missing_models = [Path(ref) for ref in model_refs if not Path(ref).exists()]
    if missing_models:
        raise FileNotFoundError(f"Model path not found: {missing_models[0]}")

    stream_inputs = list(as_collection(stm_tests)) if stm_tests is not None else None
    if stream_inputs and len(stream_inputs) == 1 and not isinstance(stream_inputs[0], dict):
        list_path = Path(stream_inputs[0])
        if list_path.is_file() and list_path.suffix.lower() == '.txt':
            stream_inputs = resolve_json_files(list_path, root_path)

    sanity_op_dir.mkdir(parents=True, exist_ok=True)
    target_dir = get_unique_name(
        sanity_op_dir/f"sanity_{datetime.now().strftime('%y%m%d_%H-%M-%S')}")
    target_dir.mkdir()

    from scripts import test_models

    print(f'\nStarting sanity inference -> {target_dir}')
    tested_dirs = test_models(model_refs, ds_tests=ds_tests, stm_tests=stream_inputs,
                              out_dir=target_dir, **kwargs)
    comparison, report_path = _run_comparison(
        ref_dir, target_dir, output_dir=target_dir, report_mode=report_mode,
        atol=atol, variance_k=variance_k)
    return {'target_dir': target_dir,
            'tested_dirs': [Path(path) for path in tested_dirs],
            'report_path': report_path,
            'comparison': comparison}


# endregion


# region Helpers
class _NumericComparison:
    """Accumulate tolerant numeric comparison statistics across scalar or array values."""

    def __init__(self, atol, rtol, max_issues, variance_k):
        self.atol = float(atol)
        self.rtol = None if rtol is None else float(rtol)
        self.variance_k = float(variance_k)
        self.max_issues = int(max_issues)
        self.count = 0
        self.mismatches = 0
        self.total_abs = 0.0
        self.max_abs = 0.0
        self.max_path = None
        self.issues = []
        self.rtol_min = None
        self.rtol_max = None

    def resolve_rtol(self, ref):
        if self.rtol is not None:
            return self.rtol

        values = np.asarray(ref)
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return 0.0
        median = np.median(finite)
        spread = 1.4826 * np.median(np.abs(finite - median))
        scale = np.sqrt(np.mean(np.abs(finite) ** 2))
        return self.variance_k * float(spread) / max(float(scale), self.atol)

    def add(self, test, ref, path, rtol=None):
        test_arr, ref_arr = np.asarray(test), np.asarray(ref)
        if test_arr.shape != ref_arr.shape:
            raise ValueError(f'numeric shape mismatch at {path}: {test_arr.shape} != {ref_arr.shape}')
        if not test_arr.size:
            return

        rtol = self.resolve_rtol(ref_arr) if rtol is None else float(rtol)
        self.rtol_min = rtol if self.rtol_min is None else min(self.rtol_min, rtol)
        self.rtol_max = rtol if self.rtol_max is None else max(self.rtol_max, rtol)
        close = np.isclose(test_arr, ref_arr, atol=self.atol, rtol=rtol, equal_nan=True)
        value_type = np.complex128 if (np.iscomplexobj(test_arr) or np.iscomplexobj(ref_arr)) else np.float64
        delta = np.abs(test_arr.astype(value_type) - ref_arr.astype(value_type))
        delta = np.where(close & ~np.isfinite(delta), 0.0, delta)
        delta = np.where(~close & np.isnan(delta), np.inf, delta)

        self.count += int(delta.size)
        mismatch_mask = ~close
        self.mismatches += int(np.count_nonzero(mismatch_mask))
        self.total_abs += float(np.sum(delta))

        flat_idx = int(np.argmax(delta))
        current_max = float(delta.flat[flat_idx])
        if current_max > self.max_abs:
            index = np.unravel_index(flat_idx, delta.shape)
            self.max_abs = current_max
            self.max_path = _value_path(path, index, delta.ndim)

        remaining = self.max_issues - len(self.issues)
        if remaining <= 0:
            return
        for mismatch_idx in np.flatnonzero(mismatch_mask)[:remaining]:
            index = np.unravel_index(int(mismatch_idx), delta.shape)
            test_val = test_arr[index]
            ref_val = ref_arr[index]
            self.issues.append(
                {'path': _value_path(path, index, delta.ndim),
                 'test': _python_scalar(test_val),
                 'ref': _python_scalar(ref_val),
                 'abs_error': float(delta[index]),
                 'rtol': rtol})

    def report(self):
        return {'ok': self.mismatches == 0,
                'count': self.count,
                'mismatches': self.mismatches,
                'avg_abs': self.total_abs/self.count if self.count else 0.0,
                'max_abs': self.max_abs,
                'max_path': self.max_path,
                'atol': self.atol,
                'rtol': self.rtol,
                'variance_k': self.variance_k if self.rtol is None else None,
                'rtol_used': ([self.rtol_min, self.rtol_max]
                              if self.rtol_min is not None else None),
                'issues': self.issues}


def _byte_equal_report(kind, test, ref):
    return {'ok': True,
            'kind': kind,
            'test': test,
            'ref': ref,
            'byte_equal': True,
            'structure': None,
            'numeric': None}


def _file_kind(path):
    name = Path(path).name.lower()
    if name.endswith(STREAM_FILE_TYPES):
        return 'json'
    if name.endswith('.npz'):
        return 'npz'
    if name.endswith('.csv'):
        return 'csv'
    raise ValueError(f'Unsupported file type: {path}')


def _is_number(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _value_path(path, index, ndim):
    if ndim == 0:
        return path
    return f"{path}[{','.join(str(i) for i in index)}]"


# endregion


# region Printing reports
def _is_supported(path):
    try:
        _file_kind(path)
    except ValueError:
        return False
    return True


def _print_table(headers, rows, separator_before=None):
    widths = [max(len(str(value)) for value in [header, *(row[idx] for row in rows)])
              for idx, header in enumerate(headers)]
    fmt = ' | '.join(f'{{:^{width}}}' for width in widths)
    print(fmt.format(*headers))
    print('-+-'.join('-'*width for width in widths))
    for row_idx, row in enumerate(rows):
        if row_idx == separator_before:
            print('-+-'.join('-'*width for width in widths))
        print(fmt.format(*row))


def _file_result(file_report):
    if file_report.get('byte_equal') is True:
        return 'Byte equal'
    structure = file_report.get('structure') or {}
    numeric = file_report.get('numeric') or {}
    if structure.get('ok') and numeric.get('ok'):
        return 'Numerical equal'
    if structure.get('ok'):
        return 'Structural equal'
    return 'Not equal'


def _report_counts(file_reports):
    counts = {kind: {'total': 0, 'byte': 0, 'numerical': 0, 'structural': 0, 'failed': 0}
              for kind in ('npz', 'csv', 'json')}
    for file_report in file_reports:
        try:
            kind = _file_kind(Path(file_report['file']))
        except ValueError:
            continue
        counts[kind]['total'] += 1
        byte_equal = file_report.get('byte_equal') is True
        structural = byte_equal or bool((file_report.get('structure') or {}).get('ok'))
        numerical = structural and (byte_equal or bool((file_report.get('numeric') or {}).get('ok')))
        counts[kind]['byte'] += int(byte_equal)
        counts[kind]['structural'] += int(structural)
        counts[kind]['numerical'] += int(numerical)
        counts[kind]['failed'] += int(not file_report['ok'])
    return counts


def _print_quick(report, ref_path, target_path, ref_files=None, target_files=None, saved_path=None):
    print()
    if ref_files is None:
        print(f'Compared: {ref_path.name}  vs  {target_path.name}')
        print(f'Result: {_file_result(report)}')
    else:
        ref_all = {path.relative_to(ref_path).as_posix() for path in ref_files}
        target_all = {path.relative_to(target_path).as_posix() for path in target_files}
        ref_supported = {path.relative_to(ref_path).as_posix()
                         for path in ref_files if _is_supported(path)}
        target_supported = {path.relative_to(target_path).as_posix()
                            for path in target_files if _is_supported(path)}
        ref_ignored = ref_all - ref_supported
        target_ignored = target_all - target_supported
        counts = _report_counts(report['files'])
        total_compared = sum(row['total'] for row in counts.values())

        print(f'Compared: {ref_path}  vs  {target_path}\n')
        print('File inventory')
        _print_table(['', 'Ref', 'Target', 'Missing', 'Extra'],
                     [('All files', len(ref_all), len(target_all),
                       len(ref_all - target_all), len(target_all - ref_all)),
                      ('Supported', len(ref_supported), len(target_supported),
                       len(ref_supported - target_supported),
                       len(target_supported - ref_supported)),
                      ('Ignored', len(ref_ignored), len(target_ignored),
                       len(ref_ignored - target_ignored),
                       len(target_ignored - ref_ignored))])
        print(f"\nCompared pairs: {total_compared}  (npz: {counts['npz']['total']} |"
              f" csv: {counts['csv']['total']} | json: {counts['json']['total']})\n")
        result_rows = [(kind, counts[kind]['byte'], counts[kind]['numerical'],
                        counts[kind]['structural'], counts[kind]['failed'])
                       for kind in ('npz', 'csv', 'json')]
        result_rows.append(('Total',
                            sum(row['byte'] for row in counts.values()),
                            sum(row['numerical'] for row in counts.values()),
                            sum(row['structural'] for row in counts.values()),
                            sum(row['failed'] for row in counts.values())))
        _print_table(['Results', 'Byte', 'Numerical', 'Structural', 'Failed'], result_rows,
                     separator_before=len(result_rows) - 1)
    if saved_path is not None:
        print()
        print(f'Saved report: {saved_path}')


def _all_files(path):
    return [file for file in Path(path).rglob('*')
            if file.is_file() and not file.match('sanity_report*.json')]


def _save_report(report, output_dir):
    from common.my_local_utils import cli_warning, get_unique_name

    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = get_unique_name(output_dir/'sanity_report.json')
        return save_json_raw(report, report_path, compression='none', default=json_default)
    except Exception as error:
        cli_warning(f'Sanity report was not saved: {type(error).__name__}: {error}', 'y')
        return None


def _run_comparison(ref_path, target_path, *, output_dir=None, report_mode='quick',
                    atol=DEFAULT_ATOL, variance_k=DEFAULT_VARIANCE_K):
    ref_path, target_path = Path(ref_path), Path(target_path)
    if report_mode not in {'quick', 'full', 'none'}:
        raise ValueError("report must be 'quick', 'full', or 'none'")
    if not ref_path.exists():
        raise FileNotFoundError(f'Reference path does not exist: {ref_path}')
    if not target_path.exists():
        raise FileNotFoundError(f'Target path does not exist: {target_path}')
    if ref_path.is_dir() != target_path.is_dir():
        raise ValueError('Reference and target must both be files or both be directories')

    compare_kwargs = {'atol': atol, 'variance_k': variance_k}
    ref_files = target_files = None
    if ref_path.is_dir():
        ref_files, target_files = _all_files(ref_path), _all_files(target_path)
        options = {kind: dict(compare_kwargs) for kind in ('json', 'npz', 'csv')}
        report = compare_dirs(target_path, ref_path, patterns=SUPPORTED_PATTERNS,
                              exclude_patterns='sanity_report*.json', options=options)
    else:
        if _file_kind(ref_path) != _file_kind(target_path):
            raise ValueError('Reference and target file types must match')
        report = compare_file(target_path, ref_path, **compare_kwargs)

    report_path = _save_report(report, output_dir or ref_path.parent)
    if report_mode == 'quick':
        _print_quick(report, ref_path, target_path, ref_files, target_files, report_path)
    elif report_mode == 'full':
        print('Full sanity report is not implemented yet.')
    return report, report_path


# endregion


# region CLI
def main(argv=None):
    """Run the local sanity comparison command."""
    import argparse
    import ast
    import json

    def load_kwargs_file(config_path):
        try:
            with Path(config_path).open('r', encoding='utf-8') as f:
                parsed = json.load(f)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f'invalid test kwargs file {config_path}: {error}') from error
        if not isinstance(parsed, dict):
            raise ValueError(f'test kwargs file must contain a JSON object: {config_path}')
        return parsed

    def parse_test_kwargs(value):
        config_path = Path(value)
        if config_path.is_file():
            try:
                return load_kwargs_file(config_path)
            except ValueError as error:
                raise argparse.ArgumentTypeError(str(error)) from error
        if config_path.suffix.lower() == '.json':
            raise argparse.ArgumentTypeError(f'test kwargs file not found: {config_path}')
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise argparse.ArgumentTypeError(f'invalid Python dictionary: {error}') from error
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError('value must be a Python dictionary or JSON file')
        return parsed

    def add_comparison_args(prs):
        prs.add_argument('-k', type=float, default=DEFAULT_VARIANCE_K,
                         help=f'Adaptive relative tolerance factor (default: {DEFAULT_VARIANCE_K})')
        prs.add_argument('--atol', type=float, default=DEFAULT_ATOL,
                         help=f'Absolute tolerance (default: {DEFAULT_ATOL})')
        prs.add_argument('--report', choices=('quick', 'full', 'none'), default='quick',
                         help='Console report mode (default: quick)')

    parser = argparse.ArgumentParser(description='Compare numerical project outputs for sanity checks.')
    commands = parser.add_subparsers(dest='command', required=True)
    compare = commands.add_parser('compare', help='Compare reference and target files or directories')
    compare.add_argument('ref_path', type=Path, help='Reference file or directory')
    compare.add_argument('target_path', type=Path, help='Target file or directory')
    compare.add_argument('--op_dir', type=Path, default=None, help='Directory for the saved JSON report')
    add_comparison_args(compare)

    test = commands.add_parser('test', help='Test model and compare with reference')
    test.add_argument('models', type=Path, nargs='+', help='Model dirs or .pt paths')
    test.add_argument('ref_dir', type=Path, help='Reference results directory')
    test.add_argument('-ds', '--ds-tests', type=Path, nargs='+', default=None, help='Dataset NPZ files, directories, or masks')
    test.add_argument('-stm', '--stm-tests', type=Path, nargs='+', default=None, help='Stream testing files, dir or stream-list file')
    test.add_argument('-rp', '--root-path', type=Path, default=None, help='Root dir for paths in a stream-list file')
    test.add_argument('-arg','--test-kwargs', type=parse_test_kwargs, default=None,
                      help='Python dict or JSON file forwarded to test_models(); default: REF_DIR/test-config.json')
    test.add_argument('-op', '--sanity-op-dir', type=Path, default=DEFAULT_OP_DIR, help=f'Parent dir for generated runs (default: {DEFAULT_OP_DIR})')

    add_comparison_args(test)
    args = parser.parse_args(argv)

    try:
        if args.command == 'test':
            config_path = args.ref_dir/'test-config.json'
            if args.test_kwargs is not None:
                test_kwargs = dict(args.test_kwargs)
            elif config_path.is_file():
                test_kwargs = load_kwargs_file(config_path)
            else:
                test_kwargs = {}
            test_kwargs.update({'sanity_op_dir': args.sanity_op_dir,
                                'root_path': args.root_path,
                                'report': args.report,
                                'atol': args.atol,
                                'variance_k': args.k})
            return run_sanity_test(args.models, args.ref_dir, args.ds_tests,
                                   args.stm_tests, **test_kwargs)

        report, _ = _run_comparison(
            args.ref_path, args.target_path,
            output_dir=args.op_dir or args.ref_path.parent,
            report_mode=args.report, atol=args.atol, variance_k=args.k)
        return report
    except (FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()


# endregion
