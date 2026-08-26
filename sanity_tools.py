"""    Numerical and structural comparison tools for sanity checks.
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

import csv, json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any
#* local imports
from common.my_local_utils import as_collection, assert_path, get_unique_name, cli_warning
from json_stream_utils import compare_stream_json
from json_utils import (STREAM_FILE_TYPES, list_json_sources, load_json_raw,
                        resolve_json_files, resolve_json_source, save_json_raw)

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = None
DEFAULT_VARIANCE_K = 0.03
DEFAULT_MAX_ISSUES = 50
COMPARE_CHUNK_SIZE = 1024 * 1024
DEFAULT_OP_DIR = Path('work_dirs/sanity')
DEFAULT_JSON_IGNORE_PATHS = {'output_dir', 'raw_results_path'}
JSON_REQUIRED_STRING_PATHS = {'detector.model'}
SUPPORTED_PATTERNS = tuple(f'*{sfx}' for sfx in STREAM_FILE_TYPES) + ('*.npz', '*.csv')

# region Public API
def bytewise_equal(file_1, file_2, chunk_size=COMPARE_CHUNK_SIZE)-> bool:
    """ Return whether two files are bytewise equal """

    file_1, file_2 = assert_path(file_1, 'file'), assert_path(file_2, 'file')
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if file_1.stat().st_size != file_2.stat().st_size:
        return False
    with file_1.open('rb') as f_1, file_2.open('rb') as f_2:
        while True:
            chunk_1,chunk_2 = f_1.read(chunk_size), f_2.read(chunk_size)
            if chunk_1 != chunk_2:
                return False
            if not chunk_1:
                return True


def compare_json(test, ref, **kwargs) -> dict[str, Any]:
    """ Compare two JSON paths or dictionaries structurally and numerically. """
    def add_issue(issue_type, path, **details):
        structure['mismatches'] += 1
        if len(structure['issues']) < max_issues:
            structure['issues'].append({'type': issue_type, 'path': path, **details})

    def compare_values(tst_val, ref_val, path=''):
        if path in ignore_paths:
            return
        structure['compared'] += 1

        if isinstance(tst_val, dict) and isinstance(ref_val, dict):
            test_keys, ref_keys = set(tst_val), set(ref_val)
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
                compare_values(tst_val[key], ref_val[key], child)
            return

        if isinstance(tst_val, list) and isinstance(ref_val, list):
            if len(tst_val) != len(ref_val):
                add_issue('list_length', path, test=len(tst_val), ref=len(ref_val))
            elif all(_is_number(value) for value in tst_val + ref_val):
                numeric.add(tst_val, ref_val, path)
                return
            for idx, (test_item, ref_item) in enumerate(zip(tst_val, ref_val)):
                compare_values(test_item, ref_item, f'{path}[{idx}]')
            return

        if _is_number(tst_val) and _is_number(ref_val):
            numeric.add(tst_val, ref_val, path)
            return

        if isinstance(tst_val, str) and isinstance(ref_val, str):
            if path in JSON_REQUIRED_STRING_PATHS and tst_val != ref_val:
                add_issue('value', path, test=tst_val, ref=ref_val)
            return

        if type(tst_val) is not type(ref_val):
            add_issue('type', path, test=type(tst_val).__name__, ref=type(ref_val).__name__)
        elif tst_val != ref_val:
            add_issue('value', path, test=tst_val, ref=ref_val)

    max_issues = int(kwargs.get('max_issues', DEFAULT_MAX_ISSUES))
    ignore_paths = DEFAULT_JSON_IGNORE_PATHS | set(kwargs.get('ignore_paths') or ())
    numeric = _NumericComparison(kwargs.get('atol', DEFAULT_ATOL),
                                 kwargs.get('rtol', DEFAULT_RTOL), max_issues,
                                 kwargs.get('variance_k', DEFAULT_VARIANCE_K))
    tst_path = test if isinstance(test, (str, Path)) else None
    ref_path = ref if isinstance(ref, (str, Path)) else None
    tst_lbl = str(tst_path) if tst_path is not None else '<memory>'
    ref_lbl = str(ref_path) if ref_path is not None else '<memory>'

    byte_equal = None
    if tst_path is not None and ref_path is not None:
        test_src, ref_src = resolve_json_source(tst_path), resolve_json_source(ref_path)
        byte_equal = bytewise_equal(test_src, ref_src)
        if byte_equal:
            return _byte_equal_report('json', tst_lbl, ref_lbl)

    tst_data = load_json_raw(tst_path) if tst_path is not None else test
    ref_data = load_json_raw(ref_path) if ref_path is not None else ref
    structure = {'ok': True, 'compared': 0, 'mismatches': 0, 'issues': []}
    compare_values(tst_data, ref_data)
    structure['ok'] = structure['mismatches'] == 0
    numeric_report = numeric.report()
    return {'ok': structure['ok'] and numeric_report['ok'],
            'kind': 'json',
            'test': tst_lbl, 'ref': ref_lbl,
            'byte_equal': byte_equal,
            'structure': structure,
            'numeric': numeric_report}


def compare_npz(test, ref, **kwargs) -> dict[str, Any]:
    """Compare two NPZ files by keys, array structure, and numeric values."""
    test, ref = Path(test), Path(ref)
    if bytewise_equal(test, ref):
        return _byte_equal_report('npz', str(test), str(ref))

    max_issues = int(kwargs.get('max_issues', DEFAULT_MAX_ISSUES))
    ignored = set(kwargs.get('ignore_keys') or ())
    check_dtype = bool(kwargs.get('check_dtype', True))
    numeric = _NumericComparison(kwargs.get('atol', DEFAULT_ATOL),
                                 kwargs.get('rtol', DEFAULT_RTOL), max_issues,
                                 kwargs.get('variance_k', DEFAULT_VARIANCE_K))
    structure = {'ok': True, 'keys_tested': 0, 'mismatches': 0,
                 'missing_keys': [], 'extra_keys': [],
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
            'test': str(test), 'ref': str(ref),
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
    if bytewise_equal(test, ref):
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

    col_rtol = {}
    for row_idx, ref in enumerate(ref_rows):
        for col_idx, value in enumerate(ref):
            if header_idx is not None and row_idx >= header_idx and col_idx in ignored_indices:
                continue
            num = as_number(value)
            if num is not None:
                col_rtol.setdefault(col_idx, []).append(num)
    col_rtol = {col_i: numeric.resolve_rtol(vals) for col_i, vals in col_rtol.items()}

    for row_idx, (test_row, ref) in enumerate(zip(test_rows, ref_rows)):
        if len(test_row) != len(ref):
            add_issue('column_count', f'row[{row_idx}]', test=len(test_row), ref=len(ref))
        for col_idx, (test_cell, ref_cell) in enumerate(zip(test_row, ref)):
            if header_idx is not None and row_idx >= header_idx and col_idx in ignored_indices:
                continue
            cell_ref = f'row[{row_idx}].col[{col_idx}]'
            test_num, ref_num = as_number(test_cell), as_number(ref_cell)
            if test_num is not None and ref_num is not None:
                numeric.add(test_num, ref_num, cell_ref, rtol=col_rtol.get(col_idx))
            elif test_cell != ref_cell:
                add_issue('value', cell_ref, test=test_cell, ref=ref_cell)

    structure['ok'] = structure['mismatches'] == 0
    numeric_report = numeric.report()
    return {'ok': structure['ok'] and numeric_report['ok'],
            'kind': 'csv', 'test': str(test), 'ref': str(ref),
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

    test_path = test if isinstance(test, (str, Path)) else None
    ref_path = ref if isinstance(ref, (str, Path)) else None
    test_label = str(test_path) if test_path is not None else '<memory>'
    ref_label = str(ref_path) if ref_path is not None else '<memory>'
    byte_equal = None
    if test_path is not None and ref_path is not None:
        byte_equal = bytewise_equal(resolve_json_source(test_path), resolve_json_source(ref_path))
        if byte_equal:
            return _byte_equal_report(kind, test_label, ref_label)

    ok, details = compare_stream_json(test, ref, tolerances=kwargs.get('tolerances'),
                                      ignore_video_path=bool(kwargs.get('ignore_video_path', True)))
    metadata = details['metadata']
    frames = details['frame_structure']
    annotations = details['annotations']
    struct_mises = ( len(metadata['unequal'])
                     + int(frames['frame_count'] is not None)
                     + len(frames['missing_frame_indices'])
                     + len(frames['extra_frame_indices'])
                     + len(frames['timestamp_mismatches'])
                     + len(frames['detection_count_mismatches'])
                     + int(not annotations['event_intervals_equal'])
                     + len(annotations['frame_annotation_mismatches'])
                     )
    structure = {'ok': struct_mises == 0, 'mismatches': struct_mises,
                 'metadata': metadata, 'frames': frames,
                 'annotations': annotations}
    raw_numeric = details['numeric']
    numeric = {'ok': raw_numeric['within_tolerance'],
               'count': raw_numeric['count'],
               'mismatches': int(not raw_numeric['within_tolerance']),
               'avg_abs': raw_numeric['avg_abs'],
               'max_abs': raw_numeric['max_abs'],
               'max_path': raw_numeric['max_path'],
               'tolerances': raw_numeric['tolerances'],
               'issues': []
               }
    return {'ok': ok, 'kind': kind,
            'test': test_label, 'ref': ref_label,
            'byte_equal': byte_equal, 'structure': structure, 'numeric': numeric}


def compare_dirs(test_dir, ref_dir, **kwargs) -> dict[str, Any]:
    """ Compare supported files in two directories by relative path or logical name."""
    warnings_ls = []

    def collect_files(base_dir):
        def is_excluded(pth):
            return any(pth.match(ptn) for ptn in exclude_patterns)

        def get_id(pth):
            f_id = pth.name if match_by_name else pth.relative_to(base_dir).as_posix()
            lower = f_id.lower()
            for suffix in STREAM_FILE_TYPES:
                if lower.endswith(suffix):
                    return f_id[:-len(suffix)] + '.json'
            return f_id

        files_by_id = {}
        scan_patterns = patterns if patterns is not None else ("*",)
        for ptrn in scan_patterns:
            all_paths = base_dir.rglob(ptrn) if recursive else base_dir.glob(ptrn)
            for file_path in all_paths:
                if not file_path.is_file() or is_excluded(file_path):
                    continue
                file_id = get_id(file_path)
                if file_id in files_by_id:
                    first = files_by_id[file_id]
                    if first is None:
                        warning = next(itm for itm in warnings_ls if itm['file_base'] == file_id)
                        warning['files'].append(str(file_path))
                        continue
                    if first == file_path:
                        continue
                    files_by_id[file_id] = None
                    warning = next((itm for itm in warnings_ls if itm['file_base'] == file_id), None)
                    if warning is not None:
                        warning['files'].extend((str(first), str(file_path)))
                        continue
                    warning = {'type': 'duplicated_files', 'file_base': file_id,
                               'files': [str(first), str(file_path)]}
                    warnings_ls.append(warning)
                    continue
                files_by_id[file_id] = file_path
        return {file_id: path for file_id, path in files_by_id.items() if path is not None}

    test_dir, ref_dir = assert_path(test_dir, 'dir'), assert_path(ref_dir, 'dir')
    # if not test_dir.is_dir()

    recursive = kwargs.get('recursive', True)
    match_by_name = kwargs.get('match_by_name', False)
    raw_patterns = kwargs.get('patterns')
    raw_exc = kwargs.get('exclude_patterns')
    patterns = ([raw_patterns] if isinstance(raw_patterns, str) else
                list(raw_patterns) if  raw_patterns is not None else
                None)
    exclude_patterns = ([raw_exc]     if isinstance(raw_exc, str) else
                        list(raw_exc) if raw_exc is not None      else
                        [])
    options = kwargs.get('options') or {}
    tst_files, ref_files = collect_files(test_dir), collect_files(ref_dir)
    misses = sorted(set(ref_files) - set(tst_files))
    extra  = sorted(set(tst_files) - set(ref_files))
    file_reports, errors = [], []

    for rel_path in sorted(set(tst_files) & set(ref_files)):
        try:
            kind = _file_kind(tst_files[rel_path])
            report = compare_file(tst_files[rel_path], ref_files[rel_path], kind=kind, **options.get(kind, {}))
            report['file'] = rel_path
        except Exception as error:
            report = {'ok': False, 'kind': 'error',
                      'test': str(tst_files[rel_path]),
                      'ref': str(ref_files[rel_path]),
                      'file': rel_path, 'error': f'{type(error).__name__}: {error}'
                      }
            errors.append({'file': rel_path, 'error': report['error']})
        file_reports.append(report)

    if not tst_files and not ref_files:
        errors.append({'error': 'no files found'})
    ok = not misses and not extra and not warnings_ls and not errors and all(fr['ok'] for fr in file_reports)

    return {'ok': ok,
            'test_dir': str(test_dir), 'ref_dir': str(ref_dir),
            'match_by_name': match_by_name,
            'missing': misses, 'extra': extra,
            'files': file_reports,
            'warnings': warnings_ls, 'errors': errors
            }


def compare_stream_json_dirs(test_dir, ref_dir, **kwargs) -> dict[str, Any]:
    """Compare logical Stream JSON files across plain and compressed sources."""
    test_dir, ref_dir = assert_path(test_dir, 'dir'), assert_path(ref_dir, 'dir')

    test_files = {path.name: path for path in list_json_sources(test_dir)}
    ref_files = {path.name: path for path in list_json_sources(ref_dir)}
    missing = sorted(set(ref_files) - set(test_files))
    extra = sorted(set(test_files) - set(ref_files))
    file_reports, errors = [], []

    for name in sorted(set(test_files) & set(ref_files)):
        try:
            report = compare_file(test_files[name], ref_files[name], kind='stream_json',
                                  tolerances=kwargs.get('tolerances'),
                                  ignore_video_path=kwargs.get('ignore_video_path', True))
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

    op_dir = Path(kwargs.pop('sanity_op_dir', DEFAULT_OP_DIR))
    root_path = kwargs.pop('root_path', None)
    print_report = kwargs.pop('print_report', 'none')
    save_report = kwargs.pop('save_report', True)
    report_dir = kwargs.pop('output_dir', None)
    atol = kwargs.pop('atol', DEFAULT_ATOL)
    variance_k = kwargs.pop('variance_k', DEFAULT_VARIANCE_K)
    if 'out_dir' in kwargs:
        raise ValueError("out_dir is managed by run_sanity_test; use sanity_op_dir")
    ref_dir = assert_path(ref_dir, 'dir')
    model_refs = list(as_collection(models))
    if not model_refs:
        raise ValueError('models cannot be empty')
    model_refs = [assert_path(ref) for ref in model_refs]

    stream_inputs = list(as_collection(stm_tests)) if stm_tests is not None else None
    if stream_inputs and len(stream_inputs) == 1 and not isinstance(stream_inputs[0], dict):
        list_path = Path(stream_inputs[0])
        if list_path.is_file() and list_path.suffix.lower() == '.txt':
            stream_inputs = resolve_json_files(list_path, root_path)

    op_dir.mkdir(parents=True, exist_ok=True)
    target_dir = get_unique_name(op_dir/f"sanity_{datetime.now().strftime('%y%m%d_%H-%M-%S')}")
    target_dir.mkdir()

    from scripts import test_models

    print(f'\nStarting sanity inference -> {target_dir}')
    tested_dirs = test_models(model_refs, ds_tests=ds_tests, stm_tests=stream_inputs,
                              out_dir=target_dir, **kwargs)
    comparison, report_path = _run_comparison(
        ref_dir, target_dir, output_dir=report_dir or target_dir,
        print_report=print_report, save_report=save_report,
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
        scale = np.sqrt(np.mean(np.abs(finite)**2))
        return self.variance_k * float(spread)/max(float(scale), self.atol)

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
            self.issues.append( {'path': _value_path(path, index, delta.ndim),
                                 'test': _python_scalar(test_val),
                                 'ref': _python_scalar(ref_val),
                                 'abs_error': float(delta[index]),
                                 'rtol': rtol
                                 })

    def report(self):
        return {'ok': self.mismatches == 0,
                'count': self.count,
                'mismatches': self.mismatches,
                'avg_abs': self.total_abs/self.count if self.count else 0.0,
                'max_abs': self.max_abs, 'max_path': self.max_path,
                'atol': self.atol, 'rtol': self.rtol,
                'variance_k': self.variance_k if self.rtol is None else None,
                'rtol_used': [self.rtol_min, self.rtol_max] if (self.rtol_min is not None) else None,
                'issues': self.issues}


def _byte_equal_report(kind, test, ref):
    return {'ok': True, 'kind': kind, 'test': test, 'ref': ref,
            'byte_equal': True, 'structure': None, 'numeric': None}

def _file_kind(path):
    name = Path(path).name.lower()
    if name.endswith(STREAM_FILE_TYPES):
        return 'json'
    if name.endswith('.npz'):
        return 'npz'
    if name.endswith('.csv'):
        return 'csv'
    raise ValueError(f'Unsupported file type: {path}')


def _is_number(val):
    return isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_))


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _value_path(path, index, ndim):
    return f"{path}[{','.join(str(i) for i in index)}]"  if ndim != 0 else path

# endregion


# region Printing reports
def _print_table(headers, rows, separator_before=None, equal_width=False, bold_after=()):
    widths = [max(len(str(value)) for value in [header, *(row[idx] for row in rows)])
                  for idx, header in enumerate(headers)]
    if equal_width:
        widths = [max(widths)]*len(widths)
    bold_after = set(bold_after)
    fmt = ''.join(f'{{{idx}:^{width}}}' + (' ┃ ' if idx in bold_after else ' | ')
                  for idx, width in enumerate(widths[:-1])) + f'{{{len(widths) - 1}:^{widths[-1]}}}'
    divider = ''.join('-'*width + ('-╋-' if idx in bold_after else '-+-')
                      for idx, width in enumerate(widths[:-1])) + '-'*widths[-1]
    print(fmt.format(*headers))
    print(divider)
    for row_idx, row in enumerate(rows):
        if row_idx == separator_before:
            print(divider)
        print(fmt.format(*row))


def _file_result(file_report):
    if file_report.get('byte_equal') is True:
        return 'Byte equal'
    kind = file_report.get('kind')
    structure = file_report.get('structure') or {}
    numeric = file_report.get('numeric') or {}
    if kind == 'npz' and numeric.get('ok'):
        return 'Numerical equal'
    #*ToDo: recheck that rules
    if structure.get('ok') and numeric.get('ok'):
        return 'Numerical equal'
    if structure.get('ok'):
        return 'Structural equal'
    return 'Not equal'


def _report_counts(file_reports):
    """Count each file once by its highest achieved equality level."""
    counts = {kind: {'total': 0, 'byte': 0, 'numerical': 0, 'structural': 0, 'failed': 0}
                    for kind in ('npz', 'csv', 'json')}
    for report in file_reports:
        try:
            kind = _file_kind(Path(report['file']))
        except ValueError:
            continue
        counts[kind]['total'] += 1
        byte_equal = report.get('byte_equal') is True
        structural = byte_equal or bool((report.get('structure') or {}).get('ok'))
        numeric_ok = byte_equal or bool((report.get('numeric') or {}).get('ok'))
        numerical = numeric_ok if kind == 'npz' else structural and numeric_ok
        if  byte_equal:
            counts[kind]['byte'] += 1
        elif numerical:
            counts[kind]['numerical']  += 1
        elif structural:
            counts[kind]['structural'] += 1
        else:
            counts[kind]['failed'] += 1
    return counts


def _print_quick(report, ref_path, target_path, ref_files=None, target_files=None, saved_path=None):

    def _is_supported(path):
        try:
            _file_kind(path)
        except ValueError:
            return False
        return True

    def inventory_id(path, base_dir):
        """Return the relative or logical filename used in inventory counts."""
        file_id = path.name if report.get('match_by_name') else path.relative_to(base_dir).as_posix()
        lower = file_id.lower()
        for sfx in STREAM_FILE_TYPES:
            if lower.endswith(sfx):
                return file_id[:-len(sfx)] + '.json'
        return file_id

    print()
    if ref_files is None:
        print(f'Compared: {ref_path.name}  vs  {target_path.name}')
        print(f'Result: {_file_result(report)}')
    else:
        ref_all = {inventory_id(path, ref_path) for path in ref_files}
        trg_all = {inventory_id(path, target_path) for path in target_files}
        ref_supported = {inventory_id(path, ref_path) for path in ref_files if _is_supported(path)}
        target_supported = {inventory_id(path, target_path) for path in target_files if _is_supported(path)}
        ref_ignored = ref_all - ref_supported
        target_ignored = trg_all - target_supported
        counts = _report_counts(report['files'])
        total_compared = sum(row['total'] for row in counts.values())

        print(f'Compared: {ref_path}  vs  {target_path}\n')
        print('File inventory')
        _print_table(['', 'Ref', 'Target', 'Missing', 'Extra'],
                     [('All files', len(ref_all), len(trg_all),
                            len(ref_all - trg_all), len(trg_all - ref_all)),
                            ('Supported', len(ref_supported), len(target_supported),
                            len(ref_supported - target_supported),
                            len(target_supported - ref_supported)),
                            ('Ignored', len(ref_ignored), len(target_ignored),
                            len(ref_ignored - target_ignored),
                            len(target_ignored - ref_ignored))])
        print(f"\nCompared pairs: {total_compared}  (npz: {counts['npz']['total']} |"
              f" csv: {counts['csv']['total']} | json: {counts['json']['total']})\n")
        for warning in report.get('warnings', []):
            cli_warning(f"Duplicated files for {warning['file_base']!r}: "
                        f"{', '.join(warning['files'])}", 'y')
        result_rows = [(kind, counts[kind]['total'], counts[kind]['byte'], counts[kind]['numerical'],
                        counts[kind]['structural'], counts[kind]['failed'])
                                for kind in ('npz', 'csv', 'json') if counts[kind]['total']]
        result_rows.append(('Total', total_compared,
                            sum(row['byte'] for row in counts.values()),
                            sum(row['numerical'] for row in counts.values()),
                            sum(row['structural'] for row in counts.values()),
                            sum(row['failed'] for row in counts.values())))
        _print_table(['Type', 'Compr.', 'Byte', 'Numeric', 'Struct.', 'Failed'], result_rows,
                     separator_before=len(result_rows) - 1, equal_width=True, bold_after=(0, 4))
    if saved_path is not None:
        print(f'\nSaved report: {saved_path}')


def _all_files(path):
    return [file for file in Path(path).rglob('*') if file.is_file() and not file.match('sanity_report*.json')]


def _save_report(report, output_dir):
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


def _run_comparison(ref_path, trg_path, *, output_dir=None,
                    print_report='none', save_report=True,
                    atol=DEFAULT_ATOL,
                    variance_k=DEFAULT_VARIANCE_K):

    ref_path, trg_path = assert_path(ref_path), assert_path(trg_path)
    if print_report not in {'quick', 'full', 'none'}:
        raise ValueError("print_report must be 'quick', 'full', or 'none'")
    if ref_path.is_dir() != trg_path.is_dir():
        raise ValueError('Reference and target must both be files or both be directories')

    compare_kwargs = {'atol': atol, 'variance_k': variance_k}
    ref_files = target_files = None
    if ref_path.is_dir():
        ref_files, target_files = _all_files(ref_path), _all_files(trg_path)
        options = {kind: dict(compare_kwargs) for kind in ('json', 'npz', 'csv')}
        report = compare_dirs(trg_path, ref_path, patterns=SUPPORTED_PATTERNS,
                              exclude_patterns='sanity_report*.json', options=options, match_by_name=True)
    else:
        if _file_kind(ref_path) != _file_kind(trg_path):
            raise ValueError('Reference and target file types must match')
        report = compare_file(trg_path, ref_path, **compare_kwargs)

    report_path = _save_report(report, output_dir or ref_path.parent) if save_report else None
    if print_report == 'quick':
        _print_quick(report, ref_path, trg_path, ref_files, target_files, report_path)
    elif print_report == 'full':
        print('Full sanity report is not implemented yet.')
    return report, report_path

# endregion


# region CLI
def main(argv=None):
    """Run the local sanity comparison command."""
    import argparse, ast

    def load_test_kwargs(cfg_file):
        try:
            with Path(cfg_file).open('r', encoding='utf-8') as f:
                parsed = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            raise ValueError(f'test kwargs file error: {cfg_file}: {err}') from err
        if not isinstance(parsed, dict):
            raise ValueError(f'test kwargs value error: expected a JSON object: {cfg_file}')
        return parsed

    def parse_test_kwargs(value):
        config_path = Path(value)
        if config_path.is_file() or config_path.suffix.lower() == '.json':
            return load_test_kwargs(config_path)

        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, dict):
                raise ValueError('expected a Python dictionary')
        except (SyntaxError, ValueError) as err:
            raise argparse.ArgumentTypeError(f'test kwargs value error: {err}') from err
        return parsed

    def add_comparison_args(prs):
        prs.add_argument('-k', type=float, default=DEFAULT_VARIANCE_K, help=f'k for Relative Tolerance Factor ')
        prs.add_argument('--atol', type=float, default=DEFAULT_ATOL, help=f'Absolute tolerance. default: {DEFAULT_ATOL}')
        prs.add_argument('--report', choices=('quick', 'full', 'none'), default='quick', help='Console report mode (default: quick)')
        prs.add_argument('-s', '--save-report', nargs='?', const='', default=None, metavar='DIR',
                                                help='Save JSON report to DIR (if given) or default location')

    parser = argparse.ArgumentParser(description='Compare numerical project outputs for sanity checks.')
    commands = parser.add_subparsers(dest='command', required=True)
    compare = commands.add_parser('compare', help='Compare reference and target files or directories')
    compare.add_argument('ref_path', type=Path, help='Reference file or directory')
    compare.add_argument('target_path', type=Path, help='Target file or directory')
    add_comparison_args(compare)

    test = commands.add_parser('test', help='Test model and compare with reference')
    test.add_argument('models', type=Path, nargs='+', help='Model dirs or .pt paths')
    test.add_argument('ref_dir', type=Path, help='Reference results directory')
    test.add_argument('-ds', '--ds-tests', type=Path, nargs='+', default=None, help='Dataset NPZ files, directories, or masks')
    test.add_argument('-stm', '--stm-tests', type=Path, nargs='+', default=None, help='Stream testing files, dir or stream-list file')
    test.add_argument('-rp', '--root-path', type=Path, default=None, help='Root dir for paths in a stream-list file')
    test.add_argument('-arg','--test-kwargs', type=parse_test_kwargs, default=None, help=' Dict or JSON to forward to test_models(); default: REF_DIR/test-config.json')
    test.add_argument('-op', '--sanity-op-dir', type=Path, default=DEFAULT_OP_DIR, help=f'Parent dir for generated runs (default: {DEFAULT_OP_DIR})')

    add_comparison_args(test)
    args = parser.parse_args(argv)

    try:
        save_report = args.save_report is not None
        report_dir = Path(args.save_report) if args.save_report else None
        if args.command == 'test':
            config_json = args.ref_dir/'test-config.json'
            if args.test_kwargs is not None:
                test_kwargs = dict(args.test_kwargs)
            elif config_json.is_file():
                test_kwargs = load_test_kwargs(config_json)
            else:
                test_kwargs = {}
            test_kwargs.update({'sanity_op_dir' : args.sanity_op_dir,
                                'root_path'     : args.root_path,
                                'print_report'  : args.report,
                                'save_report'   : save_report,
                                'output_dir'    : report_dir,
                                'atol'          : args.atol,
                                'variance_k'    : args.k}
                               )
            return run_sanity_test(args.models, args.ref_dir, args.ds_tests, args.stm_tests, **test_kwargs)

        report, _ = _run_comparison( args.ref_path, args.target_path, output_dir=report_dir,
                                     print_report=args.report, save_report=save_report,
                                     atol=args.atol, variance_k=args.k)
        return report
    except (FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        parser.error(str(error))
#* endregion

#939(,11,)->888()
if __name__ == '__main__':
    main()
