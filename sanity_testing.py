""" Sanity runners for train/test/eval flows and semantic output comparison.
    Usage:
    - run one sanity flow with `run_sanity_flow(...)`
    - compare one new output tree against a reference with `assert_outputs(...)`
    - keep old/new export-format compatibility in this layer instead of the evaluation modules
    ToDo: check file naming for training mode    e.g
        timeline_J-RWL_ft25_w30-15_6_11_8_full_resolution_erez.png
        timeline******************_6_11_8_full_resolution_erez.png
        J-RWL_ft25_w30-15_BM111_cam-6-11-8_FRes_Ana_stream-summary.json
"""


import csv, json
import random, re
from pathlib import Path
from typing import Any
import numpy as np
import torch
#* Project imports 
from common.my_local_utils import as_collection, print_color
from precompute_clips import RANDOM_SEED
from project_utils import get_exporting_name, strip_split_suffix, strip_timestamp_prefix
from scripts import train_models, test_models, infer_eval_threshold, evaluate_raw_test
from torch_clip_model import run_training

LOOSE_TOLERANCES = 0.05
STRICT_TOLERANCES = 0.03
DEFAULT_METRIC_TOLERANCES = {
            'loose' : dict.fromkeys(['max_abs_error', 'cm_delta', 'event_f1'], LOOSE_TOLERANCES),
            'strict': dict.fromkeys(['max_abs_error', 'cm_delta', 'event_f1'], STRICT_TOLERANCES),
             }
DEFAULT_CSV_TOLERANCES = {'loose': LOOSE_TOLERANCES, 'strict': STRICT_TOLERANCES}
DEFAULT_NPZ_TOLERANCES = {'loose': 1e-3, 'strict': 1e-4}

_MODE_TOLERANCE_PROFILE = {'all': 'loose', 'train': 'loose', 'no_train': 'strict', 'eval_only': 'strict'}

_RESULT_REPORT_NAME = 'sanity_report'
_EVAL_PLAN_NAME = '_sanity_eval_plan.json'
_EXPORTED_FILE_PATTERNS = ('*-summary.json',  '*_stream-events.json', '*.npz',
                           'ROC_*.png',  'ROC_*.csv',  '*_timeline.csv', 'timeline_*.csv', 'timeline_*.png',)

# TODO: consider moving these path-like JSON keys into a shared project-wide constant.
_JSON_PATH_KEYS = {'cache_dir', 'events_info', 'model_path', 'new_run_dir', 'output_dir', 'raw_results',
                   'raw_results_path', 'ref_dir', 'ref_run_dir', 'report_path', 'roc_csv', 'test_cache', 'threshold_dir',
                   'timeline_csv', 'timeline_csvs', 'timeline_plot', 'timeline_plots',}
_STREAM_VIDEO_SUFFIXES = {'.mp4', '.avi', '.mkv', '.mov', '.m4v'}

#* this section used only for local printing, and has no effect on sanity testing
SCI_FORMAT = {'low':1e-3, 'high':1e4}
def _fmt_num(val) -> str:
    """Format one numeric comparison value compactly for console output."""
    if val == 0:
        return "0.0"
    abs_val = abs(val)
    if  abs_val < SCI_FORMAT['low'] or SCI_FORMAT['high'] <= abs_val:
        return f"{val:.1e}"
    if type(val) is int:
        return f"{val}"
    if 1 <= abs_val:
        return f"{val:.4g}"
    return f'{val:.4f}'.rstrip('0').rstrip('.')

#* region Config And Small Shared Helpers
def _iter_run_dirs(base_dir: Path) -> list[Path]:
    """List usable run dirs that already contain models or saved test outputs."""
    run_dirs = []
    for path in sorted(base_dir.iterdir()) if base_dir.is_dir() else []:
        if not path.is_dir():
            continue
        has_model = any(path.glob('best_model.*.pt')) or (path/'model.pt').is_file()
        has_outputs = any(path.rglob('*-summary.json')) or any(path.rglob('*-tst.npz'))
        if has_model or has_outputs:
            run_dirs.append(path)
    return run_dirs

def _set_deterministic(seed: int) -> None:
    """ Apply one best-effort deterministic seed setup for Python, NumPy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _resolve_tolerances(mode: str, override: dict[str, float] | None = None) -> dict[str, float]:
    """Return one normalized tolerance dict for the requested mode."""
    profile = _MODE_TOLERANCE_PROFILE.get(str(mode), 'strict')
    tolerances = dict(DEFAULT_METRIC_TOLERANCES[profile])
    tolerances['csv'] = float(DEFAULT_CSV_TOLERANCES[profile])
    tolerances['npz'] = float(DEFAULT_NPZ_TOLERANCES[profile])
    if override:
        tolerances.update(override)
    tolerances['profile'] = profile
    return tolerances


def _iter_stream_json_files(dir_path: Path) -> dict[str, Path]:
    """Map plain JSON filenames to paths for one directory."""
    return {path.name: path for path in sorted(dir_path.iterdir())
            if path.is_file() and path.suffix.lower() == '.json'}


def _iter_stream_input_json_names(data_path: Path) -> list[str]:
    """Resolve expected output JSON names from one video file or folder."""
    if data_path.is_file():
        if data_path.suffix.lower() not in _STREAM_VIDEO_SUFFIXES:
            raise ValueError(f'Unsupported video input: {data_path}')
        return [f'{data_path.stem}.json']
    if not data_path.is_dir():
        raise FileNotFoundError(data_path)
    return [f'{path.stem}.json' for path in sorted(data_path.iterdir())
            if path.is_file() and path.suffix.lower() in _STREAM_VIDEO_SUFFIXES]


def _stream_json_cmp_score(row: dict[str, Any]) -> float:
    """Rank failures by structural and annotation drift before numeric noise."""
    return (
        int(row['metadata']) * 1_000_000_000
        + int(row['ann_intervals']) * 100_000_000
        + int(row['ann_frames']) * 1_000_000
        + int(row['frame_count']) * 500_000
        + int(row['missing_extra']) * 50_000
        + int(row['timestamps']) * 5_000
        + int(row['det_counts']) * 500
        + float(row['avg_abs']) * 10.0
        + float(row['max_abs'])
    )


def _build_stream_json_cmp_row(file_name: str, ok: bool, cmp_report: dict[str, Any]) -> dict[str, Any]:
    """Flatten one nested compare report into one compact row."""
    metadata = cmp_report.get('metadata', {})
    frame_structure = cmp_report.get('frame_structure', {})
    numeric = cmp_report.get('numeric', {})
    annotations = cmp_report.get('annotations', {})

    row = {'file': file_name,
           'status': 'pass' if ok else 'fail',
           'ok': bool(ok),
           'metadata': len(metadata.get('unequal', {})),
           'frame_count': int(frame_structure.get('frame_count') is not None),
           'missing_extra': len(frame_structure.get('missing_frame_indices', [])) + len(frame_structure.get('extra_frame_indices', [])),
           'timestamps': len(frame_structure.get('timestamp_mismatches', [])),
           'det_counts': len(frame_structure.get('detection_count_mismatches', [])),
           'ann_intervals': 0 if annotations.get('event_intervals_equal', True) else 1,
           'ann_frames': len(annotations.get('frame_annotation_mismatches', [])),
           'avg_abs': float(numeric.get('avg_abs', 0.0)),
           'max_abs': float(numeric.get('max_abs', 0.0)),
           'report': cmp_report,
           }
    row['score'] = _stream_json_cmp_score(row)
    return row


def _print_stream_json_verbose(file_report: dict[str, Any]) -> None:
    """Print one concise expanded comparison block for a single file."""
    report = file_report.get('report', {})
    metadata = report.get('metadata', {})
    frame_structure = report.get('frame_structure', {})
    annotations = report.get('annotations', {})
    numeric = report.get('numeric', {})
    color = 'g' if file_report.get('ok', False) else 'r'

    print_color(f"\n{file_report['file']} [{file_report['status']}]", color)
    if metadata.get('unequal'):
        print(f"  metadata keys: {', '.join(sorted(metadata['unequal']))}")
    if frame_structure.get('frame_count'):
        delta = frame_structure['frame_count']
        print(f"  frame count: j1={delta.get('j1')} j2={delta.get('j2')}")
    if frame_structure.get('missing_frame_indices'):
        print(f"  missing frames: {len(frame_structure['missing_frame_indices'])} "
              f"(first: {frame_structure['missing_frame_indices'][:5]})")
    if frame_structure.get('extra_frame_indices'):
        print(f"  extra frames: {len(frame_structure['extra_frame_indices'])} "
              f"(first: {frame_structure['extra_frame_indices'][:5]})")
    if frame_structure.get('timestamp_mismatches'):
        print(f"  timestamp mismatches: {len(frame_structure['timestamp_mismatches'])} "
              f"(first: {frame_structure['timestamp_mismatches'][:3]})")
    if frame_structure.get('detection_count_mismatches'):
        print(f"  detection-count mismatches: {len(frame_structure['detection_count_mismatches'])} "
              f"(first: {frame_structure['detection_count_mismatches'][:3]})")
    if not annotations.get('event_intervals_equal', True):
        print("  event intervals: mismatch")
    if annotations.get('frame_annotation_mismatches'):
        print(f"  frame annotation mismatches: {len(annotations['frame_annotation_mismatches'])} "
              f"(first: {annotations['frame_annotation_mismatches'][:3]})")
    print(f"  numeric: avg_abs={_fmt_num(float(numeric.get('avg_abs', 0.0)))} "
          f"max_abs={_fmt_num(float(numeric.get('max_abs', 0.0)))} "
          f"path={numeric.get('max_path')}")

# endregion

#* region Run Execution Helpers
def train_sanity_models(cache_dir, out_dir, *, ds_testing=None, stm_testing=None, **kwargs):
    """ Train every `*_train.npz` cache in one directory and return the created run dirs."""
    kwargs = dict(kwargs)
    cache_dir, out_dir = Path(cache_dir),  Path(out_dir)
    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_caches = sorted(cache_dir.glob('*_train.npz'))
    if not train_caches:
        raise FileNotFoundError(f'No *_train.npz caches found in {cache_dir}')

    deterministic = kwargs.pop('deterministic', False)
    if not deterministic:
        return train_models(cache_dir, out_dir, run_tests=False, **kwargs)

    base_seed = kwargs.pop('random_seed', RANDOM_SEED)
    run_dirs = []
    for index, train_cache in enumerate(train_caches):
        _set_deterministic(base_seed + index)
        train_tag = strip_split_suffix(train_cache.stem)
        run_dir = Path(run_training(train_cache, tag=train_tag, work_dir=out_dir, **kwargs))
        run_dirs.append(run_dir)
    return run_dirs


def run_sanity_tests(models_or_ref_dir, *, cache_dir, out_dir, ds_testing=None, stm_testing=None, **kwargs):
    """ Run raw tests for existing models or reference run dirs."""

    cache_dir,out_dir = Path(cache_dir), Path(out_dir)

    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tested_dirs = test_models(models_or_ref_dir, ds_tests=ds_testing, stm_tests=stm_testing,
                              npz_dir=cache_dir, out_dir=out_dir, evaluate=False,
                              infer_threshold=True, **kwargs)
    if not tested_dirs:
        raise FileNotFoundError('No model refs were provided for sanity testing')
    return tested_dirs, []


def run_sanity_eval(source_dir, *, out_dir, raw_results=None, ds_testing=None, stm_testing=None, **kwargs):
    """ Re-run only the evaluation phase from saved raw `*-tst.npz` outputs."""

    def _summary_exists(run_dir: Path, infer_mode: str)-> bool:
        if model_path is None or test_cache is None:
            return False
        summary_name = f"{get_exporting_name(model_path, test_cache, 'summary', unit=infer_mode)}.json"
        return any(p.name == summary_name for p in run_dir.rglob('*.json'))

    def _infer_eval_mode(raw_path:Path) -> str:
        # Old raw NPZ files do not store the eval mode explicitly, so reuse the mode
        # implied by whichever summary file already exists in the source run dir.
        stem = raw_path.stem
        if stem.endswith('_stream-tst'):
            return 'stream'
        if stem.endswith('_video-tst'):
            return 'video'
        run_dir = raw_path.parent
        if _summary_exists(run_dir, 'stream'):
            return 'stream'
        if _summary_exists(run_dir, 'video'):
            return 'video'
        return 'clip'

    def _load_eval_plan(base_dir: Path) -> dict[str, dict[str, Any]]:
        plan_path = base_dir/_EVAL_PLAN_NAME
        if not plan_path.is_file():
            return {}
        with plan_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    model_path,test_cache = None, None
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: switch eval iteration to raw_results` once the explicit-order flow is finalized.
    raw_paths = sorted(source_dir.rglob('*-tst.npz'))
    if not raw_paths:
        raise FileNotFoundError(f'No *-tst.npz files found in {source_dir}')

    eval_plan = _load_eval_plan(source_dir)
    raw_result_keys = None if raw_results is None else {str(Path(p).resolve()) for p in raw_results}
    evaluated = []
    prev_run_dir = None
    for path in raw_paths:
        if raw_result_keys is not None and str(path.resolve()) not in raw_result_keys:
            print_color(f'[WARN] Missing raw result in provided order list: {path}', 'y')
        rel_parent = path.parent.relative_to(source_dir)
        rel_raw = path.relative_to(source_dir).as_posix()
        target_run_dir = out_dir/rel_parent
        target_run_dir.mkdir(parents=True, exist_ok=True)
        if path.parent != prev_run_dir:
            model_tag = strip_timestamp_prefix(path.parent.name)
            print(f'\n\b=== {model_tag} model Evaluation ===\n= evaluated dir : {target_run_dir}')
            prev_run_dir = path.parent
        # The raw test NPZ already carries the model/cache refs needed to rebuild names.
        with np.load(path, allow_pickle=True) as data:
            model_path = data['model_path'].item() if isinstance(data['model_path'], np.ndarray) else data['model_path']
            test_cache = data['test_cache'].item() if isinstance(data['test_cache'], np.ndarray) else data['test_cache']
        plan_item = eval_plan.get(rel_raw, {})
        mode = plan_item.get('mode') or _infer_eval_mode(path)
        threshold = kwargs.get('threshold', plan_item.get('threshold', infer_eval_threshold(path.parent)))
        evaluate_raw_test(path, mode, target_run_dir, threshold, **kwargs)
        evaluated.append(target_run_dir)
    return sorted(set(evaluated))

# endregion

# region Output Comparison Helpers
def _compare_csv_with_tolerance(test_path: Path, ref_path: Path, *, atol: float) -> dict[str, Any]:
    """Compare one CSV semantically, allowing small numeric drift."""
    with test_path.open('r', encoding='utf-8', newline='') as f:
        sample = f.readline()
        f.seek(0)
        delim = ';' if sample.count(';') > sample.count(',') else ','
        test_rows = list(csv.reader(f, delimiter=delim))
    with ref_path.open('r', encoding='utf-8', newline='') as f:
        sample = f.readline()
        f.seek(0)
        delim = ';' if sample.count(';') > sample.count(',') else ','
        ref_rows = list(csv.reader(f, delimiter=delim))
    if len(test_rows) != len(ref_rows):
        return {'ok': False, 'kind': 'csv', 'max_fp_err': 0.0,
                'message': f'row count mismatch: {len(test_rows)} != {len(ref_rows)}',
                'issue_type': 'shape', 'issue_count': 1}
    worst = 0.0
    for row_index, (test_row, ref_row) in enumerate(zip(test_rows, ref_rows)):
        if len(test_row) != len(ref_row):
            return {'ok': False, 'kind': 'csv', 'max_fp_err': worst,
                    'message': f'column count mismatch at row {row_index}: {len(test_row)} != {len(ref_row)}',
                    'issue_type': 'shape', 'issue_count': 1}
        for col_index, (test_cell, ref_cell) in enumerate(zip(test_row, ref_row)):
            try:
                delta = abs(float(test_cell) - float(ref_cell))
                worst = max(worst, delta)
            except ValueError:
                if test_cell != ref_cell:
                    return {'ok': False, 'kind': 'csv', 'max_fp_err': worst,
                            'message': f'non-numeric mismatch at row {row_index}, col {col_index}: {test_cell!r} != {ref_cell!r}',
                            'issue_type': 'text', 'issue_count': 1}
    if worst > float(atol):
        return {'ok': False, 'kind': 'csv', 'max_fp_err': worst,
                'message': f'numeric drift {worst:.9g} exceeds tolerance {float(atol):.9g}',
                'issue_type': 'numeric', 'issue_count': 1}
    return {'ok': True, 'kind': 'csv', 'max_fp_err': worst, 'message': None, 'issue_count': 0}


def _compare_timeline_csv(test_path: Path, ref_path: Path, *, atol: float) -> dict[str, Any]:
    """Compare one timeline CSV by label flips and probability drift."""
    with test_path.open('r', encoding='utf-8', newline='') as f:
        sample = f.readline()
        f.seek(0)
        delim = ';' if sample.count(';') > sample.count(',') else ','
        test_rows = list(csv.DictReader(f, delimiter=delim))
    with ref_path.open('r', encoding='utf-8', newline='') as f:
        sample = f.readline()
        f.seek(0)
        delim = ';' if sample.count(';') > sample.count(',') else ','
        ref_rows = list(csv.DictReader(f, delimiter=delim))

    if len(test_rows) != len(ref_rows):
        return {'ok': False, 'kind': 'csv', 'max_fp_err': 0.0,
                'message': f'row count mismatch: {len(test_rows)} != {len(ref_rows)}',
                'issue_type': 'shape', 'issue_count': 1}

    req_cols = {'win_idx', 't_frm', 't_start', 'n_frm', 'gt_label', 'y_prob', 'y_pred'}
    if test_rows and (set(test_rows[0]) != set(ref_rows[0]) or not req_cols.issubset(test_rows[0])):
        return {'ok': False, 'kind': 'csv', 'max_fp_err': 0.0,
                'message': 'timeline columns mismatch', 'issue_type': 'shape', 'issue_count': 1}

    flip_count, max_prob_delta = 0, 0.0
    for row_idx, (test_row, ref_row) in enumerate(zip(test_rows, ref_rows)):
        for key in ('win_idx', 'n_frm', 'gt_label'):
            if test_row[key] != ref_row[key]:
                return {'ok': False, 'kind': 'csv', 'max_fp_err': max_prob_delta,
                        'message': f'{key} mismatch at row {row_idx}', 'issue_type': 'shape', 'issue_count': 1}
        for key in ('t_frm', 't_start'):
            if abs(float(test_row[key]) - float(ref_row[key])) > 1e-9:
                return {'ok': False, 'kind': 'csv', 'max_fp_err': max_prob_delta,
                        'message': f'{key} mismatch at row {row_idx}', 'issue_type': 'shape', 'issue_count': 1}

        prob_delta = abs(float(test_row['y_prob']) - float(ref_row['y_prob']))
        max_prob_delta = max(max_prob_delta, prob_delta)
        if int(test_row['y_pred']) != int(ref_row['y_pred']):
            flip_count += 1

    flip_rate = (flip_count / len(test_rows)) if test_rows else 0.0
    ok = (flip_count == 0 and max_prob_delta <= float(atol))
    return {'ok': ok,
            'kind': 'csv',
            'csv_style': 'timeline',
            'max_fp_err': max_prob_delta,
            'message': (None if ok else 'timeline drift'),
            'issue_count': int(flip_count > 0 or max_prob_delta > float(atol)),
            'flip_count': flip_count,
            'flip_rate': flip_rate,
            'rows': len(test_rows),
            'max_prob_delta': max_prob_delta}


def _compare_json_semantics(test_path: Path, ref_path: Path, *, tolerances: dict[str, float]) -> dict[str, Any]:
    """ Compare one JSON semantically, ignoring path-only differences."""
    summary = {'numeric_count': 0, 'other_count': 0}
    numeric_issues, other_issues = [], []

    def _canonicalize_json_value(value: Any) -> Any:
        if isinstance(value, dict):
            canon = {}
            for key, item in value.items():
                if key in _JSON_PATH_KEYS:
                    canon[key] = '<PATH>' if item is not None else None
                else:
                    canon[key] = _canonicalize_json_value(item)
            return canon
        if isinstance(value, list):
            return [_canonicalize_json_value(item) for item in value]
        if isinstance(value, str) and ('/' in value or value.endswith(('.csv', '.json', '.npz', '.png', '.pt'))):
            return '<PATH>'
        return value

    def _compare_json_values(test_value: Any, ref_value: Any, *, path: str) -> tuple[float, list[str]]:
        fp_err_max = 0.0
        issues_ls = []

        if isinstance(test_value, dict) and isinstance(ref_value, dict):
            test_keys = set(test_value)
            ref_keys = set(ref_value)
            for key in sorted(test_keys ^ ref_keys):
                issues_ls.append(f'{path}.{key}'.lstrip('.') + ': missing key')
                summary['other_count'] += 1
            for key in sorted(test_keys & ref_keys):
                child_path = f'{path}.{key}' if path else str(key)
                child_max, child_issues = _compare_json_values(test_value[key], ref_value[key], path=child_path)
                fp_err_max = max(fp_err_max, child_max)
                issues_ls.extend(child_issues)
            return fp_err_max, issues_ls

        if isinstance(test_value, list) and isinstance(ref_value, list):
            if len(test_value) != len(ref_value):
                issues_ls.append(f'{path}: list length mismatch {len(test_value)} != {len(ref_value)}')
                summary['other_count'] += 1
                return fp_err_max, issues_ls
            for index, (test_item, ref_item) in enumerate(zip(test_value, ref_value)):
                child_path = f'{path}[{index}]'
                child_max, child_issues = _compare_json_values(test_item, ref_item, path=child_path)
                fp_err_max = max(fp_err_max, child_max)
                issues_ls.extend(child_issues)
            return fp_err_max, issues_ls

        if isinstance(test_value, bool) or isinstance(ref_value, bool):
            if test_value != ref_value:
                issues_ls.append(f'{path}: {test_value!r} != {ref_value!r}')
                summary['other_count'] += 1
                other_issues.append(path)
            return fp_err_max, issues_ls

        if isinstance(test_value, (float, int)) and isinstance(ref_value, (float, int)):
            delta = abs(float(test_value) - float(ref_value))
            fp_err_max = max(fp_err_max, delta)
            tol = float(tolerances[path]) if path in tolerances else float(tolerances.get(path.rsplit('.', 1)[-1], 1e-6))
            if delta > tol:
                issues_ls.append(f'{path}: numeric drift {delta:.9g} exceeds tolerance {tol:.9g}')
                summary['numeric_count'] += 1
                numeric_issues.append({'path': path, 'delta': delta, 'tol': tol})
            return fp_err_max, issues_ls

        if test_value != ref_value:
            issues_ls.append(f'{path}: {test_value!r} != {ref_value!r}')
            summary['other_count'] += 1
            other_issues.append(path)
        return fp_err_max, issues_ls

    with test_path.open('r', encoding='utf-8') as f:
        test_json = _canonicalize_json_value(json.load(f))
    with ref_path.open('r', encoding='utf-8') as f:
        ref_json = _canonicalize_json_value(json.load(f))
    max_fp_err, issues = _compare_json_values(test_json, ref_json, path='')
    return {'ok': not issues, 'kind': 'json', 'max_fp_err': max_fp_err,
            'message': '; '.join(issues[:3]) if issues else None,
            'issue_count': len(issues),
            'numeric_count': summary['numeric_count'],
            'other_count': summary['other_count'],
            'numeric_issues': numeric_issues,
            'other_issues': other_issues,
            'test_json': test_json,
            'ref_json': ref_json}


def _summary_sample_total(summary_json: dict[str, Any]) -> int:
    """Resolve one total sample count for summary percentage display."""
    testing_set = summary_json.get('testing_set', {})
    for key in ('clips_num', 'videos_num'):
        if key in testing_set:
            return int(testing_set[key])
    cm = summary_json.get('confusion_matrix', None)
    if isinstance(cm, list) and len(cm) == 2:
        return int(sum(sum(int(cell) for cell in row) for row in cm))
    cm = summary_json.get('cm_clips', None)
    if isinstance(cm, dict):
        return int(sum(int(value) for value in cm.values()))
    return 0


def _summary_cm_cells(summary_json: dict[str, Any]) -> list[int]:
    """ Extract one flattened confusion-matrix cell list from a summary JSON."""
    cm = summary_json.get('confusion_matrix', None)
    if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
        return [int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])]
    cm = summary_json.get('cm_clips', None)
    if isinstance(cm, dict):
        return [int(cm.get('tn', 0)), int(cm.get('fp', 0)), int(cm.get('fn', 0)), int(cm.get('tp', 0))]
    return []


def _compare_summary_json(test_path: Path, ref_path: Path, *, tolerances: dict[str, float], mode='train') -> dict[str, Any]:
    """Compare one summary JSON and keep only summary-level drift metrics."""
    result = _compare_json_semantics(test_path, ref_path, tolerances=tolerances)
    test_json = result.pop('test_json')
    ref_json = result.pop('ref_json')

    metric_alias = {'ROC AUC': 'auc'}
    summary_metrics = {'accuracy', 'precision', 'recall', 'f1', 'auc', 'TPR', 'FPR'}
    metric_issues = []
    for issue in result.get('numeric_issues', []):
        path = issue['path']
        if path.startswith('confusion_matrix') or path.startswith('cm_clips'):
            continue
        metric_key = metric_alias.get(path, path.rsplit('.', 1)[-1])
        if metric_key in summary_metrics:
            metric_issues.append({'metric': metric_key, 'delta': float(issue['delta'])})

    sample_total  = _summary_sample_total(ref_json)
    ref_cm_cells  = _summary_cm_cells(ref_json)
    test_cm_cells = _summary_cm_cells(test_json)
    cm_sum_delta = 0
    if ref_cm_cells and test_cm_cells and len(ref_cm_cells) == len(test_cm_cells):
        cm_sum_delta = sum(abs(int(test_v) - int(ref_v)) for test_v, ref_v in zip(test_cm_cells, ref_cm_cells))
    cm_delta = (cm_sum_delta / sample_total) if sample_total else 0.0
    max_abs_error = max((float(item['delta']) for item in metric_issues), default=0.0)
    summary_style = 'stream_summary' if str(test_json.get('analysis_mode', '')).lower() == 'stream' else 'summary'
    metric_tol = float(tolerances.get('max_abs_error', 0.0))
    cm_tol = float(tolerances.get('cm_delta', 0.0))
    other_count = int(result.get('other_count', 0))
    metric_issue = int(max_abs_error > metric_tol)
    cm_issue = int(cm_delta > cm_tol)
    issue_count = metric_issue + other_count + cm_issue
    ok = (issue_count == 0)

    result.update({'ok': ok, 'message': (None if ok else result.get('message')),
                   'issue_count': issue_count,
                   'numeric_count': metric_issue + cm_issue,
                   'json_style': summary_style,
                   'max_abs_error': max_abs_error,
                   'metric_tol': metric_tol,
                   'cm_sum_delta': cm_sum_delta,
                   'sample_total': sample_total,
                   'cm_delta': cm_delta, 'cm_tol': cm_tol,})
    return result


def _compare_stream_events_json(test_path: Path, ref_path: Path, *, tolerances: dict[str, float]) -> dict[str, Any]:
    """Compare one stream-events JSON using one compact event-flip summary."""
    result = _compare_json_semantics(test_path, ref_path, tolerances=tolerances)
    test_json = result.pop('test_json')
    ref_json = result.pop('ref_json')
    test_videos = test_json.get('videos', [])
    ref_videos = ref_json.get('videos', [])

    def _event_totals(videos: list[dict[str, Any]]) -> dict[str, int]:
        return {'pred': sum(int(video.get('pred_events_num', 0)) for video in videos),
                'matched': sum(int(video.get('detected_events_num', 0)) for video in videos),
                'missed': sum(int(video.get('missed_events_num', 0)) for video in videos),
                'false': sum(int(video.get('false_events_num', 0)) for video in videos),
                'gt': sum(int(video.get('gt_events_num', 0)) for video in videos),}

    def _event_f1(totals: dict[str, int]) -> float:
        matched = float(totals['matched'])
        denom = (2.0*matched) + float(totals['false']) + float(totals['missed'])
        return  (2.0*matched/denom) if denom > 0 else 0.0

    changed = 0
    ref_by_video = {video.get('video'): video for video in ref_videos}
    test_by_video = {video.get('video'): video for video in test_videos}
    for video_name in sorted(set(ref_by_video) | set(test_by_video)):
        if test_by_video.get(video_name, {}) != ref_by_video.get(video_name, {}):
            changed += 1

    ref_totals = _event_totals(ref_videos)
    test_totals = _event_totals(test_videos)
    delta_cm = [abs(test_totals['matched'] - ref_totals['matched']),
                abs(test_totals['missed'] - ref_totals['missed']),
                abs(test_totals['false'] - ref_totals['false']),
                abs(test_totals['matched'] - ref_totals['matched'])]
    ref_cm = [ref_totals['matched'], ref_totals['missed'], ref_totals['false'], ref_totals['matched']]
    event_flips = int(sum(delta_cm))
    event_total = int(sum(ref_cm))
    f1_delta = _event_f1(test_totals) - _event_f1(ref_totals)
    other_count = int(result.get('other_count', 0))
    f1_tol = float(tolerances.get('event_f1', 0.0))
    f1_issue = int(abs(f1_delta) > f1_tol)
    gt_changed = (test_totals['gt'] != ref_totals['gt'])
    issue_count = int(event_flips > 0) + int(gt_changed) + other_count + f1_issue
    result.update({'json_style': 'stream_events',
                   'ok': (issue_count == 0),
                   'issue_count': issue_count,
                   'numeric_count': f1_issue,
                   'message': (None if issue_count == 0 else result.get('message') or 'stream event drift'),
                   'videos_changed': changed,
                   'videos_total': len(ref_videos),
                   'event_flips': event_flips,
                   'event_total': event_total,
                   'delta_cm': delta_cm,
                   'ref_cm': ref_cm,
                   'f1_delta': f1_delta,
                   'gt_changed': gt_changed,
                   'event_f1_tol': f1_tol,
                   'pred_event_delta': abs(test_totals['pred'] - ref_totals['pred']),
                   'matched_delta': abs(test_totals['matched'] - ref_totals['matched']),
                   'missed_delta': abs(test_totals['missed'] - ref_totals['missed']),
                   'false_delta': abs(test_totals['false'] - ref_totals['false']),})
    return result


def _cmp_png_pix(test_path: Path, ref_path: Path) -> dict[str, Any]:
    """Compare one PNG by decoded pixels rather than file bytes."""
    from PIL import Image, ImageChops

    test_img = Image.open(test_path).convert('RGBA')
    ref_img = Image.open(ref_path).convert('RGBA')
    if test_img.size != ref_img.size:
        return {'ok': False, 'kind': 'png', 'max_fp_err': 0.0,
                'message': f'image size mismatch: {test_img.size} != {ref_img.size}',
                'issue_type': 'size'}
    diff = ImageChops.difference(test_img, ref_img)
    if diff.getbbox() is not None:
        return {'ok': False, 'kind': 'png', 'max_fp_err': 0.0,
                'message': 'pixel content differs', 'issue_type': 'pixels'}
    return {'ok': True, 'kind': 'png', 'max_fp_err': 0.0, 'message': None}


def _compare_npz_semantics(test_path: Path, ref_path: Path, *, y_prob_atol: float)-> dict[str, Any]:
    """Compare one NPZ semantically, allowing tiny `y_prob` drift in no-train mode."""
    with np.load(test_path, allow_pickle=True) as test_npz, np.load(ref_path, allow_pickle=True) as ref_npz:
        test_keys = set(test_npz.files)
        ref_keys = set(ref_npz.files)
        if test_keys != ref_keys:
            return {'ok': False, 'kind': 'npz', 'max_fp_err': 0.0,
                    'message': f'npz key mismatch: {sorted(test_keys ^ ref_keys)}',
                    'issue_key': 'keys'}

        max_fp_err = 0.0
        for key in sorted(test_keys):
            test_value = test_npz[key]
            ref_value = ref_npz[key]
            if key in {'model_path', 'test_cache'}:
                continue
            if key == 'y_prob':
                max_delta = float(np.max(np.abs(test_value - ref_value)))
                max_fp_err = max(max_fp_err, max_delta)
                if not np.allclose(test_value, ref_value, rtol=0.0, atol=float(y_prob_atol), equal_nan=True):
                    return {'ok': False, 'kind': 'npz', 'max_fp_err': max_fp_err,
                            'message': f'y_prob drift {max_delta:.9g} exceeds tolerance {float(y_prob_atol):.9g}',
                            'issue_key': 'y_prob'}
                continue
            try:
                values_equal = np.array_equal(test_value, ref_value, equal_nan=True)
            except TypeError:
                values_equal = np.array_equal(test_value, ref_value)
            if not values_equal:
                return {'ok': False, 'kind': 'npz', 'max_fp_err': max_fp_err,
                        'message': f'npz payload mismatch in {key}',
                        'issue_key': key}
    return {'ok': True, 'kind': 'npz', 'max_fp_err': max_fp_err, 'message': None}


def _cmp_op_file(test_path:Path, ref_path:Path, *, mode: str, file_key='',
                 csv_atol: float, npz_y_prob_atol: float, json_tolerances: dict[str,float])-> dict[str, Any]:
    """Dispatch one semantic file comparison by suffix."""
    test_path, ref_path = Path(test_path), Path(ref_path)
    key_path = Path(file_key) if file_key else test_path
    suffix = key_path.suffix.lower()[1:]
    if suffix == 'json':
        if key_path.name.endswith('_stream-events.json'):
            return _compare_stream_events_json(test_path, ref_path, tolerances=json_tolerances)
        if key_path.name.endswith('-summary.json'):
            return _compare_summary_json(test_path, ref_path, tolerances=json_tolerances, mode=mode)
        return _compare_json_semantics(test_path, ref_path, tolerances=json_tolerances)
    elif suffix == 'png':
        return _cmp_png_pix(test_path, ref_path)
    elif suffix == 'csv':
        if key_path.stem.startswith('timeline_'):
            return _compare_timeline_csv(test_path, ref_path, atol=csv_atol)
        return _compare_csv_with_tolerance(test_path, ref_path, atol=csv_atol)
    elif suffix == 'npz':
        if mode == 'no_train':
            return _compare_npz_semantics(test_path, ref_path, y_prob_atol=npz_y_prob_atol)
        return {'ok': True, 'max_fp_err': 0.0, 'message': None, 'kind': 'npz'}
    return {'ok': True, 'max_fp_err': 0.0, 'message': None, 'kind': 'other'}


def _compare_run_outputs(test_run_dir: Path, ref_run_dir: Path, *, mode: str, csv_atol: float,
                         y_prob_atol: float, json_tolerances: dict[str, float]) -> dict[str, Any]:
    """Compare all saved outputs for one run dir against its reference run."""

    def _normalize_output_name(rel_path,  run_tag: str) -> str:
    # Todo: generalize it into function that search equivalent output file
        rel_obj = Path(rel_path)
        name, stem = rel_obj.name, rel_obj.stem

        if rel_obj.suffix.lower() == '.png' and stem.startswith('timeline_'):
            tail = stem[len('timeline_'):]
            if tail.startswith(f'{run_tag}_'):
                tail = tail[len(run_tag) + 1:]
            return str(rel_obj.with_name(f'timeline_{tail}.png'))

        if rel_obj.suffix.lower() == '.csv':
            if stem.startswith('timeline_'):
                tail = stem[len('timeline_'):]
                if tail.startswith(f'{run_tag}_'):
                    tail = tail[len(run_tag) + 1:]
                return str(rel_obj.with_name(f'timeline_{tail}.csv'))
            match = re.match(r'^.+_stream-tst_(.+)_timeline$', stem)
            if match:
                return str(rel_obj.with_name(f'timeline_{match.group(1)}.csv'))
        return rel_path

    def _collect_run_outputs(run_dir: Path) -> dict[str, Any]:
        run_tag = strip_timestamp_prefix(run_dir.name)
        files = {}
        for pattern in _EXPORTED_FILE_PATTERNS:
            for path in run_dir.rglob(pattern):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(run_dir).as_posix()
                #* Old and new timeline export names differ, _normalize_output_name
                #* allows comparison across different run dirs.
                files[_normalize_output_name(rel_path, run_tag)] = str(path)
        return {'run_dir': str(run_dir), 'files': files}

    test_outputs = _collect_run_outputs(Path(test_run_dir))
    ref_outputs = _collect_run_outputs(Path(ref_run_dir))
    test_files = test_outputs['files']
    ref_files = ref_outputs['files']
    missing = sorted(set(ref_files) - set(test_files))
    extra = sorted(set(test_files) - set(ref_files))
    common_files = sorted(set(test_files) & set(ref_files))

    mismatch_counts = {'json': 0, 'csv': 0, 'png': 0, 'npz': 0}
    semantic_issues = []
    max_fp_err = 0.0
    for f in common_files:
        result = _cmp_op_file(Path(test_files[f]), Path(ref_files[f]), mode=mode, file_key=f, csv_atol=csv_atol,
                              npz_y_prob_atol=y_prob_atol, json_tolerances=json_tolerances, )

        max_fp_err = max(max_fp_err, float(result.get('max_fp_err', 0.0)))
        kind = result.get('kind', 'other')
        if not result.get('ok', True):
            if kind in mismatch_counts:
                mismatch_counts[kind] += 1
            issue = {'file': f,
                     'kind': kind,
                     'message': result.get('message'),
                     'max_fp_err': float(result.get('max_fp_err', 0.0)),
                     'issue_count': int(result.get('issue_count', 1)),
                     'numeric_count': int(result.get('numeric_count', 0)),
                     'other_count': int(result.get('other_count', 0)),
                     'issue_type': result.get('issue_type', None),
                     'issue_key': result.get('issue_key', None),}
            for key in ('csv_style', 'flip_count', 'flip_rate', 'rows', 'max_prob_delta',
                        'json_style', 'max_abs_error', 'metric_tol',
                        'cm_sum_delta', 'sample_total', 'cm_delta', 'cm_tol',
                        'videos_changed', 'videos_total', 'pred_event_delta', 'pred_event_total',
                        'matched_delta', 'matched_total', 'missed_delta', 'missed_total',
                        'false_delta', 'false_total', 'gt_changed', 'event_flips', 'event_total',
                        'delta_cm', 'ref_cm', 'f1_delta'):
                if key in result:
                    issue[key] = result[key]
            semantic_issues.append(issue)

    return {'status': ('fail' if (missing or extra or semantic_issues) else 'pass'),
            'new_run_dir': str(test_run_dir),
            'ref_run_dir': str(ref_run_dir),
            'files_missing': missing,
            'files_extra': extra,
            'max_fp_err': max_fp_err,
            'mismatches': mismatch_counts,
            'semantic_issues': semantic_issues,}


#* Semantic report formatting
def _build_comparison_report(test_dir, ref_dir, *, mode: str) -> dict[str, Any]:
    """ Build one per-run semantic comparison report for two output trees."""
    test_dir, ref_dir = Path(test_dir), Path(ref_dir)
    tolerances = _resolve_tolerances(mode)

    test_runs = {strip_timestamp_prefix(path.name): path for path in _iter_run_dirs(test_dir)}
    ref_runs = {strip_timestamp_prefix(path.name): path for path in _iter_run_dirs(ref_dir)}
    missing_runs = sorted(set(ref_runs) - set(test_runs))
    extra_runs = sorted(set(test_runs) - set(ref_runs))
    status = 'fail' if (missing_runs or extra_runs) else 'pass'

    run_reports = []
    for run_key in sorted(set(test_runs) & set(ref_runs)):
        run_report = _compare_run_outputs(test_runs[run_key], ref_runs[run_key], mode=mode,
                                          csv_atol=float(tolerances['csv']), y_prob_atol=float(tolerances['npz']),
                                          json_tolerances=tolerances, )
        run_report['run'] = run_key
        if run_report['status'] == 'fail':
            status = run_report['status']
        run_reports.append(run_report)

    return {'status': status,
            'missing_runs': missing_runs,
            'extra_runs': extra_runs,
            'runs': run_reports,}


def _print_cli_report(report: dict[str, Any]) -> None:
    """ Print one compact per-run sanity summary table."""

    rows = []
    for run in report.get('comparison', {}).get('runs', []):
        rows.append({'run': run['run'],
                     'files_missing': len(run.get('files_missing', [])),
                     'files_extra': len(run.get('files_extra', [])),
                     'max_fp_err': run.get('max_fp_err', 0.0),
                     'json': run.get('mismatches', {}).get('json', 0),
                     'csv' : run.get('mismatches', {}).get('csv',  0),
                     'png' : run.get('mismatches', {}).get('png',  0),
                     })
    if not rows:
        print(f"sanity status: {report['status']}")
        return

    w_2r = {'run': 3, 'missing': 6, 'extra': 6, 'max': 8, 'json': 4, 'csv': 4, 'png': 4}
    for row in rows:
        w_2r['run'] = max(w_2r['run'], len(str(row['run'])))

    w_1r = {'run':w_2r['run'], 'files':w_2r['missing'] + w_2r['extra'] + 3,
            'fp_error': w_2r['max'], 'mismatches':(w_2r['json'] + w_2r['csv'] + w_2r['png'] )}
    print(f"\nsanity status: {report['status']}")


    print(f"{'run':^{w_1r['run']}} | {'files':^{w_1r['files']}} | fp error | {'mismatches':^{w_1r['mismatches'] }}")
    print(f"{'':<{w_2r['run']}} |{'missing':^{ w_2r['missing']}} | {'extra':<{w_2r['extra']}} | "
          f"{'max':^{w_2r['max']}} | json | csv | png")
    print('-+-'.join(('-'*w_2r['run'], '-'*w_1r['files'], '-'*w_1r['fp_error'] , '-'*(w_1r['mismatches']+4))))
    for row in rows:
        print(' | '.join((f"{row['run']:^{w_2r['run']}}",
                          f"{row['files_missing']:^{w_2r['missing']}} | {row['files_extra']:^{w_2r['extra']}}",
                          f"{row['max_fp_err']:^{w_2r['max']}.1e}",
                          f"{row['json']:^{w_2r['json']}} | {row['csv']:^{w_2r['csv']}} | {row['png']:^{w_2r['png']}}",
               )))
def _print_issues(issues: dict[str, Any], mode: str) -> None:  #149
    """Print one run mismatch block, with richer summaries in train mode."""
    kind_order = {'npz': 0, 'csv': 1, 'json': 2, 'png': 3}
    M_ZERO = [0, 0, 0, 0]
    cur_issue: dict[str, Any] = {}

    def _mat2str(m) -> str:
        return f"[{m[0]}, {m[1]}]/[{m[2]}, {m[3]}]"

    def _line_color(delta: float, tol: float = 0.0) -> str:
        return 'g' if abs(float(delta)) <= float(tol) else 'r'

    def _add_line(lines: list[tuple[str, str]], text: str, delta: float, tol: float = 0.0) -> None:
        if abs(float(delta)) == 0.0:
            return
        lines.append((text, _line_color(delta, tol)))

    def _render_file_block(file_name:str,lines: list[tuple[str, str]]) -> bool:
        if not lines:
            return False
        print(f'-{file_name}:')
        for text, color in lines:
            print_color(f'\t{text}', color)
        return True

    def _print_csv_issue() -> bool:
        file_name = cur_issue.get('file', '')
        lines: list[tuple[str, str]] = []
        if cur_issue.get('csv_style', None) == 'timeline':
            flip_count = int(cur_issue.get('flip_count', 0))
            rows = int(cur_issue.get('rows', 0))
            flip_rate = float(cur_issue.get('flip_rate', 0.0))
            max_prob_delta = float(cur_issue.get('max_prob_delta', 0.0))
            _add_line(lines, f'flips = {flip_count}/{rows} ({100.0 * flip_rate:.2f}%)', flip_count)
            _add_line(lines, f'max prob delta = {_fmt_num(max_prob_delta)}',
                      max_prob_delta, tolerances.get('csv', 0.0))
            return _render_file_block(file_name, lines)
        if cur_issue.get('issue_type', None) == 'numeric':
            max_fp_err = float(cur_issue.get('max_fp_err', 0.0))
            _add_line(lines, f'max drift = {_fmt_num(max_fp_err)}', max_fp_err, tolerances.get('csv', 0.0))
            return _render_file_block(file_name, lines)
        lines.append((f"csv {cur_issue.get('message', 'mismatch')}", 'r'))
        return _render_file_block(file_name, lines)

    def _print_json_issue() -> bool:
        file_name = cur_issue.get('file', '')
        json_style = cur_issue.get('json_style', None)
        lines: list[tuple[str, str]] = []
        if json_style == 'stream_events':
            videos_changed = cur_issue.get('videos_changed', 0)
            event_flips = cur_issue.get('event_flips', 0)
            f1_delta = cur_issue.get('f1_delta', 0.0)
            _add_line(lines, f"videos changed = {videos_changed}/{cur_issue.get('videos_total', 0)}", videos_changed)
            _add_line(lines, f"flipped events = {event_flips}/{cur_issue.get('event_total', 0)} "
                                 f"\tCM ref:{_mat2str(cur_issue.get('ref_cm', M_ZERO))}\t delta:{_mat2str(cur_issue.get('delta_cm', M_ZERO))}", event_flips)
            _add_line(lines, f"f1 delta = {_fmt_num(f1_delta)}", f1_delta, cur_issue.get('event_f1_tol', tolerances.get('event_f1', 0.0)))
            if cur_issue.get('gt_changed', False):
                lines.append(('gt changed = yes', 'r'))
            return _render_file_block(file_name, lines)

        if json_style in {'stream_summary', 'summary'}:
            max_abs_err = float(cur_issue.get('max_abs_error', 0.0))
            cm_sum_delta = float(cur_issue.get('cm_sum_delta', 0.0))
            cm_delta = float(cur_issue.get('cm_delta', 0.0))
            _add_line(lines, f'max metric error = {_fmt_num(max_abs_err)}',
                      max_abs_err, cur_issue.get('metric_tol', tolerances.get('max_abs_error', 0.0)))
            _add_line(lines, f"cm delta = {_fmt_num(cm_sum_delta)}/{cur_issue.get('sample_total', 0)} ({100.0 * cm_delta:.3f}%)",
                      cm_delta,  cur_issue.get('cm_tol', tolerances.get('cm_delta', 0.0)) )
            return _render_file_block(file_name, lines)

        numeric_count = int(cur_issue.get('numeric_count', 0))
        other_count = int(cur_issue.get('other_count', 0))
        max_fp_err = float(cur_issue.get('max_fp_err', 0.0))
        if numeric_count:
            lines.append((f'json drifts={numeric_count}', 'r'))
        if other_count:
            lines.append((f'json other={other_count}', 'r'))
        _add_line(lines, f'max = {_fmt_num(max_fp_err)}', max_fp_err)
        if not lines:
            lines.append((f"json issues={cur_issue.get('issue_count', 1)}", 'r'))
        return _render_file_block(file_name, lines)

    def _print_npz_issue() -> bool:
        file_name = cur_issue.get('file', '')
        lines: list[tuple[str, str]] = []
        issue_key = cur_issue.get('issue_key', None)
        if issue_key == 'y_prob':
            max_fp_err = float(cur_issue.get('max_fp_err', 0.0))
            _add_line(lines, f'npz y_prob max drift = {_fmt_num(max_fp_err)}',
                      max_fp_err, tolerances.get('npz', 0.0))
            return _render_file_block(file_name, lines)
        if issue_key is not None:
            lines.append((f'npz mismatch key = {issue_key}', ''))
            return _render_file_block(file_name, lines)
        lines.append(('npz mismatch', 'r'))
        return _render_file_block(file_name, lines)

    def _print_png_issue() -> bool:
        file_name = cur_issue.get('file', '')
        lines: list[tuple[str, str]] = []
        issue_type = cur_issue.get('issue_type', None)
        if issue_type == 'pixels':
            lines.append(('png pixel mismatch', 'r'))
            return _render_file_block(file_name, lines)
        if issue_type == 'size':
            lines.append(('png size mismatch', 'r'))
            return _render_file_block(file_name, lines)
        lines.append(('png mismatch', 'r'))
        return _render_file_block(file_name, lines)

    def _print_issue() -> bool:
        kind = cur_issue.get('kind', 'other')
        if kind == 'json':
            return _print_json_issue()
        if kind == 'csv':
            return _print_csv_issue()
        if kind == 'npz':
            return _print_npz_issue()
        if kind == 'png':
            return _print_png_issue()
        return _render_file_block(cur_issue.get('file', ''), [(cur_issue.get('message', 'mismatch'), 'r')])

    #*** func code ***#
    tolerances = _resolve_tolerances(mode)
    if not (issues.get('files_missing') or issues.get('files_extra') or issues.get('semantic_issues')):
        return
    print_color(f"\n{issues['run']}:", 'r')
    if issues.get('files_missing'):
        print_color(f"  missing files: {len(issues['files_missing'])}", 'r')
    if issues.get('files_extra'):
        print_color(f"  extra files  : {len(issues['files_extra'])}", 'r')

    for cur_issue in sorted(issues.get('semantic_issues', []),
                            key=lambda item: (kind_order.get(item.get('kind', 'other'), 99), item.get('file', ''))):
        _print_issue()

# endregion

#* region Public Sanity API
def test_stream_jsons(tst, ref, op_dir=None, **kwargs):
    """Compare generated stream JSON dirs and optionally save one JSON report."""
    from json_stream_utils import compare_stream_json
    def _resolve_tolerance(tol=None) -> dict[str, float]:
        """ Resolve stream JSON numeric tolerances for this comparison."""
        if isinstance(tol, str):
            mode = tol.lower()
            if mode == 'loose':
                return  {'avg_abs': LOOSE_TOLERANCES, 'max_abs': LOOSE_TOLERANCES}
            if mode == 'strict':
                return {'avg_abs': STRICT_TOLERANCES, 'max_abs': STRICT_TOLERANCES}
        elif isinstance(tol, dict):
            return tol
        elif isinstance(tol, (list, tuple)) and len(tol) == 2:
            return {'avg_abs': tol[0], 'max_abs': tol[1]}
        else:
            return  {'avg_abs': LOOSE_TOLERANCES, 'max_abs': LOOSE_TOLERANCES}


    tst = Path(tst)
    ref = Path(ref)
    op_dir = tst if op_dir is None else Path(op_dir)
    if not tst.is_dir():
        raise NotADirectoryError(tst)
    if not ref.is_dir():
        raise NotADirectoryError(ref)
    op_dir.mkdir(parents=True, exist_ok=True)

    selected_names = kwargs.pop('file_names', None)
    tolerances = _resolve_tolerance(kwargs.pop('tolerances', None))
    ignore_path_fields = bool(kwargs.pop('ignore_path_fields', True))
    print_cli = bool(kwargs.pop('print_cli', False))
    verbose = bool(kwargs.pop('verbose', False))
    save_json = bool(kwargs.pop('save_json', True))
    report_name = kwargs.pop('report_name', 'stream_json_sanity_report')

    tst_files = _iter_stream_json_files(tst)
    ref_files = _iter_stream_json_files(ref)
    names = sorted(set(tst_files) & set(ref_files))
    if selected_names is not None:
        selected_names = set(as_collection(selected_names))
        names = [name for name in names if name in selected_names]
    files = []
    for file_name in names:
        ok, cmp_report = compare_stream_json(tst_files[file_name], ref_files[file_name],
                                             tolerances=tolerances, ignore_path_fields=ignore_path_fields)
        files.append(_build_stream_json_cmp_row(file_name, ok, cmp_report))

    tst_names = set(tst_files) if selected_names is None else set(name for name in selected_names if name in tst_files)
    ref_names = set(ref_files) if selected_names is None else set(name for name in selected_names if name in ref_files)
    report = {'status': 'pass' if (not (tst_names - ref_names) and not (ref_names - tst_names) and
                                   all(item['ok'] for item in files)) else 'fail',
              'data_path': str(tst),
              'ref_dir': str(ref),
              'output_dir': str(op_dir),
              'missing_refs': sorted(tst_names - ref_names),
              'extra_refs': sorted(ref_names - tst_names),
              'files': files,
              }
    if save_json:
        report_path = op_dir / f'{report_name}.json'
        with report_path.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        report['report_path'] = str(report_path)
    if print_cli:
        print_strm_json_cmp(report, verbose=verbose)
    return report


def stream_json_sanity(data_path, ref_dir, output_dir, **kwargs):
    """Generate plain stream JSONs, compare them to references, and return one structured report."""
    from video_to_stream_data import process_video

    data_path = Path(data_path)
    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir)
    if not ref_dir.is_dir():
        raise NotADirectoryError(ref_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_group_tag = kwargs.get('default_group_tag', kwargs.get('group_ann', [0]))

    expected_jsons = _iter_stream_input_json_names(data_path)
    occupied = [name for name in expected_jsons if (output_dir / name).exists()]
    if occupied:
        raise FileExistsError(f'output_dir already contains target JSON files: {", ".join(occupied)}')

    process_kwargs = dict(kwargs)
    for key in ('default_group_tag', 'group_ann', 'zip_output', 'zip', 'output_path', 'file_names'):
        process_kwargs.pop(key, None)
    process_video(data_path,
                  output_path=output_dir,
                  default_grp_tag=default_group_tag,
                  zip_output=False,
                  **process_kwargs)

    report = test_stream_jsons(output_dir, ref_dir, output_dir, file_names=expected_jsons, **kwargs)
    report['data_path'] = str(data_path)
    return report


def print_strm_json_cmp(r, **kwargs):
    """Print one compact stream JSON comparison table from `stream_json_sanity(...)`."""
    rows = list(r.get('files', []))
    verbose = bool(kwargs.get('verbose', False))
    file_count = len(rows)

    if r.get('status') == 'pass':
        print_color(f"stream_json_sanity: pass ({file_count} files)", 'g')
        if verbose:
            for row in rows:
                _print_stream_json_verbose(row)
        return

    print_color(f"stream_json_sanity: fail ({file_count} compared files)", 'r')
    if r.get('missing_refs'):
        print_color(f"missing refs: {', '.join(r['missing_refs'])}", 'r')
    if r.get('extra_refs'):
        print_color(f"extra refs: {', '.join(r['extra_refs'])}", 'r')

    columns = [('file', 'file'), ('status', 'status'),  ('metadata', 'meta'),
               ('ann_intervals', 'ann-int'), ('ann_frames', 'ann-frame'),
               ('det_counts', 'det-cnt'), ('avg_abs', 'avg-abs'), ('max_abs', 'max-abs')]
    widths = {key: len(label) for key, label in columns}
    for row in rows:
        widths['file'] = max(widths['file'], len(str(row['file'])))
        widths['status'] = max(widths['status'], len(str(row['status'])))
        widths['metadata'] = max(widths['metadata'], len(str(row['metadata'])))
        widths['ann_intervals'] = max(widths['ann_intervals'], len(str(row['ann_intervals'])))
        widths['ann_frames'] = max(widths['ann_frames'], len(str(row['ann_frames'])))
        widths['det_counts'] = max(widths['det_counts'], len(str(row['det_counts'])))
        widths['avg_abs'] = max(widths['avg_abs'], len(_fmt_num(float(row['avg_abs']))))
        widths['max_abs'] = max(widths['max_abs'], len(_fmt_num(float(row['max_abs']))))

    header = ' | '.join(f"{label:<{widths[key]}}" for key, label in columns)
    rule = '-+-'.join('-' * widths[key] for key, _ in columns)
    print(header)
    print(rule)
    for row in rows:
        line = ' | '.join(( f"{row['file']:<{widths['file']}}",
                            f"{row['status']:<{widths['status']}}",
                            f"{row['metadata']:<{widths['metadata']}}",
                            f"{row['ann_intervals']:<{widths['ann_intervals']}}",
                            f"{row['ann_frames']:<{widths['ann_frames']}}",
                            f"{row['det_counts']:<{widths['det_counts']}}",
                            f"{_fmt_num(float(row['avg_abs'])):<{widths['avg_abs']}}",
                            f"{_fmt_num(float(row['max_abs'])):<{widths['max_abs']}}", ))

        if row.get('ok', False):
            print(line)
        else:
            print_color(line, 'r')

    if verbose:
        for row in rows:
            _print_stream_json_verbose(row)


def run_sanity_flow(cache_dir, out_dir, ref_dir=None, ds_testing=None, stm_testing=None, **kwargs):
    """ Run one sanity flow and optionally compare it against a reference output tree.
    :param cache_dir:
    mode ='all':` run full cycle testing:  train -> test -> eval
          'no_train': use models in ref_dir to skip training (test -> eval)
          'eval_only': reruns only evaluation from saved raw `*-tst.npz` in ref_dir
          'test_only': run only the testing phase (use models in ref_dir
    """

    def _save_report(target_dir: Path, report_name: str)->Path:
        """ Save one JSON sanity report under the output dir."""
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir/f'{report_name}.json'
        with report_path.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        return report_path

    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    ref_dir = Path(ref_dir) if ref_dir is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = kwargs.get('mode', 'all')
    if mode not in {'all', 'no_train', 'eval_only', 'test_only'}:
        raise ValueError(f'Unsupported sanity mode: {mode}')

    if mode != 'all' and ref_dir is None:
        raise ValueError(f'mode={mode!r} requires ref_dir')

    phases, run_dirs, results = [], [], []
    #* Train phase
    if mode == 'all':
        phases.append('train')
        run_dirs = train_sanity_models(cache_dir, out_dir, ds_testing=ds_testing, stm_testing=stm_testing, **kwargs)
    #* Test  phase
    if mode in {'all', 'no_train', 'test_only'}:
        phases.append('test')
        #* in no_train/'test_only' tests are preformed with existing models the run_dirs
        test_source = run_dirs if mode == 'all' else ref_dir
        run_dirs, results = run_sanity_tests(test_source, cache_dir=cache_dir, out_dir=out_dir,
                                             ds_testing=ds_testing, stm_testing=stm_testing, **kwargs)
    #* Evaluation phase
    if mode in {'all', 'no_train', 'eval_only'}:
        phases.append('eval')
        eval_dir =  ref_dir if mode == 'eval_only' else out_dir
        run_dirs = run_sanity_eval(eval_dir, out_dir=out_dir, raw_results=results or None,
                                   ds_testing=ds_testing, stm_testing=stm_testing, **kwargs)

    compare_requested = ref_dir is not None
    compare_ok = None
    comparison = None
    if compare_requested:
        compare_ok, comparison = assert_outputs(out_dir, ref_dir, mode)

    report = { 'status': 'pass' if (compare_ok is not False) else 'fail',
               'mode': mode,
               'cache_dir': str(cache_dir),
               'out_dir': str(out_dir),
               'ref_dir': (str(ref_dir) if ref_dir is not None else None),
               'phases': phases,
               'compare_mode': (mode if compare_requested else None),
               'comparison': comparison,
               'run_dirs': [str(path) for path in run_dirs],
               }
    if bool(kwargs.get('save_json', True)):
        report['report_path'] = str(_save_report(out_dir, kwargs.get('report_name', _RESULT_REPORT_NAME)))
    if bool(kwargs.get('print_cli', True)):
        _print_cli_report(report)
    return report


def assert_outputs(test_dir, ref_dir, mode='no_train') -> tuple[bool, dict[str, Any]]:
    """ Semantically compare one output tree against a reference tree.
    Usage:
    :param test_dir: dir with the new generated outputs
    :param ref_dir: dir with verified output files
    :param mode: 'all' uses the loose tolerance profile
                 'no_train' uses the strict tolerance profile
                 'eval_only' uses the strict profile and ignores NPZ drift
    :returns   tuple (res:bool, comparison:dict): res = results, comparison stats
    """
    test_dir = Path(test_dir)
    ref_dir = Path(ref_dir)
    mode = str(mode)
    if mode not in {'all', 'no_train', 'eval_only'}:
        print_color(f'assert_outputs: unsupported mode {mode!r}', 'r')
        return False, {'status': 'fail', 'missing_runs': [], 'extra_runs': [], 'runs': []}

    tolerances = _resolve_tolerances(mode)
    comparison = _build_comparison_report(test_dir, ref_dir, mode=mode)
    def _tolerance_lines() -> list[str]:
        lines = [f"  field type  |\t max tolerance ",
                 f"--------------+-----------------",
                 f"JSON  summary :\t{_fmt_num(max(tolerances['max_abs_error'], tolerances['cm_delta']))} ",
                 f"      events  :\t{_fmt_num(tolerances['event_f1']) } ",
                 f"CSV max delta :\t{_fmt_num(tolerances['csv'])}"]
        if mode != 'eval_only':
            lines += [f"NPZ   d_prob  : {_fmt_num(tolerances['npz'])}"]
        return lines

    res = not (comparison['missing_runs'] or comparison['extra_runs'] or
               any(run['files_missing'] or run['files_extra'] or run['semantic_issues']
               for run in comparison['runs']))

    if res:
        print_color(f'\nOutput assertion passed for mode={mode}', 'g')
        return res, comparison

    print(f"\nOutput assertion failed for mode={mode}\ncomparison profile: {tolerances['profile']}")
    for line in _tolerance_lines():
        print(line)

    if comparison['missing_runs']:
        print_color(f"missing runs: {', '.join(comparison['missing_runs'])}", 'r')
    if comparison['extra_runs']:
        print_color(f"extra runs: {', '.join(comparison['extra_runs'])}", 'r')

    for run_report in comparison['runs']:
        _print_issues(run_report, mode)
    return res, comparison

# endregion

#* 1191 ->1161-> 1217-> 1199-> 1188(1,22,2) -> 1166(,22,2)->1155
#* 1390(2,5,3)-> refact-01-1277(1,5,3)

if __name__ == '__main__':
    import shutil
    stream_testing = ['cam-6-11-5_ft25_w30-15.npz',
                      'cam-6-11-8_FRes_Ana_ft25_w30-15.npz',
                      'cam-6-11-8_FRes_Erz_ft25_w30-15.npz']
    ds_testsing    = ['J-All_ft25_w30-15_test.npz']

    #* 1st testing
    # d_tr = "work_dirs/json_models/sanity-testing/test_all-01"
    # d_nt = "work_dirs/json_models/sanity-testing/no_train"
    # d_rf = "work_dirs/json_models/w30-15-um"
    # b_1, _ = assert_outputs(d_tr, d_rf)
    # b_2, _ = assert_outputs(d_nt, d_rf, 'train')
    # print(f"train: {b_1}\n test:{b_2}")

    #* 2nd testing
    cache_path = 'data/cache/w30-15_um'
    rf_dir  = 'work_dirs/json_models/w30-15-um'
    res_dir = 'work_dirs/json_models/sanity-testing/test-only_01'
    md = 'all'# 'no_train'
    if Path(res_dir).is_dir():
        shutil.rmtree(res_dir)
    # run_sanity_flow(cache_path, res_dir, rf_dir,
    #                 ds_testing=ds_testsing, stm_testing=stream_testing, mode=md)
    # res_, info = assert_outputs(tst_dir, rf_dir, 'no_train')
    # print(f"Assert outputs:{res_}\ninfo:{info}")
    
    #* 3rd testing
    d_ts = "/mnt/local-data/Python/Projects/weSmart/data/json_files/tst_conv/try_05"
    d_rf = "/mnt/local-data/Python/Projects/weSmart/data/sanity-testing/json/260611-no_imgsz_g-0"
    d_op = "/mnt/local-data/Python/Projects/weSmart/data/sanity-testing/json"
    test_stream_jsons(d_ts, d_rf, d_op, print_cli=True)
    #1412(2,5,4)
