""" Sanity runners for train/test/eval flows and semantic output comparison.
    Usage:
    - run one sanity flow with `run_sanity_flow(...)`
    - compare one new output tree against a reference with `assert_outputs(...)`
    - keep old/new export-format compatibility in this layer instead of the evaluation modules
"""

import csv, json
import random, re
from pathlib import Path
from typing import Any
import numpy as np
import torch
#* Project imports 
from common.my_local_utils import as_collection, print_color
from evaluation_tools import (analyze_clip_test, analyze_stream_test, analyze_video_test,
                              DEFAULT_EVAL_THRESHOLD)
from precompute_clips import RANDOM_SEED
from project_utils import get_exporting_name, resolve_best_pt_model
from scripts import sum_all_results
from torch_clip_model import run_testing, run_training

DEFAULT_NPZ_TOLERANCES = 1e-6
DEFAULT_CSV_TOLERANCES = 1e-3
DEFAULT_METRIC_TOLERANCES = {
            'accuracy'  : 0.01,
            'precision' : 0.01,
            'recall'    : 0.01,
            'f1'        : 0.01,
            'auc'       : 0.01,
             #'balanced_accuracy': 0.01,
            'TPR'       : 0.01,
            'FPR'       : 0.01,
            'event_precision': 0.02,
            'event_recall': 0.02,
            'event_f1'  : 0.02,
            'false_positive_time': 0.05,
            'miss_time': 0.05,
            'threshold': 1e-8,
            }

_RESULT_REPORT_NAME = 'sanity_report'
_EVAL_PLAN_NAME = '_sanity_eval_plan.json'
_EXPORTED_FILE_PATTERNS = ('*-summary.json',  '*_stream-events.json', '*.npz',
                           'ROC_*.png',  'ROC_*.csv',  '*_timeline.csv', 'timeline_*.csv', 'timeline_*.png',)

_STATUS_RANK = {'pass': 0, 'pass_with_drift': 1, 'fail': 2}
# TODO: consider moving these path-like JSON keys into a shared project-wide constant.
_JSON_PATH_KEYS = {'cache_dir', 'events_info', 'model_path', 'new_run_dir', 'output_dir', 'raw_results',
                   'raw_results_path', 'ref_dir', 'ref_run_dir', 'report_path', 'roc_csv', 'test_cache', 'threshold_dir',
                   'timeline_csv', 'timeline_csvs', 'timeline_plot', 'timeline_plots',}

#* region Config And Small Shared Helpers
def _model_tag_from_run_dir(run_dir: Path) -> str:
    """Return the logical model tag, stripping one timestamp prefix when present."""
    return run_dir.name.split('_', 1)[1] if re.match(r'^\d{6}-\d{4}_.+', run_dir.name) else run_dir.name


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


def _resolve_npz_inputs(inputs, base_dir: Path) -> list[Path]:
    """Resolve NPZ files, dirs, or masks into one ordered unique list."""
    resolved = []
    seen = set()
    for item in as_collection(inputs or []):
        item = Path(item)
        if not item.is_absolute():
            item = base_dir / item
        item_str = str(item)
        if any(ch in item_str for ch in '*?[]'):
            import glob
            matches = [Path(p) for p in glob.glob(item_str)]
        elif item.is_dir():
            matches = sorted(p for p in item.iterdir() if p.is_file() and p.suffix == '.npz')
        elif item.is_file():
            matches = [item]
        else:
            matches = []
        for path in matches:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)
    return resolved


#* Shared runtime flags
def _threshold_dir(threshold:float) -> Path:
    return Path(f'th-{int(round(threshold*100.0))}')


def _set_deterministic(seed: int) -> None:
    """Apply one best-effort deterministic seed setup for Python, NumPy, and torch."""
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


def _infer_eval_threshold(run_dir: Path) -> float:
    """Reuse the saved threshold from prior summaries, else fall back to the default."""
    for summary_path in sorted(run_dir.rglob('*-summary.json')):
        try:
            with summary_path.open('r', encoding='utf-8') as f:
                summary = json.load(f)
            return float(summary.get('analysis_config', {}).get('threshold', DEFAULT_EVAL_THRESHOLD))
        except Exception:
            continue
    return DEFAULT_EVAL_THRESHOLD

# endregion

#* region Run Execution Helpers
def _run_dataset_test(model_path: Path, test_npz: Path, out_dir: Path, *, run_video: bool, **kwargs) -> str:
    """Run one dataset test and save the raw `*-tst.npz` output."""
    raw_tag = get_exporting_name(model_path, test_npz, 'raw', unit='clip')
    res = run_testing(model_path, test_npz, out_dir=out_dir, output_tag=raw_tag)
    return str(res['path'])


def _run_stream_test(model_path: Path, test_npz: Path, out_dir: Path, **kwargs) -> str:
    """Run one stream test and save the raw `*-tst.npz` output."""
    raw_tag = get_exporting_name(model_path, test_npz, 'raw', unit='stream')
    res = run_testing(model_path, test_npz, out_dir=out_dir, output_tag=raw_tag, video_mode=True)
    return str(res['path'])


def train_sanity_models(cache_dir, out_dir, *, ds_testing=None, stm_testing=None, **kwargs):
    """ Train every `*_train.npz` cache in one directory and return the created run dirs."""
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_caches = sorted(cache_dir.glob('*_train.npz'))
    if not train_caches:
        raise FileNotFoundError(f'No *_train.npz caches found in {cache_dir}')

    deterministic = kwargs.get('deterministic', False)
    base_seed = kwargs.get('random_seed', RANDOM_SEED)
    train_kw = {k: kwargs[k] for k in ('lr', 'batch_size', 'hidden_dim', 'save_every',
                                       'max_epochs', 'epochs', 'patience', 'min_delta',
                                       'split_ratio', 'split_seed')
                                        if k in kwargs and kwargs[k] is not None}
    run_dirs = []
    for index, train_cache in enumerate(train_caches):
        if deterministic:
            _set_deterministic(base_seed + index)
        train_tag = train_cache.stem[:-len('_train')] if train_cache.stem.endswith('_train') else train_cache.stem
        run_dir = Path(run_training(train_cache, tag=train_tag, work_dir=out_dir, **train_kw))
        run_dirs.append(run_dir)
    return run_dirs


def run_sanity_tests(models_or_ref_dir, *, cache_dir, out_dir, ds_testing=None, stm_testing=None, **kwargs):
    """ Run tests for existing models or reference run dirs and save one eval plan."""

    cache_dir,out_dir = Path(cache_dir), Path(out_dir)

    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(models_or_ref_dir, (str, Path)) and Path(models_or_ref_dir).is_dir():
        model_refs = _iter_run_dirs(Path(models_or_ref_dir))
    else:
        model_refs = list(as_collection(models_or_ref_dir))
    if not model_refs:
        raise FileNotFoundError('No model refs were provided for sanity testing')

    ds_targets = _resolve_npz_inputs(ds_testing, cache_dir)
    stm_targets = _resolve_npz_inputs(stm_testing, cache_dir)
    run_video = bool(kwargs.get('run_video', False))
    eval_plan = {}
    raw_results = []
    tested_dirs = []

    for mdl in model_refs:
        best_model = resolve_best_pt_model(mdl)
        mdl = Path(mdl)
        src_run_dir = mdl if mdl.is_dir() else best_model.parent
        target_run_dir = src_run_dir if src_run_dir.parent.resolve() == out_dir.resolve() else out_dir / src_run_dir.name
        target_run_dir.mkdir(parents=True, exist_ok=True)
        model_tag = _model_tag_from_run_dir(src_run_dir)
        own_test = cache_dir/f'{model_tag}_test.npz'
        threshold = float(kwargs.get('threshold', _infer_eval_threshold(src_run_dir)))

        dataset_tests = []
        seen = set()
        # Prefer the model's paired own-test cache, then add shared dataset tests once.
        if own_test.is_file():
            dataset_tests.append((own_test, False))
            seen.add(str(own_test.resolve()))
        for path in ds_targets:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            dataset_tests.append((path, run_video))

        for test_npz, use_video in dataset_tests:
            raw_path = _run_dataset_test(best_model, test_npz, target_run_dir,
                                         run_video=bool(use_video), **kwargs)
            raw_results.append(Path(raw_path))
            rel_path = Path(raw_path).relative_to(out_dir).as_posix()
            eval_plan[rel_path] = {'mode': ('video' if use_video else 'clip'), 'threshold': threshold}

        if kwargs.get('run_stream', True):
            for test_npz in stm_targets:
                raw_path = _run_stream_test(best_model, test_npz, target_run_dir, **kwargs)
                raw_results.append(Path(raw_path))
                rel_path = Path(raw_path).relative_to(out_dir).as_posix()
                eval_plan[rel_path] = {'mode': 'stream', 'threshold': threshold}
        tested_dirs.append(target_run_dir)

    plan_path = out_dir / _EVAL_PLAN_NAME
    with plan_path.open('w', encoding='utf-8') as f:
        json.dump(eval_plan, f, indent=2)
    # return {'run_dirs': tested_dirs, 'raw_results': raw_results}
    return tested_dirs, raw_results


def run_sanity_eval(source_dir, *, out_dir, raw_results=None, ds_testing=None, stm_testing=None, **kwargs):
    """Re-run only the evaluation phase from saved raw `*-tst.npz` outputs."""

    def _summary_exists(run_dir: Path, infer_mode: str)-> bool:
        if model_path is None or test_cache is None:
            return False
        summary_name = f"{get_exporting_name(model_path, test_cache, 'summary', unit=infer_mode)}.json"
        return any(path.name == summary_name for path in run_dir.rglob('*.json'))

    def _infer_eval_mode(run_dir: Path) -> str:
        # Old raw NPZ files do not store the eval mode explicitly, so reuse the mode
        # implied by whichever summary file already exists in the source run dir.
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
            model_tag = _model_tag_from_run_dir(path.parent)
            print(f'\n\b=== {model_tag} model Evaluation ===\n= evaluated dir : {target_run_dir}')
            prev_run_dir = path.parent
        # The raw test NPZ already carries the model/cache refs needed to rebuild names.
        with np.load(path, allow_pickle=True) as data:
            model_path = data['model_path'].item() if isinstance(data['model_path'], np.ndarray) else data['model_path']
            test_cache = data['test_cache'].item() if isinstance(data['test_cache'], np.ndarray) else data['test_cache']
        plan_item = eval_plan.get(rel_raw, {})
        mode = plan_item.get('mode') or _infer_eval_mode(path.parent)
        threshold = kwargs.get('threshold', plan_item.get('threshold', _infer_eval_threshold(path.parent)))
        output_name = get_exporting_name(model_path, test_cache, 'summary', unit=mode)
        common = {'out_path': target_run_dir,
                  'threshold': threshold,
                  'threshold_dir': _threshold_dir(threshold),
                  'overwrite': True,
                  'show_roc': kwargs.get('show_roc', False),
                  'roc_csv': kwargs.get('roc_csv', True),
                  'print_policy': kwargs.get('print_policy', 'summary'),
                  'print':  kwargs.get('print_report', False),
                  }
        if mode == 'stream':
             analyze_stream_test(path, output_name=output_name,
                                details_name=f"{get_exporting_name(model_path, test_cache, 'events')}.json",
                                events_json=kwargs.get('events_json', True), plotting=kwargs.get('plotting', 'save'), **common,)
        elif mode == 'video':
            analyze_video_test(path, output_name=output_name, **common)
        elif mode == 'clip':
            analyze_clip_test(path, output_name=output_name, **common)
        else:
            raise ValueError(f'Unrecognized mode: {mode}')
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


def _compare_json_semantics(test_path: Path, ref_path: Path, *, tolerances: dict[str, float]) -> dict[str, Any]:
    """ Compare one JSON semantically, ignoring path-only differences."""
    summary = {'numeric_count': 0, 'other_count': 0}

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
            return fp_err_max, issues_ls

        if isinstance(test_value, (float, int)) and isinstance(ref_value, (float, int)):
            delta = abs(float(test_value) - float(ref_value))
            fp_err_max = max(fp_err_max, delta)
            tol = float(tolerances[path]) if path in tolerances else float(tolerances.get(path.rsplit('.', 1)[-1], 1e-6))
            if delta > tol:
                issues_ls.append(f'{path}: numeric drift {delta:.9g} exceeds tolerance {tol:.9g}')
                summary['numeric_count'] += 1
            return fp_err_max, issues_ls

        if test_value != ref_value:
            issues_ls.append(f'{path}: {test_value!r} != {ref_value!r}')
            summary['other_count'] += 1
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
            'other_count': summary['other_count']}


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


def _cmp_op_file(test_path:Path, ref_path:Path, *, mode: str,
                 csv_atol: float, npz_y_prob_atol: float, json_tolerances: dict[str,float])-> dict[str, Any]:
    """Dispatch one semantic file comparison by suffix."""
    test_path, ref_path = Path(test_path), Path(ref_path)
    suffix = test_path.suffix.lower()[1:]
    if suffix == 'json':
        return _compare_json_semantics(test_path, ref_path, tolerances=json_tolerances)
    elif suffix == 'png':
        return _cmp_png_pix(test_path, ref_path)
    elif suffix == 'csv':
        return _compare_csv_with_tolerance(test_path, ref_path, atol=csv_atol)
    elif suffix == 'npz':
        if mode == 'no_train':
            return _compare_npz_semantics(test_path, ref_path, y_prob_atol=npz_y_prob_atol)
        return {'ok': True, 'max_fp_err': 0.0, 'message': None, 'kind': 'npz'}
    return {'ok': True, 'max_fp_err': 0.0, 'message': None, 'kind': 'other'}


def _compare_run_outputs(test_run_dir: Path, ref_run_dir: Path, *, mode: str, csv_atol: float,
                         y_prob_atol:float, json_tolerances:dict[str, float])-> dict[str, Any]:
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
        run_tag = _model_tag_from_run_dir(run_dir)
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
        result = _cmp_op_file(Path(test_files[f]), Path(ref_files[f]), mode=mode, csv_atol=csv_atol,
                              npz_y_prob_atol=y_prob_atol, json_tolerances=json_tolerances, )

        max_fp_err = max(max_fp_err, float(result.get('max_fp_err', 0.0)))
        kind = result.get('kind', 'other')
        if not result.get('ok', True):
            if kind in mismatch_counts:
                mismatch_counts[kind] += 1
            semantic_issues.append({'file': f,
                                    'kind': kind,
                                    'message': result.get('message'),
                                    'max_fp_err': float(result.get('max_fp_err', 0.0)),
                                    'issue_count': int(result.get('issue_count', 1)),
                                    'numeric_count': int(result.get('numeric_count', 0)),
                                    'other_count': int(result.get('other_count', 0)),
                                    'issue_type': result.get('issue_type', None),
                                    'issue_key': result.get('issue_key', None),})

    return {'status': ('fail' if (missing or extra or semantic_issues) else 'pass'),
            'new_run_dir': str(test_run_dir),
            'ref_run_dir': str(ref_run_dir),
            'files_missing': missing,
            'files_extra': extra,
            'max_fp_err': max_fp_err,
            'mismatches': mismatch_counts,
            'semantic_issues': semantic_issues,}


#* Semantic report formatting
def _build_comparison_report(test_dir, ref_dir, *, mode: str, tolerance_update=None) -> dict[str, Any]:
    """Build one per-run semantic comparison report for two output trees."""
    test_dir = Path(test_dir)
    ref_dir = Path(ref_dir)
    tolerances = dict(DEFAULT_METRIC_TOLERANCES)
    tolerances.update(tolerance_update or {})

    test_runs = {_model_tag_from_run_dir(path): path for path in _iter_run_dirs(test_dir)}
    ref_runs = {_model_tag_from_run_dir(path): path for path in _iter_run_dirs(ref_dir)}
    missing_runs = sorted(set(ref_runs) - set(test_runs))
    extra_runs = sorted(set(test_runs) - set(ref_runs))
    status = 'fail' if (missing_runs or extra_runs) else 'pass'

    run_reports = []
    for run_key in sorted(set(test_runs) & set(ref_runs)):
        run_report = _compare_run_outputs(test_runs[run_key], ref_runs[run_key], mode=mode,
                                          csv_atol=float(DEFAULT_CSV_TOLERANCES),
                                          y_prob_atol=float(DEFAULT_NPZ_TOLERANCES),
                                          json_tolerances=tolerances, )
        run_report['run'] = run_key
        if _STATUS_RANK[run_report['status']] > _STATUS_RANK[status]:
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

# endregion

#* region Public Sanity API
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

    if mode in {'all', 'no_train', 'eval_only'}:
        try:
            sum_all_results(out_dir, save_json=kwargs.get('save_summery', True))
        except Exception as exc:
            print_color(f'[WARN] sum_all_results failed for {out_dir}: {type(exc).__name__}: {exc}', 'y')

    compare_mode = 'train' if mode == 'all' else 'no_train'
    compare_requested = ref_dir is not None
    compare_ok = None
    comparison = None
    if compare_requested:
        compare_ok, comparison = assert_outputs(out_dir, ref_dir, compare_mode)

    report = { 'status': 'pass' if (compare_ok is not False) else 'fail',
               'mode': mode,
               'cache_dir': str(cache_dir),
               'out_dir': str(out_dir),
               'ref_dir': (str(ref_dir) if ref_dir is not None else None),
               'phases': phases,
               'compare_mode': (compare_mode if compare_requested else None),
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
    :param mode: 'no_train' allows old/new export-name compatibility and tiny numeric drift
                 'train'  applies the training tolerances in `DEFAULT_METRIC_TOLERANCES`
    :returns   tuple (res:bool, comparison:dict): res = results, comparison stats
    """
    test_dir = Path(test_dir)
    ref_dir = Path(ref_dir)
    mode = str(mode)
    if mode not in {'no_train', 'train'}:
        print_color(f"assert_outputs: unsupported mode {mode!r}", 'r')
        return False, {'status': 'fail', 'missing_runs': [], 'extra_runs': [], 'runs': []}

    comparison = _build_comparison_report(test_dir, ref_dir, mode=mode, tolerance_update=DEFAULT_METRIC_TOLERANCES)
    def _fmt_num(value: float) -> str:
        value = float(value)
        abs_val = abs(value)
        if value == 0:
            return '0'
        if abs_val < 1e-3 or abs_val >= 1e4:
            return f'{value:.1e}'
        if abs_val >= 1:
            return str(int(value)) if value.is_integer() else f'{value:.4g}'
        text = f'{value:.4f}'.rstrip('0').rstrip('.')
        return text or '0'

    def _tolerance_lines() -> list[str]:
        json_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'TPR', 'FPR',
                     'event_precision', 'event_recall', 'event_f1',
                     'false_positive_time', 'miss_time', 'threshold']
        json_parts = [f'{key}={_fmt_num(DEFAULT_METRIC_TOLERANCES[key])}'
                      for key in json_keys if key in DEFAULT_METRIC_TOLERANCES]
        return [f"JSON tol: {', '.join(json_parts)}",
                f'CSV tol : {_fmt_num(DEFAULT_CSV_TOLERANCES)}',
                f'NPZ tol : y_prob={_fmt_num(DEFAULT_NPZ_TOLERANCES)}']

    def _issue_sort_key(issue: dict[str, Any]) -> tuple[int, str]:
        kind_order = {'npz': 0, 'csv': 1, 'json': 2, 'png': 3}
        return kind_order.get(issue.get('kind', 'other'), 99), issue.get('file', '')

    def _summarize_issue(issue: dict[str, Any]) -> str:
        kind = issue.get('kind', 'other')
        file_name = issue.get('file', '')
        fp_err = issue.get('max_fp_err', 0.0)

        if kind == 'json':
            drift_count = issue.get('numeric_count', 0)
            other_count = issue.get('other_count', 0)
            parts = []
            if drift_count:
                parts.append(f'drifts={drift_count}')
            if other_count:
                parts.append(f'other={other_count}')
            if not parts:
                parts.append(f'issues={issue.get("issue_count", 1)}')
            return f'{file_name}: json {", ".join(parts)}, max={_fmt_num(fp_err)}'

        if kind == 'csv':
            issue_type = issue.get('issue_type', None)
            if issue_type == 'numeric':
                return f'{file_name}: csv max drift = {_fmt_num(fp_err)}'
            return f'{file_name}: csv {issue.get("message", "mismatch")}'

        if kind == 'npz':
            issue_key = issue.get('issue_key', None)
            if issue_key == 'y_prob':
                return f'{file_name}: npz y_prob max drift={_fmt_num(fp_err)}'
            if issue_key is not None:
                return f'{file_name}: npz mismatch key={issue_key}'
            return f'{file_name}: npz mismatch'

        if kind == 'png':
            issue_type = issue.get('issue_type', None)
            if issue_type == 'pixels':
                return f'{file_name}: png pixel mismatch'
            if issue_type == 'size':
                return f'{file_name}: png size mismatch'
            return f'{file_name}: png mismatch'

        return f'{file_name}: {issue.get("message", "mismatch")}'

    res = not (comparison['missing_runs'] or comparison['extra_runs'] or
               any(run['files_missing'] or run['files_extra'] or run['semantic_issues']
               for run in comparison['runs']))

    if res:
        print_color(f'\nOutput assertion passed for mode={mode}', 'g')
        return res, comparison

    print_color(f'\nOutput assertion failed for mode={mode}', 'r')
    for line in _tolerance_lines():
        print_color(line, 'r')

    if comparison['missing_runs']:
        print_color(f"missing runs: {', '.join(comparison['missing_runs'])}", 'r')
    if comparison['extra_runs']:
        print_color(f"extra runs: {', '.join(comparison['extra_runs'])}", 'r')

    for run_report in comparison['runs']:
        if not (run_report['files_missing'] or run_report['files_extra'] or run_report['semantic_issues']):
            continue
        print_color(f'\n{run_report["run"]}:', 'r')
        if run_report['files_missing']:
            print_color(f'  missing files: {len(run_report["files_missing"])}', 'r')
        if run_report['files_extra']:
            print_color(f'  extra files  : {len(run_report["files_extra"])}', 'r')
        for issue in sorted(run_report['semantic_issues'], key=_issue_sort_key):
            print_color(f'  {_summarize_issue(issue)}', 'r')
    return res, comparison

# endregion

#*647->874/904(,1,1)915 -> 780(,9,3)-> 780(,9,1) -> 770(,14,1) self clr-> 734(,15,1) -> 672
#* -> export-pt-res 693 -> commenting -> 749(,2,1)/777

if __name__ == '__main__':
    import shutil
    stream_testing = ['cam-6-11-5_ft25_w30-15.npz',
                      'cam-6-11-8_FRes_Ana_ft25_w30-15.npz',
                      'cam-6-11-8_FRes_Erz_ft25_w30-15.npz']
    ds_testsing    = ['J-All_ft25_w30-15_test.npz']

    # 1st testing
    # d_tr = "work_dirs/json_models/sanity-testing/test_all-01"
    # d_nt = "work_dirs/json_models/sanity-testing/no_train"
    # d_rf = "work_dirs/json_models/w30-15-um"
    # b_1, _ = assert_outputs(d_tr, d_rf)
    # b_2, _ = assert_outputs(d_nt, d_rf, 'train')
    # print(f"train: {b_1}\n test:{b_2}")

    tst_dir = 'work_dirs/json_models/sanity-testing/test-only_01'
    rf_dir  = 'work_dirs/json_models/w30-15-um'

    if Path(tst_dir).is_dir():
        shutil.rmtree(tst_dir)
    run_sanity_flow('data/cache/w30-15', tst_dir, rf_dir,
                    ds_testing=ds_testsing, stm_testing=stream_testing, mode='no_train')
    # res_, info = assert_outputs(tst_dir, rf_dir, 'no_train')
    # print(f"Assert outputs:{res_}\n")
    # print(f"inof:{info}")
