""" Project batch helpers for cache building, model train/test runs,
    result aggregation, and small local utility flows.
    Usage:
    - build one cache config from JSON directories with `build_cache(...)`
    - build multi-mode cache batches with `build_cache_batch(...)`
    - train model batches with `train_models(...)` or `train_test_study(...)`
    - rerun tests for existing models with `test_models(...)`
    - collect summary tables with `sum_all_results(...)`
    - run paired stream-JSON conversions with `run_stream_json_dual(...)`
"""

import json, pickle, glob, re, time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
#* Imports from local project
from common.my_local_utils import as_collection, cli_warning, get_unique_name, list_file_list, print_color
from cache_builder import (build_cache, build_cache_batch, build_cache_pair,
                           build_caches, draw_ttp, resolve_lists)
from precompute_clips import (build_cache_from_json, extract_stream_features, merge_cache_npz, get_split_pair,
                              WINDOW_SEC, STRIDE_SEC)
from tms_trainer import run_training, run_testing
from torch_clip_model import run_stream_testing
from evaluation_core import analyze_clip_test, analyze_video_test, support_pair, DEFAULT_EVAL_THRESHOLD
from analysis_api import evaluate_raw_test, print_eval_group
from json_stream_utils import load_stream_inputs
from json_utils import (STREAM_FILE_TYPES, list_json_sources, load_json_raw, resolve_json_files,
                        resolve_json_source)
from motion_feature_schema import load_cache_contract_compact, resolve_stream_schema
from project_utils import (get_exporting_name, resolve_best_pt_model, strip_split_suffix,
                           strip_timestamp_prefix)
from stream_utils import filter_yolo, resample_fps


#* general configuration
RWF_DIR  = Path("data/json_files/RWF-2000/ds")
RLVS_DIR = Path("data/json_files/RLVS/ds")

MAIN_WORK_DIR = Path("work_dirs/json_models")
MAIN_CACHE_DIR = Path("data/cache")
# STUDY_CACHE_DIR  = MAIN_CACHE_DIR/"win-study"

DATASETS = [('RWF', RWF_DIR), ('RLVS', RLVS_DIR)]
JOINT_DS =  'J-RWL'
RESULT_NAME = 'all_results'


def resolve_npz_inputs(inputs, base_dir:Path|None = None) -> list[Path]:
    """ Resolve files, dirs, or masks into one ordered unique NPZ list."""
    resolved = []
    seen = set()
    for item in as_collection(inputs or []):
        item = Path(item)
        if not item.is_absolute() and item.parent == Path('.'):
            if base_dir is None:
                print_color(f"[WARN] NPZ name requires explicit base_dir: {item}", 'o')
                continue
            item = base_dir/item
        item_str = str(item)
        if any(ch in item_str for ch in '*?[]'):
            matches = [Path(p) for p in glob.glob(item_str)]
        elif item.is_dir():
            matches = sorted(p for p in item.iterdir() if p.is_file() and p.suffix == '.npz')
        elif item.is_file():
            matches = [item]
        else:
            matches = []

        if not matches:
            print_color(f"[WARN] No NPZ files matched: {item}", 'o')
            continue

        for path in matches:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)
    return resolved


def infer_eval_threshold(run_dir: Path, default=DEFAULT_EVAL_THRESHOLD) -> float:
    """ Reuse the saved threshold from prior summaries, else fall back to default."""
    summary_paths = []
    for pattern in ('*-summary.json', '*_clip-sum.json', '*_reports.json'):
        summary_paths.extend(Path(run_dir).rglob(pattern))
    for summary_path in sorted(summary_paths):
        try:
            with summary_path.open('r', encoding='utf-8') as f:
                summary = json.load(f)
            return float(summary.get('analysis_config', {}).get('threshold', default))
        except Exception:
            continue
    return float(default)


def sum_all_results(res_dir: str | Path, **kwargs):  # 107 -250
    """ Collect all summary/report JSON files under one work dir into one flat results table.
    :param res_dir: at a model/run root containing summary/report JSON files
    optional sort, save_json, and print_cli control output formatting and export
    """

    def _parse_ds_tag(tag: str) -> tuple[str, str, str]:
        """Extract dataset short name, window, and stride from a cache/model tag."""

        def _fmt_num(value: float) -> str:
            return f"{value:g}"

        m = re.match(r"^(?P<ds>.+?)_[0-9]+ft_(?P<w>[0-9o]+)w-(?P<s>[0-9o]+)$", tag)
        if m:
            return m.group("ds"), m.group("w").replace("o", "."), m.group("s").replace("o", ".")

        m = re.match(r"^(?P<ds>.+?)_(?P<ft>[0-9]+ft)$", tag)
        if m:
            return m.group("ds"), _fmt_num(WINDOW_SEC), _fmt_num(STRIDE_SEC)

        return tag, "", ""

    def _model_disp(model_path: str) -> tuple[str, str]:
        """Return compact model label and best-epoch string for printing/sorting."""
        mdl_path = Path(model_path)
        model_name = strip_timestamp_prefix(mdl_path.parent.name)
        best_epoch = mdl_path.stem.split(".")[-1] if "." in mdl_path.stem else ""
        return model_name, best_epoch

    def _pool_short(pool_mode) -> str:
        return {'mean_max': 'mm', 'mean_std_max': 'msm'}.get(str(pool_mode), str(pool_mode))

    def _fmt_meta(value):
        if value in (None, ''):
            return 'N/A'
        if isinstance(value, float):
            return f"{value:g}"
        return value

    def _read_json(path: Path) -> dict:
        try:
            with path.open("r") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _cache_contract(cache_path: Path) -> dict:
        try:
            contract, _ = load_cache_contract_compact(cache_path)
            return contract
        except Exception:
            return {}

    def _cache_meta(cache_path: Path) -> tuple[dict, dict]:
        contract = _cache_contract(cache_path)
        return dict(contract.get('feature_schema', {}) or {}), dict(contract.get('temporal_schema', {}) or {})

    def _model_meta(run_dir: Path) -> tuple[dict, dict]:
        cfg = _read_json(run_dir/'config.json')
        feature_schema = dict(cfg.get('feature_schema', {}) or {})
        temporal_schema = {}
        train_caches = cfg.get('train_caches', [])
        if train_caches and isinstance(train_caches[0], dict):
            temporal_schema = dict(train_caches[0].get('temporal_schema', {}) or {})
        if not temporal_schema:
            temporal_profile = cfg.get('temporal_profile', {}) or {}
            temporal_schema = {'window': temporal_profile.get('target_window'),
                               'stride': temporal_profile.get('target_stride')}
        return feature_schema, temporal_schema

    def _drop_na_columns(rows: list[dict]) -> list[dict]:
        if not rows:
            return rows
        cols = list(rows[0].keys())
        keep = [col for col in cols if any(row.get(col) not in (None, '', 'N/A') for row in rows)]
        return [{col: row.get(col, 'N/A') for col in keep} for row in rows]

    def _sort_flags() -> list[tuple[str, bool]]:
        """Normalize sort flags into `(flag, reverse)` pairs."""
        sort_flags = kwargs.get('sort', 'model')
        if isinstance(sort_flags, str):
            sort_flags = [sort_flags]
        else:
            sort_flags = list(sort_flags)
        if not sort_flags:
            sort_flags = ['model']
        if len(sort_flags) > 3:
            raise ValueError('sort accepts up to 3 flags')

        out = []
        for flag in sort_flags:
            reverse = flag.endswith('-R')
            out.append((flag[:-2] if reverse else flag, reverse))
        return out

    def _sort_key(row: dict, flag: str) -> tuple:
        """ Build a sort key for one requested sort mode."""

        def _num_or_inf(val):
            return float(val) if val not in (None, '', 'N/A') else float('inf')

        if flag == 'model':
            return (_model_disp(row['model'])[0],)
        elif flag == 'win-str':
            return _num_or_inf(row.get('window')), _num_or_inf(row.get('stride'))
        elif flag == 'pool':
            return (row.get('pool', 'N/A'),)
        elif flag == 'fps_ref':
            return (_num_or_inf(row.get('fps_ref')),)
        elif flag == 'trn-tst':
            return row.get('train ds', ''), row.get('test ds', '')
        elif flag == 'auc':
            auc = row.get('AUC')
            return 1 if auc is None else 0, -(auc if auc is not None else 0.0)
        elif flag in {'clp-vid', 'clip-vid', 'vid-clp'}:
            return ({'clip': 0, 'video': 1}.get(row.get('unit'), 99),)
        else:
            raise ValueError(f"Unsupported sort mode: {flag}")

    def _sort_table(rows: list[dict]):
        """Apply stable sorting so each flag can define its own direction."""
        for flag, reverse in reversed(_sort_flags()):
            rows.sort(key=lambda row, f=flag: _sort_key(row, f), reverse=reverse)

    def _row_from_summary(summary_path: Path) -> dict:
        """Convert one summary json file into one flat table row."""
        with summary_path.open("r") as fh:
            summary = json.load(fh)
        if 'streams' in summary and 'testing_set' not in summary:
            return {}

        # Summary files come from clip/video/stream paths, so the table normalizes them
        # onto one shared row schema before sorting and printing.
        testing_set = summary.get('testing_set', {})
        test_cache = Path(testing_set.get('test_cache', summary.get('test_cache', '')))
        model_path = Path(summary.get('model', summary.get('model_path', '')))
        model_schema, model_temporal = _model_meta(model_path.parent)
        test_schema, test_temporal = _cache_meta(test_cache)
        train_tag = strip_timestamp_prefix(model_path.parent.name)
        test_tag = test_cache.stem[:-5] if test_cache.stem.endswith('_test') else test_cache.stem
        train_ds, window, stride = _parse_ds_tag(train_tag)
        test_ds, _, _ = _parse_ds_tag(test_tag)
        window = _fmt_meta(model_temporal.get('window', test_temporal.get('window', window)))
        stride = _fmt_meta(model_temporal.get('stride', test_temporal.get('stride', stride)))
        pool = _fmt_meta(model_schema.get('pool_mode', test_schema.get('pool_mode')))
        if pool != 'N/A':
            pool = _pool_short(pool)
        fps_ref = _fmt_meta(model_schema.get('motion_fps_ref', test_schema.get('motion_fps_ref')))
        feat_dim = _fmt_meta(model_schema.get('feature_dim', test_schema.get('feature_dim')))
        unit = summary.get('analysis_mode', '')
        analysis_cfg = summary.get('analysis_config', {})
        threshold = analysis_cfg.get('threshold', summary.get('threshold', None))

        if unit.startswith('video'):
            samples = testing_set.get('videos_num', None)
            support = support_pair(testing_set.get('videos_support', None))
        else:
            samples = testing_set.get('clips_num', summary.get('num_samples', None))
            support = support_pair(testing_set.get('clips_support', summary.get('support', None)))
        support_str = f'{support[0]}/{support[1]}' if support is not None else 'N/A'

        cm = summary.get('confusion_matrix', summary.get('cm_clips', [[None, None], [None, None]]))
        auc = summary.get('ROC AUC', summary.get('roc_auc', None))
        return {'model': str(model_path), 'cache': test_cache.stem,
                'train ds': train_tag or train_ds,
                'test ds': test_tag or test_ds,
                'unit': unit, 'samples': samples, 'support': support_str,
                'window': window, 'stride': stride, 'pool': pool,
                'fps_ref': fps_ref, 'feat_dim': feat_dim, 'threshold': threshold,
                'FF': cm[0][0], 'FT': cm[0][1],
                'TF': cm[1][0], 'TT': cm[1][1],
                'Acc': summary.get('accuracy', None),
                'Rec': summary.get('recall', None),
                'FPR': summary.get('FPR', None),
                'AUC': auc,
                }

    res_dir = Path(res_dir)
    if not res_dir.is_dir():
        raise NotADirectoryError(res_dir)

    summary_paths = []
    for pattern in ('*-summary.json', '*_clip-sum.json', '*_reports.json'):
        summary_paths.extend(res_dir.rglob(pattern))
    summary_paths = sorted(summary_paths)
    if not summary_paths:
        raise FileNotFoundError(f"No summary/report JSON files found in {res_dir}")

    table = [row for row in (_row_from_summary(p) for p in summary_paths) if row]
    _sort_table(table)
    table = _drop_na_columns(table)

    output_path = res_dir / kwargs.get('op_name', RESULT_NAME)
    with (output_path.with_suffix('.pkl')).open('wb') as f:
        pickle.dump(table, f)

    if kwargs.get('save_json', False):
        with (output_path.with_suffix('.json')).open('w') as f:
            json.dump(table, f, indent=2)

    if kwargs.get('print_cli', True):
        print_summary_results(table, mode=kwargs.get('print_mode', 'short'))

    return table


def print_summary_results(rows: list[dict], mode='short', **kwargs):
    """Print a summary-results table in `short` or `full` mode."""

    def _clip_text(text, limit=24) -> str:
        text = str(text)
        return text if len(text) <= limit else text[:limit - 1] + '...'

    def _model_disp(model_path: str) -> tuple[str, str]:
        mdl_path = Path(model_path)
        model_name = strip_timestamp_prefix(mdl_path.parent.name)
        best_epoch = mdl_path.stem.split(".")[-1] if "." in mdl_path.stem else ""
        return model_name, best_epoch

    if mode not in {'short', 'full'}:
        raise ValueError(f"Unsupported print mode: {mode}")
    if not rows:
        print('\n=== Summary Results ===')
        return

    if mode == 'full':
        cols = ['model', 'BE', 'train ds', 'test ds', 'window', 'stride', 'pool', 'fps_ref', 'feat_dim',
                'threshold', 'unit', 'samples', 'support', 'FF', 'FT', 'TF', 'TT', 'Acc', 'Rec', 'FPR', 'AUC']
    else:
        cols = ['model', 'BE', 'window', 'stride', 'pool', 'fps_ref', 'feat_dim',
                'threshold', 'unit', 'samples', 'Acc', 'Rec', 'FPR', 'AUC']

    cols = [col for col in cols if col == 'BE' or any(col in row for row in rows)]
    header_labels = {c: c if c.isupper() else c.title() for c in cols}
    header_labels.update({'train ds': 'Train ds', 'test ds': 'Test ds',
                          'fps_ref': 'FPS ref', 'feat_dim': 'Feat dim'})

    display_rows = []
    for row in rows:
        disp = row.copy()
        disp['model'], disp['BE'] = _model_disp(disp['model'])
        if 'train ds' in disp:
            disp['train ds'] = _clip_text(disp['train ds'])
        if 'test ds' in disp:
            disp['test ds'] = _clip_text(disp['test ds'])
        for key in ('Acc', 'Rec', 'FPR', 'AUC'):
            if key in disp and isinstance(disp[key], float):
                disp[key] = f'{disp[key]:.4f}'
        if isinstance(disp.get('threshold'), float):
            disp['threshold'] = f"{disp['threshold']:.2f}"
        elif disp.get('threshold') in (None, ''):
            disp['threshold'] = 'N/A'
        display_rows.append(disp)

    widths = {c: len(header_labels[c]) for c in cols}
    for row in display_rows:
        for col in cols:
            widths[col] = max(widths[col], len(str(row.get(col, ''))))

    print('\n=== Summary Results ===')
    print(' | '.join(f'{header_labels[col]:<{widths[col]}}' for col in cols))
    print('-+-'.join('-' * widths[col] for col in cols))
    for row in display_rows:
        line = []
        for col in cols:
            val = str(row.get(col, ''))
            if col in {'window', 'stride', 'threshold', 'BE', 'fps_ref', 'feat_dim'}:
                line.append(f'{val:^{widths[col]}}')
            elif col == 'samples':
                line.append(f'{val:>{widths[col]}}')
            else:
                line.append(f'{val:<{widths[col]}}')
        print(' | '.join(line))


def train_models(cache_dir, main_op_dir, ds_tests=None, stm_tests=None, **kwargs):
    """Train every `*_train.npz` cache in a directory, optionally then test the models."""

    kwargs = dict(kwargs)
    run_tests = kwargs.pop('run_tests', False)
    summary = kwargs.pop('summary', True)
    cache_dir = Path(cache_dir)
    main_op_dir = Path(main_op_dir)
    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    main_op_dir.mkdir(parents=True, exist_ok=True)

    train_caches = sorted(cache_dir.glob("*_train.npz"))
    if not train_caches:
        print(f"[WARN] No *_train.npz caches found in {cache_dir}")
        return []

    built_runs = []
    for train_cache in train_caches:
        train_tag = strip_split_suffix(train_cache.stem)
        try:
            run_dir = Path(run_training(train_cache, tag=train_tag, work_dir=main_op_dir, **kwargs))
        except Exception as exc:
            print(f"[WARN] Training failed for {train_cache.name}: {type(exc).__name__}: {exc}")
            continue

        built_runs.append(run_dir)

    if run_tests and built_runs:
        kwargs.update({'npz_dir': cache_dir, 'test_pair': True, 'summary': main_op_dir if summary else False})
        test_models(built_runs, ds_tests=ds_tests, stm_tests=stm_tests, **kwargs)

    return built_runs

def test_models(models, ds_tests=None, stm_tests=None, **kwargs):
    """ Run tests for existing trained models without retraining them. #417-362-270
    :param models: single or list of model dir/ checkpoint path
    :param ds_tests: NPZ file, directory, mask, or list for dataset evaluations
    :param stm_tests: Stream JSON/ZIP path, directory, stream dict, or homogeneous list
    :param kwargs['stream_schema']: optional nested or flat feature/temporal contract
    :param kwargs['threshold']: one threshold or a list/tuple/set of values between 0 and 1
    :param kwargs['resample_fps']: optional lower FPS used to resample streams before testing
    :param kwargs['fps_mode']: `standard` keeps streams that cannot be resampled;
                              `force`/`require` skips them
    :param kwargs['yolo_th']: optional higher YOLO confidence threshold applied before testing
    :param kwargs['summary']: if summary==True aggregates summaries with sum_all_results(...)
                              summary=<path> stores the aggregate summary
    """

    def config_value(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): config_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [config_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    rerun_kwargs = {key: value for key, value in kwargs.items() if key != 'out_dir'}

    npz_dir = kwargs.pop('npz_dir', None)
    out_dir = kwargs.pop('out_dir', None)
    evaluate = kwargs.pop('evaluate', True)
    print_reports = kwargs.pop('print_reports', None)
    print_report_old = kwargs.pop('print_report', None)
    infer_threshold = kwargs.pop('infer_threshold', False)
    thresholds = kwargs.pop('threshold', None)
    summary = kwargs.pop('summary', False)
    test_pair = kwargs.pop('test_pair', False)
    ds_eval = kwargs.pop('ds_eval_mode', 'clip')
    stream_schema = kwargs.pop('stream_schema', None)
    fps_rsmp = kwargs.pop('resample_fps', None)
    fps_mode = kwargs.pop('fps_mode', 'standard')
    yolo_th = kwargs.pop('yolo_th', None)

    npz_dir = Path(npz_dir) if npz_dir is not None else None
    if yolo_th is not None:
        yolo_th = float(yolo_th)
        if not 0 < yolo_th <= 1:
            raise ValueError('yolo_th must be greater than 0 and no greater than 1')
    if fps_rsmp is not None:
        fps_mode = str(fps_mode).strip().lower()
        if fps_mode == 'require':
            fps_mode = 'force'
        if fps_mode not in {'standard', 'force'}:
            raise ValueError("fps_mode must be 'standard', 'force', or 'require'")
    if print_reports is None:
        print_reports = 'eval' if print_report_old is not False else 'none'
    print_reports = str(print_reports).strip().lower()
    if print_reports == 'evaluation':
        print_reports = 'eval'
    if print_reports not in {'eval', 'stream', 'all', 'none'}:
        raise ValueError("print_reports must be 'eval', 'evaluation', 'stream', 'all', or 'none'")
    print_eval_report = print_reports in {'eval', 'all'}
    print_stream_report = print_reports in {'stream', 'all'}

    def _prepare_stream(name, data):
        details = []
        if fps_rsmp is not None:
            frames_before = data.get('frames')
            try:
                data = resample_fps(data, fps_rsmp)
            except ValueError as exc:
                if fps_mode == 'force':
                    raise
                details.append(f"FPS resample {fps_rsmp:g} skipped ({exc}); using original")
            else:
                if data.get('frames') is not frames_before:
                    sampling = data.get('sampling rate', data.get('sampling_rate', {}))
                    original_sampling = data['original_sampling_rate']
                    counts = data['frame_count']
                    source_fps = original_sampling.get('measured', original_sampling.get('effective'))
                    details.append(f"FPS {source_fps:.3f} -> {sampling['effective']:.3f} "
                                   f"(resample {sampling['target']:g}); frames {counts['original']} -> {counts['current']}")

        if yolo_th is not None:
            data = filter_yolo(data, yolo_th)
            detector = data.get('detector')
            original_yolo = detector.get('threshold') if isinstance(detector, dict) else None
            source_th = 'unknown' if original_yolo is None else f'{original_yolo:g}'
            counts = data['detection_count']
            filtered = [int(key.rsplit('_', 1)[1]) for key in counts if key.startswith('filtered_')]
            filter_idx = max(filtered)
            previous = counts.get(f'filtered_{filter_idx - 1:02d}', counts['original'])
            details.append(f"YOLO {source_th} -> {data['detection_threshold']:g}; "
                           f"detections removed {previous - counts['current']}")
        if details:
            print(f"Stream preparation: {name} | {' | '.join(details)}")
        return data

    def _prepare_streams(loaded, load_failures):
        prepared, failures = [], list(load_failures)
        for failure in failures:
            print_color(f"[WARN] Skipping stream {failure['stream']}: {failure['error']}", 'o')
        for name, data in loaded:
            try:
                prepared.append((name, _prepare_stream(name, data)))
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                print_color(f'[WARN] Skipping stream {name}: {error}', 'o')
                failures.append({'stream': name, 'reason': 'bad data', 'error': error})
        return prepared, failures

    def _model_refs(refs) -> list[Path]:
        out = []
        for ref in as_collection(refs):
            ref = Path(ref)
            if ref.is_dir() and not any(ref.glob("best_model.*.pt")) and not (ref/'model.pt').is_file():
                children = [p for p in sorted(ref.iterdir()) if p.is_dir()]
                usable = [p for p in children if any(p.glob("best_model.*.pt")) or (p/'model.pt').is_file()
                          or any(p.glob("checkpoint_ep-*.pt"))]
                out.extend(usable if usable else [ref])
            else:
                out.append(ref)
        return out

    def _run_raw_test(model_path:Path, tst_npz:Path, out_dir:Path, mode:str, thres:list[float]):
        raw_tag = get_exporting_name(model_path, tst_npz, 'raw', unit=mode)
        run_kwargs = {'video_mode': True}
        if kwargs.get('batch_size') is not None:
            run_kwargs['batch_size'] = kwargs['batch_size']
        res = run_testing(model_path, tst_npz, out_dir=out_dir, output_tag=raw_tag, **run_kwargs)
        if evaluate:
            output_name = get_exporting_name(model_path, tst_npz, 'summary', unit=mode)
            reports = []
            for th in thres:
                eval_kwargs = dict(kwargs)
                eval_kwargs['print_report'] = False
                eval_kwargs['print_policy'] = 'none'
                reports.append(evaluate_raw_test(res['path'], mode, out_dir, th, **eval_kwargs))
            if print_eval_report:
                print_eval_group(reports, output_name)

    def _run_stream_test(model_path:Path, stream_inputs, source_name:str, schema, out_dir:Path,
                         thres:list[float], prep_failures):
        ftr_schema, tmp_schema = schema
        state = torch.load(model_path, map_location='cpu')
        input_dim = int(state['net.0.weight'].shape[1])
        if input_dim != int(ftr_schema['feature_dim']):
            raise ValueError(
                f"feature_dim mismatch: schema={ftr_schema['feature_dim']}, model={input_dim}")
        feature_parts = []
        build_failures = list(prep_failures)
        for name, data in stream_inputs:
            try:
                part = extract_stream_features([(name, data)], ftr_schema, tmp_schema)
                if len(part[1]):
                    feature_parts.append(part)
                else:
                    build_failures.append({'stream': name, 'reason': 'short duration', 'error': None})
            except Exception as exc:
                build_failures.append({'stream': name,
                                       'reason': 'other errors',
                                       'error': f"{type(exc).__name__}: {exc}"})
        if not feature_parts:
            raise ValueError("no valid stream windows were produced")
        X = np.concatenate([part[0] for part in feature_parts])
        y = np.concatenate([part[1] for part in feature_parts])
        meta = np.concatenate([part[2] for part in feature_parts])
        if X.shape[1] != int(ftr_schema['feature_dim']):
            raise ValueError(
                f"extracted feature width mismatch: schema={ftr_schema['feature_dim']}, actual={X.shape[1]}")
        raw_tag = get_exporting_name(model_path, Path(source_name), 'raw', unit='stream')
        run_kwargs = {'out_dir': out_dir, 'output_tag': raw_tag}
        if kwargs.get('batch_size') is not None:
            run_kwargs['batch_size'] = kwargs['batch_size']
        res = run_stream_testing(model_path, X, y, meta, source_name, **run_kwargs)
        if evaluate:
            evaluate_raw_test(res['path'], 'stream', out_dir, thres,
                              build_failures=build_failures,
                              print_report=print_stream_report,
                              **kwargs)

    def _find_test_pair() -> Path|None: #23
        config_path = run_dir/'config.json'
        if not config_path.is_file():
            print_color(f"[WARN] No config file in : {run_dir}", 'o')
            return None
        try:
            with config_path.open('r', encoding='utf-8') as f:
                cfg = json.load(f)
            pair = Path(get_split_pair(cfg['train_cache']))
        except Exception as exc:
            print_color(f"[WARN] Failed extracting paired test {config_path}: {type(exc).__name__}: {exc}", 'o')
            return None
        if not pair.is_file():
            print_color(f"[WARN] Extracted paired test cache was not found for {run_dir.name}: {pair}", 'o')
            return None
        return pair

    try:
        loaded_streams, source_name, load_failures = load_stream_inputs(stm_tests)
        streams, stream_failures = _prepare_streams(loaded_streams, load_failures)
    except Exception as exc:
        print_color(f"[WARN] Stream inputs skipped: {type(exc).__name__}: {exc}", 'o')
        streams, source_name, stream_failures = [], 'streams', []

    tested = []
    for ref_mdl in _model_refs(models):
        try:
            b_mdl = resolve_best_pt_model(ref_mdl)
            run_dir = ref_mdl if ref_mdl.is_dir() else b_mdl.parent
        except Exception as exc:
            print(f"[WARN] Bad model ref {ref_mdl}: {type(exc).__name__}: {exc}")
            continue

        target_dir = Path(out_dir)/run_dir.name if out_dir is not None else run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            thres = thresholds
            if thres is None:
                thres = infer_eval_threshold(run_dir) if infer_threshold else DEFAULT_EVAL_THRESHOLD
            thres = [float(value) for value in as_collection(thres)]
            if any(not 0.0 < value < 1.0 for value in thres):
                raise ValueError(f"Thresholds must be between 0 and 1: {thres}")
        except Exception as exc:
            print_color(f"[WARN] Tests skipped for {run_dir.name}: {type(exc).__name__}: {exc}", 'o')
            continue

        ds_npz = []
        if test_pair:
            tst_pair = _find_test_pair()
            if tst_pair is not None:
                ds_npz.append(tst_pair)
        # if test_pair is not None and not test_pair.is_file():
        #     print_color(f"[WARN] Missing paired test cache for {run_dir.name}: {test_pair}", 'o')
        for tst_npz in resolve_npz_inputs(ds_tests, npz_dir):
            if tst_npz not in ds_npz:
                ds_npz.append(tst_npz)

        # if test_pair.is_file():
        #     try:
        #         _run_raw_test(b_mdl, test_pair, target_dir, ds_mode, thres)
        #     except Exception as exc:
        #         print(f"[WARN] Dataset test failed for {run_dir.name} on {test_pair.name}: {type(exc).__name__}: {exc}")
        # ds_mode = 'video' if vid_mode else 'clip'
        for tst_npz in ds_npz:
            try:
                _run_raw_test(b_mdl, tst_npz, target_dir, ds_eval, thres)
            except Exception as exc:
                print(f"[WARN] Dataset test failed for {run_dir.name} on {tst_npz.name}: {type(exc).__name__}: {exc}")

        if streams:
            try:
                schema_source = stream_schema
                if schema_source is None:
                    config_path = b_mdl.parent/'config.json'
                    if not config_path.is_file():
                        raise FileNotFoundError(f"no stream_schema and no config file: {config_path}")
                    with config_path.open('r', encoding='utf-8') as f:
                        schema_source = json.load(f)
                schema = resolve_stream_schema(schema_source)
                _run_stream_test(b_mdl, streams, source_name, schema, target_dir, thres,
                                 stream_failures)
            except Exception as exc:
                print_color(
                    f"[WARN] Stream test skipped for {run_dir.name}: {type(exc).__name__}: {exc}", 'o')

        tested.append(target_dir)

    if summary:
        summary_dir = Path(summary) if summary not in (True, False, None) else Path(tested[0].parent if tested else '.')
        try:
            sum_all_results(summary_dir, save_json=True)
        except Exception as exc:
            print(f"[WARN] sum_all_results failed for {summary_dir}: {type(exc).__name__}: {exc}")

    config_dir = Path(out_dir) if out_dir is not None else \
                 (tested[0] if len(tested) == 1 else tested[0].parent if tested else None)
    if config_dir is not None:
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            with (config_dir/'test-config.json').open('w', encoding='utf-8') as f:
                json.dump(config_value(rerun_kwargs), f, indent=2)
        except Exception as exc:
            print_color(f"[WARN] Failed saving test config in {config_dir}: "
                        f"{type(exc).__name__}: {exc}", 'o')

    return tested


#* region video stream to json

def reconvert_streams(stream_dir, video_dir, output_dir, ann_path=None, grp_tag=None, **kwargs):
    """Reconvert videos referenced by Stream JSONs while preserving their stream names."""
    from video_to_stream_data import VIDEO_SUFFIXES, process_video

    def index_files(root: Path, suffixes: set[str]) -> dict[str, list[Path]]:
        """ Index relevant files recursively by stem."""
        index = {}
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in suffixes:
                index.setdefault(path.stem, []).append(path)
        return index

    def resolve_video(vid_ref: Path) -> Path:
        """ Resolve one video stem, using its original parent to disambiguate duplicates."""
        candidates = video_index.get(vid_ref.stem, [])
        if len(candidates) == 1:
            return candidates[0]
        parent_matches = [p for p in candidates if p.parent.name == vid_ref.parent.name]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if not candidates:
            raise FileNotFoundError(f'video not found below {video_dir}: {vid_ref.name}')
        raise ValueError(f'ambiguous video stem {vid_ref.stem}: {len(candidates)} matches')

    def infer_tags(strm: dict):
        """ Resolve a constant stream tag from its first and last frames."""
        frames = strm.get('frames')
        if not isinstance(frames, list) or not frames:
            raise ValueError('group event cannot be resolved: stream has no frames')
        first = frames[0].get('group_events')
        last = frames[-1].get('group_events')
        if first is None or last is None:
            raise ValueError('group event cannot be resolved: first or last frame has no group_events')
        if first != last:
            raise ValueError(f'group event cannot be resolved: first={first}, last={last}')
        return first

    stream_dir, video_dir, output_dir = Path(stream_dir), Path(video_dir), Path(output_dir)
    if not stream_dir.is_dir():
        raise NotADirectoryError(stream_dir)
    if not video_dir.is_dir():
        raise NotADirectoryError(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index once because duplicate-stem resolution is needed for every source stream.
    video_index = index_files(video_dir, set(VIDEO_SUFFIXES))
    sources = list_json_sources(stream_dir)
    generated, skipped = [], []

    # An empty matching annotation disables process_video's automatic sibling lookup.
    with TemporaryDirectory(prefix='wesmart-no-annotations-') as empty_ann_dir:
        empty_ann_dir = Path(empty_ann_dir)
        for src_path in sources:
            try:
                stream = load_json_raw(src_path)
                video_value = stream.get('video')
                if not video_value:
                    raise ValueError("missing 'video' field")
                video_path = resolve_video(Path(video_value))

                tags = grp_tag
                process_ann = ann_path
                if ann_path is None:
                    if tags is None:
                        tags = infer_tags(stream)
                    process_ann = empty_ann_dir
                    (empty_ann_dir/video_path.stem).with_suffix('.ann').touch(exist_ok=True)

                before = set(list_json_sources(output_dir))
                process_video(video_path, output_dir/f'{src_path.stem}.json',
                              ann_path=process_ann, default_grp_tag=tags, **kwargs)
                generated.extend(resolve_json_source(path)
                                 for path in set(list_json_sources(output_dir)) - before)
            except Exception as exc:
                reason = f'{type(exc).__name__}: {exc}'
                cli_warning(f'Skipping {src_path.name}: {reason}', 'o')
                skipped.append({'stream': src_path.name, 'reason': reason})

    print(f"Reconversion: {len(sources) - len(skipped)}/{len(sources)} streams scheduled; "
          f"{len(generated)} outputs created")
    return {'total': len(sources), 'scheduled': len(sources) - len(skipped),
            'outputs': generated, 'skipped': skipped}

def convert_vid_2_json():
    from video_to_stream_data import process_video
    # main_dir = Path("data/video")
    main_dir = Path("/mnt/local-data/Projects/Wesmart/Video-datasets")
    json_dir = Path("data/json_files")
    fps, grp_tag = 3, 0

    #* RLVS
    out_dir = json_dir/"RLVS/3fps"
    vid_dir = main_dir/"RLVS/NonViolence"
    # process_video(vid_dir, out_dir, default_grp_tag=0, sample_rate=fps, zip_output=False)
    vid_dir = main_dir/"RLVS/Violence"
    # process_video(vid_dir, out_dir, default_grp_tag=4, sample_rate=fps, zip_output=False)

    #* RWF-2000
    out_dir = json_dir/"RWF-2000/3fps/NonFight"
    vid_dir = main_dir/"RWF-2000/train/Train_NonFight/"
    # process_video(vid_dir, out_dir, default_grp_tag=0, sample_rate=fps, zip_output=False)
    vid_dir = main_dir/"RWF-2000/val/Val_NonFight/"
    # process_video(vid_dir, out_dir, default_grp_tag=0, sample_rate=fps, zip_output=False)
    out_dir = json_dir/"RWF-2000/3fps/Fight"

    vid_dir = main_dir/"RWF-2000/train/Train_Fight/"
    process_video(vid_dir, out_dir, default_grp_tag=4, sample_rate=fps, zip_output=False)
    vid_dir = main_dir/"RWF-2000/val/Val_Fight/"
    process_video(vid_dir, out_dir, default_grp_tag=4, sample_rate=fps, zip_output=False)
    return
    #* UBI
    vid_dir = main_dir/"UBI_FIGHTS/videos/fight"
    ann_dir = main_dir/"UBI_FIGHTS/ann_ws_ready"
    out_dir = json_dir/"UBI/fight2"
    process_video(vid_dir, out_dir, ann_path=ann_dir, skip_without_ann=True,
                  default_grp_tag=grp_tag, sample_rate=fps, zip_output=False)
    out_dir = json_dir/"UBI/5fps/fight"
    fps = 5
    process_video(vid_dir, out_dir, ann_path=ann_dir, skip_without_ann=True,
                  default_grp_tag=grp_tag, sample_rate=fps, zip_output=False)

def run_stream_json_dual(data_dir, output_dir,tag=None, **kwargs):
    """Run two stream-JSON conversions for one video dir: plain and `group 0`."""
    # from video_to_stream_data import process_video
    data_dir, output_dir = Path(data_dir),  Path(output_dir)

    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_str =  tag if tag is not  None else datetime.now().strftime("%y%m%d") #    "260312"
    dir_none = output_dir/(t_str + '_g-na')
    dir_zero = output_dir/(t_str + '_g-0')
    common_kwargs = {'sample_rate': kwargs.get('sample_rate', 5),
                     'yolo_threshold': kwargs.get('yolo_threshold', 0.5),
                     'model_path' : kwargs.get('model_path', None),
                     'zip_output' : False}
    # process_video(data_dir, output_path=dir_none, **common_kwargs)
    # process_video(data_dir, output_path=dir_zero, default_grp_tag=[0], **common_kwargs)

#* endregion

#*** region study specific scripts  ***

def train_test_study(cache_dir:str|Path, **kwargs): #92 -> 63
    """ Train every cache in a study directory and run clip/video evaluations.
    Usage:
    - point `cache_dir` at a directory of `*_train.npz` study caches
    - by default, reusable existing runs are skipped when a usable model already exists
    """

    def _dataset_tag(stem:str, split_suffix:str) -> str:
        """Strip the split suffix from a cache stem."""
        return stem[:-len(split_suffix)] if stem.endswith(split_suffix) else stem

    def _best_model_info(model_dir: Path) -> tuple[Path, int]:
        """Return the saved best-model path and its epoch number."""
        best_models = sorted(model_dir.glob("best_model.*.pt"))
        if not best_models:
            raise FileNotFoundError(f"No best_model.*.pt found in {model_dir}")
        bm = best_models[-1]
        be = int(bm.stem.split(".")[-1])
        return bm, be

    def _existing_run_dir(tag: str, base_work_dir: Path) -> Path | None:
        """ Return the newest matching prior run dir for `tag`, if it is usable."""
        if not base_work_dir.is_dir():
            return None

        run_name = re.compile(rf"^\d{{6}}_\d{{2}}-\d{{2}}-\d{{2}}_{re.escape(tag)}$")
        matches = [p for p in base_work_dir.iterdir()  if p.is_dir() and run_name.fullmatch(p.name)]
        ready_runs = [p for p in matches if any(p.glob("best_model.*.pt"))]
        if ready_runs:
            return sorted(ready_runs)[-1]
        return None

    def _run_one_test(test_npz:Path, test_mode: str):
        """ Run one test job and the matching analysis."""
        output_tag = f"{get_exporting_name(best_model, test_npz, 'raw', unit=test_mode)}.npz"
        output_name = f"{get_exporting_name(best_model, test_npz, 'summary', unit=test_mode)}.json"

        if test_mode == 'clip':
            res = run_testing(best_model, test_npz, out_dir=run_dir, output_tag=output_tag)
            analyze_clip_test(res["path"], out_path=run_dir, output_name=output_name, show_roc=False)
            return
        else: # test_mode == 'video'
            res = run_testing(best_model, test_npz, video_mode=True,
                              out_dir=run_dir, output_tag=output_tag)
            analyze_video_test(res['path'], out_path=run_dir, output_name=output_name,show_roc=False,)

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)

    # work_dir = Path("work_dirs/json_models")/cache_dir.stem
    work_dir = MAIN_WORK_DIR/cache_dir.stem
    train_caches = sorted(cache_dir.glob("*_train.npz"))
    if not train_caches:
        raise FileNotFoundError(f"No *_train.npz caches found in {cache_dir}")

    for train_cache in train_caches:
        train_tag = _dataset_tag(train_cache.stem   , "_train")
        try:
            run_dir = _existing_run_dir(train_tag, work_dir) if kwargs.get('skip_existing', True) else None
            if run_dir is None:
                run_dir = run_training(train_cache, tag=train_tag, work_dir=work_dir, save_every=20)
                run_dir = Path(run_dir)
            else:
                print(f"Skipping training for {train_tag}: using {run_dir.name}")

            best_model, best_epoch = _best_model_info(run_dir)
        except Exception as exc:
            print(f"[train_test_stdy] Training failed for {train_tag}: {type(exc).__name__}: {exc}")
            continue

        pair_name = get_split_pair(train_cache)
        own_test = pair_name if isinstance(pair_name, Path) else cache_dir / pair_name
        suffix = train_tag.split("_", 1)[1] if "_" in train_tag else ""
        joint_test = cache_dir/f"{JOINT_DS}_{suffix}_test.npz" if suffix else cache_dir/f"{JOINT_DS}_test.npz"
        if train_tag.startswith(JOINT_DS):
            joint_test = own_test

        # Each run is tested on its own cache and, when available, the matching joint cache.
        test_targets = []
        for p in (own_test, joint_test):
            if p not in test_targets:
                if p.is_file():
                    test_targets.append(p)
                else:
                    print(f"[train_test_study] Missing test cache for {train_tag}: {p}")

        if not test_targets:
            print(f"[train_test_stdy] No test caches available for {train_tag}; skipping tests")
            continue

        for test_cache in test_targets:
            for test_mode in ('clip', 'video'):
                try:
                    _run_one_test(test_cache, test_mode)
                except Exception as exc:
                    print(f"[train_test_stdy] Test failed for {train_tag} on {test_cache.name} ({test_mode}): {type(exc).__name__}: {exc}")


def build_window_study(): # 80 -> 65
    """ Small batch script for the window/stride cache study.
    - uses the hard-coded WINDOW_SETTINGS
    - Builds train/test caches for  RWF-2000 and  RLVS datasets using the existing split files
    in each dataset directory, then merges the matching train/test caches into a joint
    dataset per window/stride option.
    """
    # WINDOW_SETTINGS = [(2.0, 1.0), (3.0, 1.5), (3.0, 1.0), (4.0, 2.0), ]
    WINDOW_SETTINGS = [(0.6, 0.4),
                       (1.2, 0.6),
                       (3.6, 1.2),
                       (5.0, 2.5),]
    def _fmt_num(x: float) -> str:
        """ Format numeric values for filenames, replacing '.' with 'o'."""
        return str(int(x)) if float(x).is_integer() else str(x).replace(".", "o")

    def _cache_name(name: str) -> str:
        """Return the requested cache filename format."""
        return f"{name}_25ft_{_fmt_num(window)}w-{_fmt_num(stride)}_{split}.npz"

    def _build_one() -> Path:
        """Build one cache file for a dataset/split/window configuration."""
        list_file = ds_dir/f"{split}_videos.txt"
        out_path = STUDY_CACHE_DIR / _cache_name(ds_name)

        if not list_file.is_file():
            raise FileNotFoundError(f"Missing split file: {list_file}")

        with open(list_file, 'r') as f:
            json_paths = [ds_dir / ln.strip() for ln in f if ln.strip()]

        build_cache_from_json(json_paths, out_path, window=window, stride=stride)
        return out_path.with_suffix(".npz")

    STUDY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for window, stride in WINDOW_SETTINGS:
        for split in ("train", "test"):
            built = []
            for ds_name, ds_dir in DATASETS:
                built.append(_build_one())
            merge_cache_npz(built, STUDY_CACHE_DIR/_cache_name(JOINT_DS))


#* endregion *#

#_______________________________________________________________________#
# * local runners (not to be used outside this model)  ***

#CURRENT_CACHE_DIR = MAIN_CACHE_DIR/"Joint_sets"
CURRENT_CACHE_DIR = MAIN_CACHE_DIR/"gen_03"

#*  Cache Building ***#
def cache_builder():
    # cache_dir = MAIN_CACHE_DIR/"new_format"
    cache_dir = CURRENT_CACHE_DIR
    json_dirs = [Path("data/json_files/HMC/ann-streams"),
                 Path("data/json_files/HMC/cam-streams"),
                 Path("data/json_files/HMC/events"),
                 Path("data/json_files/RLVS/5fps"),
                 Path("data/json_files/RWF-2000/5fps"),
                 ]
    pooling = ['max', 'lse', 'top_k', 'mm']
    t_slc = [(3.6, 1.2),
             (3.0, 1.0),
             (1.2, 0.6)]
    yolo_th = 0.4
    #build_cache_batch(json_dirs, pooling, t_slc, {'cache_dir': cache_dir})
    ubi_json_dir = Path("data/json_files/UBI/6fps")
    # ubi_pooling = pooling
    # ubi_t_slc = t_slc
    build_cache_batch(ubi_json_dir, pooling, t_slc,
                      output_dir=cache_dir,
                      split_dir=cache_dir / 'UBI-6fps_ttp',
                      root_dir=Path.cwd(),
                      split_ratio=0.2, random_seed=42)

    # *  Test test_models


# DEFAULT_MDL_DIR = "work_dirs/models"
STREAM_TEST_DIR = "data/json_files/testing"
DEFAULT_MDL_DIR = "work_dirs/models-lib"
DEFAULT_RES_DIR = "work_dirs/testing-lib"


STREAM_SUB_TST = ["data/json_files/testing/weSmart_demo.json",
                  "data/json_files/testing/Russian_Road_Rage- Micky_Mouse_&_Sponge_Bob.json",
                  "data/json_files/testing/F_60_1_2_0_0.json",
                  "data/json_files/testing/F_121_1_0_0_0.zip",
                  "data/json_files/testing/N_529_0_1_1_0.zip",
                  "data/json_files/testing/N_390_0_0_1_0.zip",
                  "data/json_files/testing/N_383_0_0_1_0.zip",
                  "data/json_files/testing/N_115_0_0_1_0.zip"]

def test_runner(tst_strm, **kwargs):

    t0 = time.time()

    mdl_dir = Path(kwargs.pop('mdl_dir', DEFAULT_MDL_DIR))
    root_path = kwargs.pop('root_path', None)
    kwargs.setdefault('out_dir', DEFAULT_RES_DIR)
    out_dir = Path(kwargs['out_dir'])

    tst_strm = Path(tst_strm)
    if tst_strm.is_file() and tst_strm.suffix.lower() == '.txt':
        # tst_strm = list_file_list(tst_strm, kwargs.get('root_path'))
        tst_strm = resolve_json_files(tst_strm, root_path)
    elif tst_strm.is_dir():
        tst_strm = sorted({p for sfx in STREAM_FILE_TYPES for p in tst_strm.rglob(f'*{sfx}')})
    else:
        tst_strm = [tst_strm]
    # tst_strm = None #Path("/mnt/local-data/Python/Projects/weSmart/data/cache/tmp_test/strm")
    # tst_strm = strm_test_set # Path("data/json_files/testing")
    # tl_chart = kwargs.pop('plot_charts', False)

    kwargs.setdefault('summary', out_dir)
    kwargs.setdefault('test_pair', True)
    kwargs.setdefault('plotting', False)
    kwargs.setdefault('threshold', kwargs.pop('th', [0.5, 0.6]))
    test_models(mdl_dir, stm_tests=tst_strm, **kwargs)

    print(f"\n--- Streams for inference: ({len(tst_strm)} in total) ---")
    for path in tst_strm:
        print(f"\t{path.name}")
    print(f"test_runner for {out_dir.name} completed; duration for {time.time() - t0:4f}\n{'*'*80}\n")

# * endregion

#1358(7,32,7)
#1354(8,32,8)-> strm-util 1300(.)
#1174(8,28,8) -> 1370(10,36,8)-> 1282(29,29,8)-
#>

if __name__ == "__main__":
    pass

    # cache_builder()
    test_runner(tst_strm=Path(STREAM_TEST_DIR)/"test_er-24_default.txt",
                mdl_dir=Path(DEFAULT_MDL_DIR)/'G3',
                out_dir=Path(DEFAULT_RES_DIR)/'gen-3/test-02',
                plotting=True, th=[0.6, 0.7],
                )

    #* region Train models
    cache_dir = Path("data/cache/Joint_sets")
    cache_dir = Path("data/cache/gen_03/Joint_sets")
    work_dir = Path("work_dirs/models")
    sum_trn = True

    # train_models(cache_dir, work_dir, run_tests=True, summary=sum_trn)

    #endregion
    # _______________________________________________________________________#
    #* region Time windows study  ***
    study_dir = 'win-study-tst'
    # study_dir = 'ftr-study'
    STUDY_CACHE_DIR = MAIN_CACHE_DIR/study_dir
    #* endregion

    #* train & test for win study
    # build_window_study()
    # train_test_stdy(STUDY_CACHE_DIR)
    # sum_all_results(MAIN_WORK_DIR/study_dir, sort=['win-str','vid-clp', 'trn-tst-R'],save_json=True)
    # * endregion

    # cache_dir = "data/cache/w30-15_um"
    # output_dir= "work_dirs/json_models/w30-15-um"
    # stream_testing = ["cam-6-11-5_ft25_w30-15.npz",
    #                   "cam-6-11-8_FRes_Ana_ft25_w30-15.npz",
    #                   "cam-6-11-8_FRes_Erz_ft25_w30-15.npz"]
    # ds_testsing = ['J-All_ft25_w30-15_test.npz']
    # train_models(cache_dir, output_dir, ds_tests=ds_testsing, stm_tests=stream_testing)
    # sum_all_results(output_dir)

    d_d = "/mnt/local-data/Projects/Wesmart/Video-datasets/draft_set/tst_conv"
    #op_d = "data/json_files/tst_conv/test_260611_batch"
    op_d = "data/sanity-testing/json/"
    # run_stream_json_dual(d_d, op_d, '260611-no_imgsz'  )
    # run_stream_json_dual(d_d, op_d, '260312' )

    # convert_vid_2_json()
