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

import json, pickle, glob
import re
import time
from datetime import datetime
from pathlib import Path
import numpy as np
#* Imports from local project
from precompute_clips import (build_cache_from_json, merge_cache_npz, get_split_pair, WINDOW_SEC, STRIDE_SEC,
                              RANDOM_SEED, split_json_ds, MOTION_FPS_REF)
from tms_trainer import run_training, run_testing
from evaluation_core import analyze_clip_test, analyze_video_test, support_pair, DEFAULT_EVAL_THRESHOLD
from stream_analysis import analyze_stream_test
from common.my_local_utils import as_collection, get_unique_name, print_color
from json_utils import list_json_sources
from motion_feature_schema import load_cache_contract_compact
from project_utils import get_exporting_name, resolve_best_pt_model, strip_split_suffix, strip_timestamp_prefix

#* general configuration
RWF_DIR  = Path("data/json_files/RWF-2000/ds")
RLVS_DIR = Path("data/json_files/RLVS/ds")

MAIN_WORK_DIR = Path("work_dirs/json_models")
MAIN_CACHE_DIR = Path("data/cache")
# STUDY_CACHE_DIR  = MAIN_CACHE_DIR/"win-study"

DATASETS = [('RWF', RWF_DIR), ('RLVS', RLVS_DIR)]
JOINT_DS =  'J-RWL'
RESULT_NAME = 'all_results'

#*** region cache building ***

def build_cache(json_dir, cache_dir=None, *, pool_mode='max', window=WINDOW_SEC, stride=STRIDE_SEC,
                cache_tag=None, **kwargs):
    """ Build train/test caches for one JSON dir and one cache configuration."""

    def _json_paths_from_list(json_dir: Path, list_file: Path) -> list[Path]:
        with list_file.open('r', encoding='utf-8') as handle:
            names = [ln.strip() for ln in handle if ln.strip()]
        return [json_dir/name for name in names]

    def _dir_tag(j_dir: Path)->str:
        return f"{j_dir.parent.name}-{j_dir.name}"

    def _pool_tag(pool_name: str) -> str:
        return {'mean_max': 'mm', 'mean_std_max': 'msm'}.get(pool_name, pool_name)

    def _time_tag(win:float, strd:float) -> str:
        return f"{str(win).replace('.', '')}-{str(strd).replace('.', '')}"

    def _stream_src(j_dir: Path) -> str:
        return f"{j_dir.parent.name}_{j_dir.name}"

    def _write_log(row: dict):
        columns = [('time-stamp',20,'<'), ('status',8,'<'), ('source',12,'<'),('count',8,'>'),
                   ('set',7,'<'), ('split', 10, '>'),  ('fps ref', 8, '>'), ('pool',8,'<'),
                   ('win-stride', 14,'<'), ('t', 8, '>'), ('cache name', 0, '<')]

        def fmt_line(values: dict) -> str:
            parts = []
            for key, width, align in columns:
                txt = str(values[key])
                parts.append(f"{txt:{align}{width}}" if width else txt)
            return '  '.join(parts)

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            need_header = not log_path.is_file() or log_path.stat().st_size == 0
            with log_path.open('a', encoding='utf-8') as f:
                if need_header:
                    f.write(fmt_line({key: key for key, _, _ in columns}) + '\n')
                f.write(fmt_line(row) + '\n')
        except Exception as exc:
            print(f"[WARN] Failed to write cache build log {log_path}: {type(exc).__name__}: {exc}")

    def _build_one_cache(json_paths: list[Path], out_path: Path):
        control_keys = {'cache_dir', 'split_ratio', 'test_ratio', 'random_seed', 'log_path'}
        build_kwargs = {key: val for key, val in kwargs.items() if key not in control_keys}
        build_kwargs.update({'window': window, 'stride': stride, 'pool_mode': pool_mode})
        build_cache_from_json(json_paths, out_path, **build_kwargs)

    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise NotADirectoryError(json_dir)

    out_dir = Path(cache_dir or kwargs.get('cache_dir', MAIN_CACHE_DIR/'new_format'))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(kwargs.get('log_path', out_dir/'caching_log.txt'))
    split_seed = kwargs.get('random_seed', RANDOM_SEED)
    fps_ref = kwargs.get('motion_fps_ref', MOTION_FPS_REF)

    train_txt = json_dir/'train_videos.txt'
    test_txt = json_dir/'test_videos.txt'
    has_split = train_txt.is_file() and test_txt.is_file() and 'split_ratio' not in kwargs
    all_jsons = list_json_sources(json_dir)
    if not all_jsons:
        print(f" No JSON sources found in {json_dir}")
        return []

    if has_split:
        train_jsons = _json_paths_from_list(json_dir, train_txt)
        test_jsons = _json_paths_from_list(json_dir, test_txt)
        split_jobs = [('train', train_jsons), ('test', test_jsons)]
    else:
        split_ratio = kwargs.get('split_ratio', kwargs.get('test_ratio', 0.0))
        if split_ratio:
            splits = split_json_ds(json_dir, test_ratio=split_ratio, random_seed=kwargs.get('random_seed', RANDOM_SEED))
            train_jsons, test_jsons = splits['train'], splits['test']
        else:
            train_jsons, test_jsons = all_jsons, []
        split_jobs = [('train', train_jsons)]
        if test_jsons:
            split_jobs.append(('test', test_jsons))

    if cache_tag is None:
        cache_tag = f"{_dir_tag(json_dir)}_P-{_pool_tag(str(pool_mode))}_W{_time_tag(window, stride)}"

    results = []
    split_total = len(train_jsons) + len(test_jsons)
    for split_name, json_paths in split_jobs:
        if split_name == 'test' and not has_split and not test_jsons:
            continue
        out_path = out_dir/f"{cache_tag}_{split_name}.npz"
        t0 = time.time()
        try:
            print(f"building cache: {out_path.name}  ")
            _build_one_cache(json_paths, out_path)
            ok = out_path.is_file()
            t = time.time() - t0
        except Exception as exc:
            ok = False
            t = time.time() - t0
            print(f"[FAIL] {out_path}: {type(exc).__name__}: {exc}")
        else:
            print(f"OK | t = {t:.2f}  |  {out_path.name} " if ok else f"FAILED {out_path.name}")

        split_part = (len(json_paths)/split_total) if split_total else 0.0
        _write_log({'time-stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'OK' if ok else 'FAIL',
                    'stream_src': _stream_src(json_dir),
                    'n_jsons': len(json_paths),
                    'set': split_name,
                    'split': f"{split_part:.3f}/{split_seed}",
                    'fps_ref': fps_ref,
                    'pool': pool_mode,
                    'window-stride': f"{window:g}-{stride:g}",
                    't_wrk': f"{t:.2f}",
                    'cache_name': out_path.name})
        results.append({'json_dir': json_dir, 'pool': pool_mode, 'window': window, 'stride': stride,
                        'split': split_name, 'cache': out_path, 'ok': ok, 't_wrk': t})
    return results


def build_cache_batch(dir_ls, pooling, t_slc, kwargs=None):
    """Build a cache grid for one or more JSON dirs, pool modes, and time slices."""
    kwargs = dict(kwargs or {})
    out_dir = Path(kwargs.get('cache_dir', MAIN_CACHE_DIR/'new_format'))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for json_dir in as_collection(dir_ls):
        json_dir = Path(json_dir)
        if not json_dir.is_dir():
            raise NotADirectoryError(json_dir)
        cache_kwargs = {key: val for key, val in kwargs.items() if key != 'cache_dir'}
        for pool_name in as_collection(pooling):
            pool_name = str(pool_name)
            for slc in as_collection(t_slc):
                if isinstance(slc, dict):
                    win  = float(slc['window'])
                    strd = float(slc['stride'])
                elif isinstance(slc, (tuple, list)) and len(slc) >= 2:
                    win = float(slc[0])
                    strd = float(slc[1])
                elif isinstance(slc, str):
                    parts = [p.strip() for p in slc.split(':') if p.strip()]
                    if len(parts) != 2:
                        raise ValueError(f"Bad time-slice spec: {slc}")
                    win, strd = float(parts[0]), float(parts[1])
                else:
                    raise TypeError(f"Unsupported time-slice spec: {slc}")

                results.extend(build_cache(json_dir, out_dir, pool_mode=pool_name,
                                           window=win, stride=strd, **cache_kwargs))
    return results


def build_caches(dir_ls, pooling, t_slc, kwargs=None):
    """ Compatibility wrapper. Prefer build_cache_batch(...)."""
    return build_cache_batch(dir_ls, pooling, t_slc, kwargs)


def merge_caches(cache_path, output_path=None):
    """Merge same-tag train/test caches in one directory into Joint_* outputs."""
    cache_path = Path(cache_path)
    if not cache_path.is_dir():
        raise NotADirectoryError(cache_path)

    out_dir = Path(output_path) if output_path is not None else cache_path
    out_dir.mkdir(parents=True, exist_ok=True)

    def _split_tag(stem: str) -> tuple[str, str]|None:
        if stem.startswith("Joint_"):
            return None
        if stem.endswith("_train"):
            set_t = "train"
        elif stem.endswith("_test"):
            set_t = "test"
        else:
            return None

        base = stem[:-len(f"_{set_t}")]
        if '_' not in base:
            return None
        return base.split("_", 1)[1], set_t

    groups: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(cache_path.glob("*.npz")):
        tag_info = _split_tag(path.stem)
        if tag_info is None:
            continue
        tag, split = tag_info
        groups.setdefault(tag, {}).setdefault(split, []).append(path)

    merged = []
    for tag, split_map in sorted(groups.items()):
        for split in ("train", "test"):
            members = split_map.get(split, [])
            if not members:
                print(f"[WARN] Missing {split} caches for tag {tag}")
                continue

            out_name = get_unique_name(out_dir / f"Joint_{tag}_{split}.npz")
            try:
                merge_cache_npz(members, out_name)
                print(f"[OK]   {out_name}")
                merged.append(out_name)
            except Exception as exc:
                print(f"[FAIL] {out_name}: {type(exc).__name__}: {exc}")

    return merged

#* endregion *#

def resolve_npz_inputs(inputs, base_dir: Path | None = None) -> list[Path]:
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
    for summary_path in sorted(Path(run_dir).rglob('*-summary.json')):
        try:
            with summary_path.open('r', encoding='utf-8') as f:
                summary = json.load(f)
            return float(summary.get('analysis_config', {}).get('threshold', default))
        except Exception:
            continue
    return float(default)


def evaluate_raw_test(raw_path, mode, out_dir, threshold=DEFAULT_EVAL_THRESHOLD, **kwargs):
    """Evaluate one raw test NPZ as clip, video, or stream output."""

    raw_path, out_dir  = Path(raw_path), Path(out_dir)

    with np.load(raw_path, allow_pickle=True) as data:
        model_path = data['model_path'].item() if isinstance(data['model_path'], np.ndarray) else data['model_path']
        test_cache = data['test_cache'].item() if isinstance(data['test_cache'], np.ndarray) else data['test_cache']

    output_name = get_exporting_name(model_path, test_cache, 'summary', unit=mode)
    common = {'out_path': out_dir,
              'threshold': threshold,
              'threshold_dir': Path(f"th-{int(round(threshold * 100.0))}"),
              'overwrite': True,
              'show_roc': kwargs.get('show_roc', False),
              'roc_csv': kwargs.get('roc_csv', True),
              'print_policy': kwargs.get('print_policy', 'summary'),
              'print': kwargs.get('print_report', False),}

    if   mode == 'stream':
        return analyze_stream_test(raw_path, output_name=output_name,
                                   details_name=f"{get_exporting_name(model_path, test_cache, 'events')}.json",
                                   events_json=kwargs.get('events_json', True),
                                   plotting=kwargs.get('plotting', 'save'), **common)
    elif mode == 'video':
        return analyze_video_test(raw_path, output_name=output_name, **common)
    elif mode == 'clip':
        return analyze_clip_test(raw_path, output_name=output_name, **common)
    else:
        raise ValueError(f'Unrecognized mode: {mode}')


def sum_all_results(work_dir: str | Path, **kwargs):  # 107 -250
    """ Collect all summary JSON files under one work dir into one flat results table.
    Usage:
    :param work_dir: at a model/run root containing `*-summary.json` files
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

        cm = summary.get('confusion_matrix', [[None, None], [None, None]])
        auc = summary.get('ROC AUC', summary.get('roc_auc', None))
        return {'model': str(model_path), 'cache': test_cache.stem,
                'train ds': train_tag or train_ds,
                'test ds': test_tag or test_ds,
                'unit': unit, 'samples': samples, 'support': support_str,
                'window': window, 'stride': stride, 'pool': pool,
                'fps_ref': fps_ref, 'feat_dim': feat_dim, 'threshold': threshold,
                'FF': cm[0][0],
                'FT': cm[0][1],
                'TF': cm[1][0],
                'TT': cm[1][1],
                'Acc': summary.get('accuracy', None),
                'Rec': summary.get('recall', None),
                'FPR': summary.get('FPR', None),
                'AUC': auc,
                }

    def _print_rows(rows: list[dict]):
        """Print the aggregated table in a simple aligned layout."""

        def _clip_text(text, limit=24) -> str:
            text = str(text)
            return text if len(text) <= limit else text[:limit - 1] + '…'

        cols = ['model', 'BE', 'train ds', 'test ds', 'window', 'stride', 'pool', 'fps_ref', 'feat_dim',
                'threshold', 'unit',
                'samples', 'support', 'FF', 'FT', 'TF', 'TT', 'Acc', 'Rec', 'FPR', 'AUC']
        cols = [col for col in cols if col == 'BE' or any(col in row for row in rows)]
        header_labels = {c: c if c.isupper() else c.title() for c in cols}
        header_labels.update({'train ds': 'Train ds', 'test ds': 'Test ds',
                              'fps_ref': 'FPS ref', 'feat_dim': 'Feat dim'})
        display_rows = []
        for row in rows:
            disp = row.copy()
            disp['model'], disp['BE'] = _model_disp(disp['model'])
            disp['train ds'] = _clip_text(disp['train ds'])
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

    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        raise NotADirectoryError(work_dir)

    summary_paths = sorted(work_dir.rglob('*-summary.json'))
    if not summary_paths:
        raise FileNotFoundError(f"No *-summary.json files found in {work_dir}")

    table = [_row_from_summary(p) for p in summary_paths]
    _sort_table(table)
    table = _drop_na_columns(table)

    output_path = work_dir/kwargs.get('op_name', RESULT_NAME)
    with (output_path.with_suffix('.pkl')).open('wb') as f:
        pickle.dump(table, f)

    if kwargs.get('save_json', False):
        with (output_path.with_suffix('.json')).open('w') as f:
            json.dump(table, f, indent=2)

    if kwargs.get('print_cli', True):
        _print_rows(table)

    return table


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
    """ Run tests for existing trained models without retraining them.
    :param models: single or list of model dir/ checkpoint path
    :param ds_tests, stm_tests:  list or fir off npz files for dataset/stream evaluations
    :param kwargs['summary']: if summary==True aggregates summaries with sum_all_results(...)
                              summary=<path> stores the aggregate summary
    """

    npz_dir = kwargs.pop('npz_dir', None)
    out_dir = kwargs.pop('out_dir', None)
    evaluate = kwargs.pop('evaluate', True)
    infer_threshold = kwargs.pop('infer_threshold', False)
    threshold_arg = kwargs.pop('threshold', None)
    summary = kwargs.pop('summary', False)
    test_pair = kwargs.pop('test_pair', False)
    ds_eval = kwargs.pop('ds_eval_mode', 'clip')
    npz_dir = Path(npz_dir) if npz_dir is not None else None

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

    def _threshold_for_run(run_dir: Path) -> float:
        if threshold_arg is not None:
            return float(threshold_arg)
        if infer_threshold:
            return infer_eval_threshold(run_dir, DEFAULT_EVAL_THRESHOLD)
        return float(DEFAULT_EVAL_THRESHOLD)

    def _run_raw_test(model_path:Path, tst_npz:Path, out_dir:Path, mode:str, thrsh:float):
        raw_tag = get_exporting_name(model_path, tst_npz, 'raw', unit=mode)
        run_kwargs = {'video_mode': True}
        if kwargs.get('batch_size') is not None:
            run_kwargs['batch_size'] = kwargs['batch_size']
        res = run_testing(model_path, tst_npz, out_dir=out_dir, output_tag=raw_tag, **run_kwargs)
        if evaluate:
            evaluate_raw_test(res['path'], mode, out_dir, thrsh, **kwargs)

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
        threshold = _threshold_for_run(run_dir)

        ds_npz = []
        if test_pair:
            pair = _find_test_pair()
            if pair is not None:
                ds_npz.append(pair)
        pair = _find_test_pair()
        ds_npz = [pair] if pair is not None else []
        # if test_pair is not None and not test_pair.is_file():
        #     print_color(f"[WARN] Missing paired test cache for {run_dir.name}: {test_pair}", 'o')
        for tst_npz in resolve_npz_inputs(ds_tests, npz_dir):
            if tst_npz not in ds_npz:
                ds_npz.append(tst_npz)

        # if test_pair.is_file():
        #     try:
        #         _run_raw_test(b_mdl, test_pair, target_dir, ds_mode, threshold)
        #     except Exception as exc:
        #         print(f"[WARN] Dataset test failed for {run_dir.name} on {test_pair.name}: {type(exc).__name__}: {exc}")
        # ds_mode = 'video' if vid_mode else 'clip'
        for tst_npz in ds_npz:
            try:
                _run_raw_test(b_mdl, tst_npz, target_dir, ds_eval, threshold)
            except Exception as exc:
                print(f"[WARN] Dataset test failed for {run_dir.name} on {tst_npz.name}: {type(exc).__name__}: {exc}")

        for tst_npz in resolve_npz_inputs(stm_tests, npz_dir):
            try:
                _run_raw_test(b_mdl, tst_npz, target_dir, 'stream', threshold)
            except Exception as exc:
                print(f"[WARN] Stream test failed for {run_dir.name} on {tst_npz.name}: {type(exc).__name__}: {exc}")

        tested.append(target_dir)

    if summary:
        summary_dir = Path(summary) if summary not in (True, False, None) else Path(tested[0].parent if tested else '.')
        try:
            sum_all_results(summary_dir, save_json=True)
        except Exception as exc:
            print(f"[WARN] sum_all_results failed for {summary_dir}: {type(exc).__name__}: {exc}")

    return tested

#* region video stream to json

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
    from video_to_stream_data import process_video
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
    process_video(data_dir, output_path=dir_none, **common_kwargs)
    process_video(data_dir, output_path=dir_zero, default_grp_tag=[0], **common_kwargs)

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

#733(,20,4)-> 755(,23,6)
#build_caches 966(1,26,7)/ log 985(1,33,4)- build-rf 968(1,31,4)-cleanup : 926
#tst/trn-refactor 850(1,26,2)-> 872->
#sumallres-rfc 947(5,22,4)


#_______________________________________________________________________#
# * local runners (not to be used outside this model)  ***

CURRENT_CACHE_DIR = MAIN_CACHE_DIR/"Joint_sets"

#*  Cache Building ***#
def cache_builder():
    # cache_dir = MAIN_CACHE_DIR/"new_format"
    cache_dir = CURRENT_CACHE_DIR
    json_dirs = [Path("data/json_files/HMC/ann-streams"),
                 Path("data/json_files/HMC/cam-streams"),
                 Path("data/json_files/HMC/events"),
                 Path("data/json_files/RLVS/5fps"),
                 Path("data/json_files/RWF-2000/5fps"), ]
    pooling = ['max', 'lse', 'top_k', 'mm']
    t_slc = [(3.6, 1.2),
             (1.2, 0.6)]
    build_cache_batch(json_dirs, pooling, t_slc, {'cache_dir': cache_dir})
    ubi_json_dir = Path("data/json_files/UBI/6fps")
    # ubi_pooling = pooling
    # ubi_t_slc = t_slc
    build_cache_batch(ubi_json_dir, pooling, t_slc,
                      {'cache_dir': cache_dir, 'split_ratio': 0.2, 'random_seed': 42})

    # *  Test test_models
def test_runner():
    mdl_dir  = Path("/mnt/local-data/Python/Projects/weSmart/work_dirs/json_models/w30-15_models/w30-15-tst")
    op_dir = Path("/mnt/local-data/Python/Projects/weSmart/work_dirs/json_models/sanity-testing/testing")
    tst_ds = Path("/mnt/local-data/Python/Projects/weSmart/data/cache/tmp_test/ds")
    tst_strm = Path("/mnt/local-data/Python/Projects/weSmart/data/cache/tmp_test/strm")
    test_models(mdl_dir, out_dir=op_dir, ds_tests= tst_ds, stm_tests=tst_strm)

# * endregion

if __name__ == "__main__":
    pass

    # cache_builder()
    # test_runner()



    #* region Train models
    cache_dir = Path("data/cache/Joint_sets")
    work_dir = Path("work_dirs/models")
    sum_trn = True

    train_models(cache_dir, work_dir, run_tests=True, summary=sum_trn)

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
