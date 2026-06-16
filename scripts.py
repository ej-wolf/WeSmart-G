""" Project batch helpers for cache building, model train/test runs,
    result aggregation, and small local utility flows.
    Usage:
    - build caches from JSON directories with `build_cache_batch(...)`
    - run end-to-end train/test flows with `train_models(...)` or `train_test_study(...)`
    - rerun tests for existing models with `test_models(...)`
    - collect summary tables with `sum_all_results(...)`
    - run paired stream-JSON conversions with `run_stream_json_dual(...)`
"""

import json, pickle, glob
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
#* Imports from local project
from precompute_clips import (build_cache_from_json, merge_cache_npz, WINDOW_SEC, STRIDE_SEC,
                              TEST_RATIO, RANDOM_SEED, DEFAULT_TYPE, _run_build_cache_ds)
from tms_trainer import run_training, run_testing
from evaluation_core import analyze_clip_test, analyze_video_test, support_pair
from stream_analysis import analyze_stream_test
from common.my_local_utils import as_collection, get_unique_name
from project_utils import get_exporting_name

#* general configuration
RWF_DIR  = Path("data/json_files/RWF-2000/ds")
RLVS_DIR = Path("data/json_files/RLVS/ds")

MAIN_WORK_DIR = Path("work_dirs/json_models")
MAIN_CACHE_DIR = Path("data/cache")
# STUDY_CACHE_DIR  = MAIN_CACHE_DIR/"win-study"

DATASETS = [('RWF', RWF_DIR), ('RLVS', RLVS_DIR)]
JOINT_DS =  'J-RWL'
RESULT_NAME = 'all_results'
DEFAULT_REG_MODEL = Path("work_dirs/json_models/win-study/260414-1721_J-RWL_25ft_3w-1o5-stream-tst/best_model.148.pt")


def build_cache_batch(json_dirs, output_path, **kwargs):
    """ Build train/test caches for one or more dataset dirs.
    Usage:
        - pass one dir or a list of dirs in json_dirs
        - use `kwargs` to override the shared cache-build config for the whole batch
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    base_args = {'cache_name': None, 'split_dir': None, 'new_split': False, 'test_ratio': TEST_RATIO,
                 'random_seed': RANDOM_SEED, 'window': WINDOW_SEC, 'stride': STRIDE_SEC,
                 'allow_empty': False, 'pure_motion': False, 'legacy': False,
                 'no_temp_smooth': False, 'json_type': DEFAULT_TYPE,}
    base_args.update(kwargs)

    built = []
    for json_dir in as_collection(json_dirs):
        json_dir = Path(json_dir)
        if not json_dir.is_dir():
            raise NotADirectoryError(json_dir)

        run_args = SimpleNamespace(**base_args)
        run_args.jsons_dir = json_dir
        run_args.cache_dir = output_path
        run_args.cache_name = run_args.cache_name or json_dir.name
        run_args.split_dir = Path(run_args.split_dir) if run_args.split_dir else json_dir

        _run_build_cache_ds(run_args)
        built.append({'jsons_dir': json_dir,
                      'train_cache': output_path/f"{run_args.cache_name}_train.npz",
                      'test_cache': output_path/f"{run_args.cache_name}_test.npz",})
    return built


def train_models(cache_dir, main_op_dir, ds_tests=None, stm_tests=None, **kwargs):
    """ Train every `*_train.npz` cache in a directory and run post-training tests.
    :param cache_dir: should contain matching *_train.npz/*_test.npz files
    optional:  `ds_tests` and `stm_tests` add shared dataset and stream evaluations
    """

    def _resolve_npz_inputs(inputs, base_dir: Path) -> list[Path]:
        """Resolve NPZ files, dirs, or glob masks into one unique ordered list."""
        resolved = []
        seen = set()
        for item in as_collection(inputs or []):
            item = Path(item)
            if not item.is_absolute():
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
                print(f"[WARN] No NPZ files matched: {item}")
                continue

            for path in matches:
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(path)
        return resolved

    def _dataset_tag(stem: str, split_suffix: str) -> str:
        """Strip one split suffix from a cache stem."""
        return stem[:-len(split_suffix)] if stem.endswith(split_suffix) else stem

    def _best_model_info() -> tuple[Path, int]:
        """Return the newest saved best-model checkpoint and its epoch number."""
        best_models = sorted(run_dir.glob("best_model.*.pt"))
        if not best_models:
            raise FileNotFoundError(f"No best_model.*.pt found in {run_dir}")
        bm = best_models[-1]
        return bm, int(bm.stem.split(".")[-1])

    def _run_dataset_test():
        """ Run one clip-level dataset test and save its summary."""
        raw_tag = get_exporting_name(best_model, test_npz, 'raw', unit='clip')
        summary_name = get_exporting_name(best_model, test_npz, 'summary', unit='clip')
        res = run_testing(best_model, test_npz, out_dir=run_dir, output_tag=raw_tag)
        analyze_clip_test(res['path'], out_path=run_dir, output_name=summary_name, show_roc=False)

    def _run_stream_test():
        """Run one stream-level test and save stream-analysis outputs."""
        raw_tag = get_exporting_name(best_model, test_npz, 'raw', unit='stream')
        summary_name = get_exporting_name(best_model, test_npz, 'summary', unit='stream')
        events_name = f"{get_exporting_name(best_model, test_npz, 'events')}.json"
        res = run_testing(best_model, test_npz, out_dir=run_dir, output_tag=raw_tag, video_mode=True)
        analyze_stream_test(res['path'], out_path=run_dir, output_name=summary_name,
                            details_name=events_name, show_roc=False, plotting= 'save')

    cache_dir = Path(cache_dir)
    main_op_dir = Path(main_op_dir)
    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    main_op_dir.mkdir(parents=True, exist_ok=True)

    ds_targets = _resolve_npz_inputs(ds_tests, cache_dir)
    stm_targets = _resolve_npz_inputs(stm_tests, cache_dir)
    train_caches = sorted(cache_dir.glob("*_train.npz"))
    if not train_caches:
        print(f"[WARN] No *_train.npz caches found in {cache_dir}")
        return []

    built_runs = []
    for train_cache in train_caches:
        train_tag = _dataset_tag(train_cache.stem, "_train")
        model_dir = get_unique_name(main_op_dir/train_tag)
        try:
            run_dir = Path(run_training(train_cache, tag=train_tag, work_dir=model_dir.parent))
            if run_dir != model_dir:
                run_dir.rename(model_dir)
                run_dir = model_dir
            best_model, _ = _best_model_info()
        except Exception as exc:
            print(f"[WARN] Training failed for {train_cache.name}: {type(exc).__name__}: {exc}")
            continue

        # Always evaluate the model on its own paired `_test` cache before shared cross-tests.
        own_test = cache_dir/f"{train_tag}_test.npz"
        if own_test.is_file():
            try:
                test_npz = own_test
                _run_dataset_test()
            except Exception as exc:
                print(f"[WARN] Own dataset test failed for {train_tag}: {type(exc).__name__}: {exc}")
        else:
            print(f"[WARN] Missing own test cache for {train_tag}: {own_test}")

        for test_npz in ds_targets:
            try:
                _run_dataset_test()
            except Exception as exc:
                print(f"[WARN] Dataset test failed for {train_tag} on {test_npz.name}: {type(exc).__name__}: {exc}")

        for test_npz in stm_targets:
            try:
                _run_stream_test()
            except Exception as exc:
                print(f"[WARN] Stream test failed for {train_tag} on {test_npz.name}: {type(exc).__name__}: {exc}")

        built_runs.append(run_dir)

    try:
        sum_all_results(main_op_dir, save_json=kwargs.get('save_summery',True))
    except Exception as exc:
        print(f"[WARN] sum_all_results failed for {main_op_dir}: {type(exc).__name__}: {exc}")

    return built_runs


def test_models(models, general_tests=None, stream_tests=None, **kwargs):
    """Run tests for existing trained models without retraining them.

    Usage:
    - pass one model dir, one checkpoint path, or a list of them in `models`
    - optional `general_tests` / `stream_tests` add shared dataset and stream evaluations
    """

    def _dataset_tag(stem: str, split_suffix: str) -> str:
        """Strip one split suffix from a cache stem."""
        return stem[:-len(split_suffix)] if stem.endswith(split_suffix) else stem

    def _resolve_model_ref(mdl_ref) -> tuple[Path, Path]:
        """Resolve a model ref to `(best_model_path, run_dir)`."""
        mdl_ref = Path(mdl_ref)
        if mdl_ref.is_file():
            return mdl_ref, mdl_ref.parent
        if mdl_ref.is_dir():
            best_models = sorted(mdl_ref.glob("best_model.*.pt"))
            if best_models:
                return best_models[-1], mdl_ref
            model_pt = mdl_ref/'model.pt'
            if model_pt.is_file():
                return model_pt, mdl_ref
            checkpoints = sorted(mdl_ref.glob("checkpoint_ep-*.pt"))
            if checkpoints:
                return checkpoints[-1], mdl_ref
            raise FileNotFoundError(f"No model checkpoint found in {mdl_ref}")
        raise FileNotFoundError(mdl_ref)

    def _threshold_dir(run_dir: Path) -> Path:
        """Resolve one shared threshold dir for the active batch threshold."""
        thr = float(kwargs.get('threshold', 0.5))
        return Path(f"th-{int(round(thr*100))}")

    def _resolve_npz_inputs(inputs, base_dir: Path) -> list[Path]:
        """Resolve files, dirs, or masks into one ordered unique NPZ list."""
        resolved = []
        seen = set()
        for item in as_collection(inputs or []):
            item = Path(item)
            if not item.is_absolute():
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
                print(f"[WARN] No NPZ files matched: {item}")
                continue

            for path in matches:
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(path)
        return resolved

    def _run_dataset_test(model_path: Path, test_npz: Path, out_dir: Path, run_video=False):
        """Run one dataset test with clip-only or clip+video post-analysis."""
        raw_tag = get_exporting_name(model_path, test_npz, 'raw', unit='clip')
        thr_dir = _threshold_dir(out_dir)
        res = run_testing(model_path, test_npz, out_dir=out_dir, output_tag=raw_tag, video_mode=True)
        if run_video:
            video_summary = get_exporting_name(model_path, test_npz, 'summary', unit='video')
            analyze_video_test(res['path'], out_path=out_dir, output_name=video_summary,
                               threshold=kwargs.get('threshold', 0.5),
                               threshold_dir=thr_dir, overwrite=True,
                               show_roc=bool(kwargs.get('show_roc', False)),
                               roc_csv=bool(kwargs.get('roc_csv', True)),
                               print=bool(kwargs.get('print_report', False)),
                               )
        else:
            clip_summary = get_exporting_name(model_path, test_npz, 'summary', unit='clip')
            analyze_clip_test(res['path'], out_path=out_dir, output_name=clip_summary,
                              threshold=kwargs.get('threshold', 0.5),
                              threshold_dir=thr_dir, overwrite=True,
                              show_roc=bool(kwargs.get('show_roc', False)),
                              roc_csv=bool(kwargs.get('roc_csv', True)),
                              print=bool(kwargs.get('print_report', False)),
                             )
    def _run_stream_test(model_path: Path, test_npz: Path, out_dir: Path):
        """Run one stream-level test and summary."""
        raw_tag = get_exporting_name(model_path, test_npz, 'raw', unit='stream')
        summary_name = get_exporting_name(model_path, test_npz, 'summary', unit='stream')
        events_name = f"{get_exporting_name(model_path, test_npz, 'events')}.json"
        thr_dir = _threshold_dir(out_dir)
        res = run_testing(model_path, test_npz, out_dir=out_dir, output_tag=raw_tag, video_mode=True)
        analyze_stream_test(res['path'], out_path=out_dir, output_name=summary_name, details_name=events_name,
                            threshold=kwargs.get('threshold', 0.5),
                            threshold_dir=thr_dir, overwrite=True,
                            show_roc=bool(kwargs.get('show_roc', False)),
                            roc_csv=bool(kwargs.get('roc_csv', True)),
                            events_json=bool(kwargs.get('events_json', True)),
                            print=bool(kwargs.get('print_report', False)),
                            )

    tested = []
    for mdl in as_collection(models):
        try:
            best_model, run_dir = _resolve_model_ref(Path(mdl))
        except Exception as exc:
            print(f"[WARN] Bad model ref {mdl}: {type(exc).__name__}: {exc}")
            continue

        cache_group = run_dir.parent.name
        model_tag = run_dir.name.split('_', 1)[1] if re.match(r'^\d{6}-\d{4}_.+', run_dir.name) else run_dir.name
        cache_dir = Path(kwargs.get('cache_dir', Path('data/cache') / cache_group))
        own_test = cache_dir / f"{model_tag}_test.npz"

        dataset_tests = []
        seen = set()
        # Prefer the model's paired own-test cache, then add any extra requested dataset tests once.
        if own_test.is_file():
            dataset_tests.append((own_test, False))
            seen.add(str(own_test.resolve()))
        else:
            print(f"[WARN] Missing own test cache for {run_dir.name}: {own_test}")

        for path in _resolve_npz_inputs(general_tests, cache_dir):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            dataset_tests.append((path, bool(kwargs.get('run_video', False))))

        stream_targets = _resolve_npz_inputs(stream_tests, cache_dir)

        for test_npz, use_video in dataset_tests:
            try:
                _run_dataset_test(best_model, test_npz, run_dir, run_video=use_video)
            except Exception as exc:
                print(f"[WARN] Dataset test failed for {run_dir.name} on {test_npz.name}: {type(exc).__name__}: {exc}")

        if bool(kwargs.get('run_stream', True)):
            for test_npz in stream_targets:
                try:
                    _run_stream_test(best_model, test_npz, run_dir)
                except Exception as exc:
                    print(f"[WARN] Stream test failed for {run_dir.name} on {test_npz.name}: {type(exc).__name__}: {exc}")

        tested.append(run_dir)

    return tested


def run_core_regression_suite(phase='refactor', **kwargs):
    """Run one fixed clip/video regression suite and collect the generated outputs.

    Usage:
    - use the default regression model, or override it with `model_path=...`
    - results are written under `out_root/<phase>/evaluation_core`
    """
    model_path = Path(kwargs.get('model_path', DEFAULT_REG_MODEL))
    out_root = Path(kwargs.get('out_root', "work_dirs/json_models/testing")) / phase / 'evaluation_core'
    show_roc = bool(kwargs.get('show_roc', False))
    save_roc_csv = bool(kwargs.get('roc_csv', True))
    threshold = float(kwargs.get('threshold', 0.5))

    cases =(('train_clip' , Path("data/cache/win-study/J-RWL_25ft_3w-1o5_train.npz"), False),
            ('train_video', Path("data/cache/win-study/J-RWL_25ft_3w-1o5_train.npz"), True),
            ('test_clip'  , Path("data/cache/win-study/J-RWL_25ft_3w-1o5_test.npz"), False),
            ('test_video' , Path("data/cache/win-study/J-RWL_25ft_3w-1o5_test.npz"), True),)

    outputs = {}
    for tag, cache_path, video_mode in cases:
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_tag = f"{tag}_raw"
        res = run_testing(model_path, cache_path, out_dir=out_dir, output_tag=raw_tag, video_mode=video_mode)
        eval_kw = {'out_path': out_dir, 'show_roc': show_roc, 'roc_csv': save_roc_csv, 'print': False, 'threshold': threshold}
        report = analyze_video_test(res['path'], **eval_kw) if video_mode else analyze_clip_test(res['path'], **eval_kw)
        outputs[tag] = {'raw_results': res['path'], 'summary': report}
    return outputs


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

        run_name = re.compile(rf"^\d{{6}}-\d{{4}}_{re.escape(tag)}$")
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

        own_test = cache_dir/f"{train_tag}_test.npz"
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
    """Build one window/stride cache study batch.

    Usage:
    - uses the hard-coded `WINDOW_SETTINGS`
    - builds per-dataset train/test caches, then merges them into one joint cache per setting

    Small batch script for the window/stride cache study.
    Builds train/test caches for  RWF-2000 and  RLVS datasets
    using the existing split files in each dataset directory, then merges the
    matching train/test caches into a joint dataset per window/stride option.
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

    # def _build_one(ds_name: str, ds_dir: Path) -> Path:
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
            merge_cache_npz(built, STUDY_CACHE_DIR / _cache_name(JOINT_DS))


def sum_all_results(work_dir:str|Path, **kwargs): #107
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
                                    
    def _model_disp(model_path:     str) -> tuple[str, str]:
        """Return compact model label and best-epoch string for printing/sorting."""
        mdl_path = Path(model_path)
        parent_name = mdl_path.parent.name
        if re.match(r'^\d{6}-\d{4}_.+', parent_name):
            model_name = parent_name.split("_", 1)[1]
        else:
            model_name = parent_name
        best_epoch = mdl_path.stem.split(".")[-1] if "." in mdl_path.stem else ""
        return model_name, best_epoch

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
            return float(val) if val not in (None, '') else float('inf')

        if  flag == 'model':
            return (_model_disp(row['model'])[0],)
        elif flag == 'win-str':
            return _num_or_inf(row['window']), _num_or_inf(row['stride'])
        elif flag == 'trn-tst':
            return row['train ds'], row['test ds']
        elif flag == 'auc':
            auc = row['AUC']
            return 1 if auc is None else 0, -(auc if auc is not None else 0.0)
        elif flag in {'clp-vid', 'clip-vid', 'vid-clp'}:
            return ({'clip': 0, 'video': 1}.get(row['unit'], 99),)
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
        train_tag = model_path.parent.name.split('_', 1)[1] if '_' in model_path.parent.name else model_path.parent.name
        test_tag = test_cache.stem[:-5] if test_cache.stem.endswith('_test') else test_cache.stem
        train_ds, window, stride = _parse_ds_tag(train_tag)
        test_ds, _, _ = _parse_ds_tag(test_tag)
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
                'window': window, 'stride': stride, 'threshold': threshold,
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

        cols = ['model', 'BE', 'train ds', 'test ds', 'window', 'stride', 'threshold', 'unit',
                'samples', 'support', 'FF', 'FT', 'TF', 'TT', 'Acc', 'Rec', 'FPR', 'AUC']
        header_labels = {c: c if c.isupper() else c.title() for c in cols}
        header_labels.update({'train ds': 'Train ds', 'test ds': 'Test ds'})
        display_rows = []
        for row in rows:
            disp = row.copy()
            disp['model'], disp['BE'] = _model_disp(disp['model'])
            disp['train ds'] = _clip_text(disp['train ds'])
            disp['test ds'] = _clip_text(disp['test ds'])
            for key in ('Acc', 'Rec', 'FPR', 'AUC'):
                if isinstance(disp[key], float):
                    disp[key] = f'{disp[key]:.4f}'
            if isinstance(disp['threshold'], float):
                disp['threshold'] = f"{disp['threshold']:.2f}"
            elif disp['threshold'] in (None, ''):
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
                if col in {'window', 'stride', 'threshold', 'BE'}:
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

    output_path = work_dir/kwargs.get('op_name', RESULT_NAME)
    with (output_path.with_suffix('.pkl')).open('wb') as f:
        pickle.dump(table, f)

    if kwargs.get('save_json', False):
        with (output_path.with_suffix('.json')).open('w') as f:
            json.dump(table, f, indent=2)

    if kwargs.get('print_cli', True):
        _print_rows(table)

    return table


def run_stream_json_dual(data_dir, output_dir,tag=None, **kwargs):
    """Run two stream-JSON conversions for one video dir: plain and `group 0`."""
    # from video_to_json_bb_keypoints_folder import process_video
    from video_to_stream_data import process_video
    data_dir, output_dir = Path(data_dir),  Path(output_dir)

    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # t_str = datetime.now().strftime("%y%m%d")
    t_str =  tag if tag is not  None else datetime.now().strftime("%y%m%d") #    "260312"
    dir_none = output_dir/(t_str + '_g-na')
    dir_zero = output_dir/(t_str + '_g-0')
    common_kwargs = {'sample_rate': kwargs.get('sample_rate', 5),
                     'conf_thresh': kwargs.get('conf_thresh', 0.5),
                     'model_path': kwargs.get('model_path', None),
                     'zip_output': False}
    process_video(data_dir, output_path=dir_none, **common_kwargs)
    process_video(data_dir, output_path=dir_zero, default_grp_tag=[0], **common_kwargs)


def gen_tst():
    """Run the hard-coded local end-to-end test flow used during development."""
    cache_dir = "data/cache/w30-15_um"
    output_dir= "work_dirs/json_models/w30-15-um"
    stream_testing = ["cam-6-11-5_ft25_w30-15.npz",
                      "cam-6-11-8_FRes_Ana_ft25_w30-15.npz",
                      "cam-6-11-8_FRes_Erz_ft25_w30-15.npz"]
    ds_testsing = ['J-All_ft25_w30-15_test.npz']
    train_models(cache_dir, output_dir, ds_tests=ds_testsing, stm_tests=stream_testing)
    sum_all_results(output_dir)

#733(,20,4)-> 755(,23,6)
if __name__ == "__main__":
    pass
    study_dir = 'win-study-tst'
    # study_dir = 'ftr-study'
    STUDY_CACHE_DIR = MAIN_CACHE_DIR/study_dir

    #* train & test for win study
    # build_window_study()
    # train_test_stdy(STUDY_CACHE_DIR)
    # sum_all_results(MAIN_WORK_DIR/study_dir, sort=['win-str','vid-clp', 'trn-tst-R'],save_json=True)

    # cache_dir = "data/cache/w30-15_um"
    # output_dir= "work_dirs/json_models/w30-15-um"
    # stream_testing = ["cam-6-11-5_ft25_w30-15.npz",
    #                   "cam-6-11-8_FRes_Ana_ft25_w30-15.npz",
    #                   "cam-6-11-8_FRes_Erz_ft25_w30-15.npz"]
    # ds_testsing = ['J-All_ft25_w30-15_test.npz']
    # train_models(cache_dir, output_dir, ds_tests=ds_testsing, stm_tests=stream_testing)
    # sum_all_results(output_dir)
    # gen_tst()

    d_d = "/mnt/local-data/Projects/Wesmart/Video-datasets/draft_set/tst_conv"
    #op_d = "data/json_files/tst_conv/test_260611_batch"
    op_d = "data/sanity-testing/json/"
    run_stream_json_dual(d_d, op_d, '260611-no_imgsz'  )
    run_stream_json_dual(d_d, op_d, '260312' )


# 321(,6,2)->300(,6,2)   (23,6)
