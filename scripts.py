import json, pickle
import re
from pathlib import Path
#* Imports from local project
from precompute_clips import build_cache_from_json, merge_cache_npz, WINDOW_SEC, STRIDE_SEC
from tms_trainer import run_training, run_testing
from evaluation_tools import analyze_clip_test, analyze_video_test, _support_pair

#* general configuration
RWF_DIR  = Path("data/json_files/RWF-2000/ds")
RLVS_DIR = Path("data/json_files/RLVS/ds")

MAIN_WORK_DIR = Path("work_dirs/json_models")
MAIN_CACHE_DIR = Path("data/cache")
# STUDY_CACHE_DIR  = MAIN_CACHE_DIR/"win-study"

DATASETS = [('RWF', RWF_DIR), ('RLVS', RLVS_DIR)]
JOINT_DS =  'J-RWL'
RESULT_NAME = 'all_results'

def build_window_study(): # 80 -> 65
    """  Small batch script for the window/stride cache study.
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



def train_test_stdy(cache_dir:str|Path, **kwargs): #92 -> 63
    """Train all `*_train.npz` caches in a directory and run clip/video tests."""

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
        """Return the newest matching prior run dir for `tag`, if it is usable."""
        if not base_work_dir.is_dir():
            return None

        run_name = re.compile(rf"^\d{{6}}-\d{{4}}_{re.escape(tag)}$")
        matches = [p for p in base_work_dir.iterdir()
                   if p.is_dir() and run_name.fullmatch(p.name)]
        ready_runs = [p for p in matches if any(p.glob("best_model.*.pt"))]
        if ready_runs:
            return sorted(ready_runs)[-1]
        return None

    def _run_one_test(test_npz:Path, test_mode: str):
        """Run one test job and the matching analysis."""
        test_name = _dataset_tag(test_npz.stem, "_test")
        output_tag = f"{train_tag}_BM{best_epoch}_{test_name}_{test_mode}-tst.npz"
        output_name = f"{train_tag}_BM{best_epoch}_{test_name}_{test_mode}-summary.json"

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

        test_targets = []
        for p in (own_test, joint_test):
            if p not in test_targets:
                if p.is_file():
                    test_targets.append(p)
                else:
                    print(f"[train_test_stdy] Missing test cache for {train_tag}: {p}")

        if not test_targets:
            print(f"[train_test_stdy] No test caches available for {train_tag}; skipping tests")
            continue

        for test_cache in test_targets:
            for test_mode in ('clip', 'video'):
                try:
                    _run_one_test(test_cache, test_mode)
                except Exception as exc:
                    print(f"[train_test_stdy] Test failed for {train_tag} on {test_cache.name} ({test_mode}): {type(exc).__name__}: {exc}")


def sum_all_results(work_dir: str | Path, **kwargs): #107
    """ Collect all `*-summary.json` files under a study work dir into one table."""

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
        parent_name = mdl_path.parent.name
        parts = parent_name.split("_", 2)
        model_name = parts[2] if len(parts) >= 3 else parent_name
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
            return (_num_or_inf(row['window']), _num_or_inf(row['stride']))
        elif flag == 'trn-tst':
            return (row['train ds'], row['test ds'])
        elif flag == 'auc':
            auc = row['AUC']
            return (1 if auc is None else 0, -(auc if auc is not None else 0.0))
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
        with summary_path.open("r") as f:
            summary = json.load(f)

        testing_set = summary.get('testing_set', {})
        test_cache = Path(testing_set.get('test_cache', ''))
        model_path = Path(summary.get('model', ''))
        train_tag = model_path.parent.name.split('_', 1)[1] if '_' in model_path.parent.name else model_path.parent.name
        test_tag = test_cache.stem[:-5] if test_cache.stem.endswith('_test') else test_cache.stem
        train_ds, window, stride = _parse_ds_tag(train_tag)
        test_ds, _, _ = _parse_ds_tag(test_tag)
        unit = summary.get('analysis_mode', '')

        if unit == 'video':
            samples = testing_set.get('videos_num', None)
            support = _support_pair(testing_set.get('videos_support', None))
        else:
            samples = testing_set.get('clips_num', None)
            support = _support_pair(testing_set.get('clips_support', None))
        support_str = f'{support[0]}/{support[1]}' if support is not None else 'N/A'

        cm = summary.get('confusion_matrix', [[None, None], [None, None]])
        return {'model': str(model_path), 'cache': test_cache.stem,
                'train ds': train_ds, 'test ds': test_ds,
                'unit': unit, 'samples': samples, 'support': support_str,
                'window': window, 'stride': stride,
                'FF': cm[0][0],
                'FT': cm[0][1],
                'TF': cm[1][0],
                'TT': cm[1][1],
                'Acc': summary.get('accuracy', None),
                'Rec': summary.get('recall', None),
                'FPR': summary.get('FPR', None),
                'AUC': summary.get('ROC AUC', None),
                }

    def _print_rows(rows: list[dict]):
        """Print the aggregated table in a simple aligned layout."""
        cols = ['model', 'BE', 'train ds', 'test ds', 'window', 'stride', 'unit',
                'samples', 'support', 'FF', 'FT', 'TF', 'TT', 'Acc', 'Rec', 'FPR', 'AUC']
        header_labels = {c: c if c.isupper() else c.title() for c in cols}
        header_labels.update({'train ds': 'Train ds', 'test ds': 'Test ds'})
        display_rows = []
        for row in rows:
            disp = row.copy()
            disp['model'], disp['BE'] = _model_disp(disp['model'])
            for key in ('Acc', 'Rec', 'FPR', 'AUC'):
                if isinstance(disp[key], float):
                    disp[key] = f'{disp[key]:.4f}'
            display_rows.append(disp)

        widths = {c: len(header_labels[c]) for c in cols}
        for row in display_rows:
            for col in cols:
                widths[col] = max(widths[col], len(str(row.get(col, ''))))

        print(' | '.join(f'{header_labels[col]:<{widths[col]}}' for col in cols))
        print('-+-'.join('-' * widths[col] for col in cols))
        for row in display_rows:
            line = []
            for col in cols:
                val = str(row.get(col, ''))
                if col in {'window', 'stride', 'BE'}:
                    line.append(f'{val:^{widths[col]}}')
                elif col == 'samples':
                    line.append(f'{val:>{widths[col]}}')
                else:
                    line.append(f'{val:<{widths[col]}}')
            print(' | '.join(line))

    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        raise NotADirectoryError(work_dir)

    summary_paths = sorted(work_dir.glob('*/*-summary.json'))
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


if __name__ == "__main__":
    pass
    study_dir = 'win-study'
    # study_dir = 'ftr-study'
    STUDY_CACHE_DIR = MAIN_CACHE_DIR/study_dir

    # build_window_study()
    train_test_stdy(STUDY_CACHE_DIR)
    sum_all_results(MAIN_WORK_DIR/study_dir, sort=['win-str','vid-clp', 'trn-tst-R'],save_json=True)

# 321(,6,2)->300(,6,2)
