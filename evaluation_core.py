import json
from pathlib import Path
import numpy as np
#* project imports
from common.my_local_utils import get_unique_name, print_color, serialize_json_data, resolve_output_path
from project_utils import get_exporting_name, get_test_title_lines
from visual_util import plot_roc_curve

DEFAULT_ROC_RES = 100
DEFAULT_MIN_CLIPS = 2
DEFAULT_EVAL_THRESHOLD = 0.5
DEFAULT_THRESHOLD_RANGE = (0.0, 1.0, 0.01)

PRINT_POLICY = 'all'
# Shared evaluation print policy.
# all     -> print both low-level save messages and the grouped evaluation summary
# summary -> print only the grouped evaluation summary
# save    -> print only low-level save messages
# none    -> suppress evaluation-related printing
# Summary printing is handled here in `evaluation_core`; save-time printing is
# honored by lower-level plotting/export helpers such as those in `visual_util`.

#* region Raw input / validation   --------------------------------------
# -----------------------------------------------------------------------
def load_raw_results_npz(npz_path: str|Path) -> dict:
    """ Load a raw test-results NPZ file into a normalized dictionary."""
    def _scalar_or_none(arr):
        """Convert scalar-like numpy values to plain Python scalars."""
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            if arr.shape == ():
                return arr.item()
            if arr.size == 1:
                return arr.reshape(()).item()
        return arr

    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    raw = {'model_path': _scalar_or_none(data['model_path']) if 'model_path' in data.files else None,
           'test_cache': _scalar_or_none(data['test_cache']) if 'test_cache' in data.files else None,
           'y_true': data['y_true'].astype(np.int64),
           'y_prob': data['y_prob'].astype(np.float32) if 'y_prob' in data.files else None,
           }
    if 'y_pred' in data.files:
        raw['y_pred'] = data['y_pred'].astype(np.int64)
    if 'meta_video' in data.files:
        raw['meta_video'] = data['meta_video']
    if 'meta_t_start' in data.files:
        raw['meta_t_start'] = data['meta_t_start']
    if 'meta_t_end' in data.files:
        raw['meta_t_end'] = data['meta_t_end']
    if 'meta_n_frames' in data.files:
        raw['meta_n_frames'] = data['meta_n_frames']
    return raw


def resolve_input(test_results):
    """ Normalize public analyzer input into raw results dict and optional source path."""
    if isinstance(test_results, (str, Path)):
        src_path = Path(test_results)
        raw = load_raw_results_npz(src_path)
    elif isinstance(test_results, dict):
        src_path = test_results.get('path', None)
        if isinstance(src_path, str):
            src_path = Path(src_path)
        raw = test_results
    else:
        raise TypeError(f"Can't load test npz file:{test_results}")
    return raw, src_path


def validate_raw_results(raw: dict) -> np.ndarray:
    """ Validate raw payload and return `y_true` as a 1D int64 array."""
    if not isinstance(raw, dict):
        raise TypeError("raw results must be a dict")
    if 'y_true' not in raw:
        raise KeyError("raw results missing required key: y_true")

    y_true = np.asarray(raw['y_true'], dtype=np.int64)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be a 1D array, got {y_true.shape}")
    return y_true


def get_eval_arrays(raw: dict, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """ Return y_true, derived y_pred, scoring array, and score type."""
    y_true = validate_raw_results(raw)
    y_prob = raw.get('y_prob', None)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.ndim != 1 or len(y_prob) != len(y_true):
            raise ValueError(f"y_prob length mismatch: {y_prob.shape} vs {len(y_true)}")
        y_pred = apply_threshold(y_prob, threshold)
        return y_true, y_pred, y_prob, 'y_prob'

    #* Backward compatibility for older raw NPZ files.
    y_pred = raw.get('y_pred', None)
    if y_pred is None:
        raise KeyError("raw results missing y_prob (or legacy y_pred)")
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_pred.ndim != 1 or len(y_pred) != len(y_true):
        raise ValueError(f"y_pred length mismatch: {y_pred.shape} vs {len(y_true)}")
    return y_true, y_pred, np.asarray(y_pred, dtype=np.float64), 'y_pred'


def require_prob_scores(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return y_true and y_prob for threshold sweeps, rejecting legacy y_pred-only payloads."""
    y_true = validate_raw_results(raw)
    y_prob = raw.get('y_prob', None)
    if y_prob is None:
        raise KeyError("threshold optimization requires raw results with y_prob")

    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 1 or len(y_prob) != len(y_true):
        raise ValueError(f"y_prob length mismatch: {y_prob.shape} vs {len(y_true)}")
    return y_true, y_prob

# endregion

#* region Summary / output helpers --------------------------------------
# -----------------------------------------------------------------------
def resolve_unique_output_file(path: Path | str, overwrite=False) -> Path:
    """Return one writable file path, enumerating only the filename when needed."""
    path = Path(path)
    if overwrite or not path.exists():
        return path
    return get_unique_name(path)


def save_analyze_summary(summary, out_path:Path|str, overwrite=False, **kwargs):
    def _rel_to_output_dir(path_value):
        """Return one display path relative to the main output dir when possible."""
        if path_value in {None, 'N/A'}:
            return path_value
        path_obj = Path(path_value)
        if output_dir_path is None:
            return str(path_obj)
        try:
            return str(path_obj.relative_to(output_dir_path))
        except ValueError:
            return path_obj.name if path_obj.parent == output_dir_path else str(path_obj)

    def _print_evaluation_results():
        """Print one compact evaluation outputs section."""
        model_dir = Path(summary.get('model_path', '')).parent
        model_tag, test_tag = get_test_title_lines(summary.get('model_path', None),
                                                   summary.get('test_cache', None))
        analysis_type = summary.get('analysis_mode', None) or 'N/A'

        print(f"\n==== Evaluation for {model_tag} ===")
        print(f"== Test data: {test_tag}")
        print(f"== Test type: {analysis_type}")
        if print_policy == 'save':
            print(f"== Output dir: {output_dir}")
        roc_plot = summary.get('roc_plot', None)
        roc_csv_rel = save_summary.get('roc_csv', None)
        if roc_plot not in {None, 'N/A'}:
            print_color(f"\tROC plot image : {_rel_to_output_dir(roc_plot)}", 'b')
        if roc_csv_rel not in {None, 'N/A'}:
            print_color(f"\tROC table      : {_rel_to_output_dir(roc_csv_rel)}", 'b')
        # print("\tAnalysis complete")
        # print_color(f"  Summary json   : {_rel_to_output_dir(out_path)}", 'b')
        print_color(f"\tSummary json   : {out_path.name}", 'b')
        timeline_csvs_local = summary.get('timeline_csvs', None) or []
        for timeline_csv in timeline_csvs_local:
            print_color(f"\tTimeline CSV   : {_rel_to_output_dir(timeline_csv)}", 'b')
        timeline_plots_local = summary.get('timeline_plots', None) or []
        for timeline_plot in timeline_plots_local:
            print_color(f"\tTimeline plot  : {_rel_to_output_dir(timeline_plot)}", 'b')

    """ Write one normalized summary JSON to out_path."""
    # TODO: Consider generalizing this into a common JSON report writer.
    print_policy = str(kwargs.get('print_policy', PRINT_POLICY)).strip().lower()

    out_path = Path(out_path)
    if not out_path.parent.is_dir():
        print(f"[WARN] Bad out_path; {out_path.name} was not saved")
        return out_path
    out_path = resolve_unique_output_file(out_path, overwrite=overwrite)

    analysis_mode = summary.get('analysis_mode', None)
    testing_set = {'test_cache': summary.get('test_cache', None)}
    if summary.get('num_clips', None) is not None:
        testing_set['clips_num'] = summary.get('num_clips', None)
    if summary.get('support_clips', None) is not None:
        testing_set['clips_support'] = summary.get('support_clips', None)
    if analysis_mode in {'video', 'stream'} and summary.get('num_videos', None) is not None:
        testing_set['videos_num'] = summary.get('num_videos', None)
    if analysis_mode == 'video' and summary.get('support_video', None) is not None:
        testing_set['videos_support'] = summary.get('support_video', None)

    output_dir = summary.get('output_dir', None)
    if output_dir is None:
        output_dir = str(Path(out_path).parent)
    output_dir_path = Path(output_dir) if output_dir is not None else None
    events_info = summary.get('events_info', None)
    timeline_csvs = summary.get('timeline_csvs', None)
    roc_csv = summary.get('roc_csv', None)

    save_summary = {'raw_results_path': summary.get('raw_results_path', None),
                    'model': summary.get('model_path', ''),
                    'output_dir': output_dir,
                    'threshold_dir': summary.get('threshold_dir', None),
                    'analysis_config': serialize_json_data(summary.get('analysis_config', None)),
                    'events_info': Path(events_info).name if events_info not in {None, 'N/A'} else events_info,
                    'timeline_csvs': serialize_json_data(timeline_csvs),
                    'analysis_mode': analysis_mode,
                    'testing_set': testing_set,
                    'accuracy': summary.get('accuracy', None),
                    'recall': summary.get('recall', None),
                    'FPR': summary.get('FPR', None),
                    'ROC AUC': summary.get('roc_auc', None),
                    'roc_type': summary.get('raw_results_type', None),
                    'roc_csv': Path(roc_csv).name if roc_csv not in {None, 'N/A'} else roc_csv,
                    }

    if analysis_mode != 'stream':
        save_summary['confusion_matrix'] = summary.get('confusion_matrix', None)

    if analysis_mode == 'stream':
        extra_keys = ('cm_clips', 'false_positive_time', 'miss_time', 'gt_events_num',
                      'pred_events_num', 'detected_events', 'missed_events',
                      'false_events_num', 'mean_onset_delay', 'mean_offset_err',
                      'match_config', 'detection_lag',)
        for key in extra_keys:
            if key in summary:
                save_summary[key] = serialize_json_data(summary.get(key))

    with out_path.open('w') as f:
        json.dump(serialize_json_data(save_summary), f, indent=2)
    if print_policy in {'all', 'summary'}:
        _print_evaluation_results()
    return out_path


def _roc_csv_name(roc, roc_base, save_roc_csv=True):
    """ Return the saved ROC CSV filename or its disabled marker."""
    if save_roc_csv in {False, None}:
        return 'N/A'
    if roc is None or roc_base is None:
        return None
    return roc_base.with_suffix('.csv').name


def resolve_threshold_dir(base_dir, threshold: float, overwrite=False, threshold_dir=None) -> Path:
    """Resolve one stable threshold-scoped output dir."""
    base_dir = Path(base_dir)
    if threshold_dir is not None:
        thr_dir = Path(threshold_dir)
        if not thr_dir.is_absolute():
            # If the caller already passed a repo-relative path like
            # `work_dirs/.../th-50`, do not prepend `base_dir` again.
            if base_dir.parts and len(thr_dir.parts) >= len(base_dir.parts) and tuple(thr_dir.parts[:len(base_dir.parts)]) == base_dir.parts:
                thr_dir = Path.cwd() / thr_dir
            else:
                thr_dir = base_dir / thr_dir
        thr_dir = thr_dir.resolve()
        thr_dir.mkdir(parents=True, exist_ok=True)
        return thr_dir

    thr_value = int(round(float(threshold) * 100.0))
    thr_dir = base_dir / f"th-{thr_value}"
    thr_dir.mkdir(parents=True, exist_ok=True)
    return thr_dir


def companion_summary_name(output_name: str, mode='clip')-> str:
    """ Build a related summary filename for clip/video companion outputs."""
    for suffix in ('_clip-summary', '_video-summary', '_stream-summary'):
        if output_name.endswith(suffix):
            return f"{output_name[:-len(suffix)]}_{mode}-summary"
    return f"{output_name}_{mode}-summary"

# endregion

# region Core metrics/ small helpers   ----------------------------------
# -----------------------------------------------------------------------
def support_counts(y_true):
    """ Return class counts as [class_0_count, class_1_count]."""
    y_true = np.asarray(y_true, dtype=np.int64)
    return [int(np.sum(y_true == 0)), int(np.sum(y_true == 1))]


def support_pair(obj) -> tuple[int, int]|None:
    """ Normalize current list and older dict/json support formats."""
    try:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return int(obj.get(0, obj.get('0'))), int(obj.get(1, obj.get('1')))
        if isinstance(obj, (list, tuple, np.ndarray)):
            return (int(obj[0]), int(obj[1])) if len(obj) == 2 else None
        return None
    except (TypeError, ValueError):
        return None


def cm_dict(cm) -> dict[str, int] | None:
    """ Convert a 2x2 confusion matrix to a named dict."""
    if cm is None or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        return None
    return {'tn': int(cm[0][0]), 'fp': int(cm[0][1]), 'fn': int(cm[1][0]), 'tp': int(cm[1][1])}


def _balanced_accuracy(metrics: dict) -> float:
    """ Return balanced accuracy from the standard binary metric dict."""
    return 0.5*(float(metrics.get('recall', 0.0)) + (1.0 - float(metrics.get('FPR', 0.0))))


def _f1_from_cm(cm) -> float:
    """Return F1 score from a 2x2 confusion matrix."""
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2.0*precision*recall/(precision + recall) if (precision + recall) > 0 else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """ Compute the basic binary metrics and 2x2 confusion matrix."""
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    n = len(y_true)
    return {'confusion_matrix': [[tn, fp], [fn, tp]],
             'accuracy': (tn + tp)/n  if n > 0 else 0.0,
             'recall':   tp/(tp + fn) if (tp + fn) > 0 else 0.0,
             'FPR':      fp/(fp + tn) if (fp + tn) > 0 else 0.0,}


def apply_threshold(probability_vector, threshold: float) -> np.ndarray:
    """ Convert one probability vector into hard 0/1 clip/window predictions."""
    probs = np.asarray(probability_vector, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probability_vector must be 1D, got {probs.shape}")
    return (probs >= float(threshold)).astype(np.int64)

# endregion

#* region ROC/ score helpers    -----------------------------------------
# -----------------------------------------------------------------------
def roc_summary(y_true, scores, score_name, max_resolution=DEFAULT_ROC_RES):
    """ Return ROC curve data plus the summary fields derived from it."""
    try:
        roc = roc_from_scores(y_true, scores, max_resolution=max_resolution)
        return roc, {'roc_auc': roc['auc']}
    except Exception as e:
        return None, {'roc_auc': None, 'roc_error': str(e)}


def roc_from_scores(y_true: np.ndarray, scores: np.ndarray, max_resolution=DEFAULT_ROC_RES):
    """Compute ROC arrays and AUC from binary labels and scores."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and scores must be 1D arrays")
    if len(y_true) != len(scores):
        raise ValueError(f"y_true/scores length mismatch: {len(y_true)} vs {len(scores)}")

    uniq = np.unique(y_true)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"y_true must be binary 0/1, got classes: {uniq.tolist()}")

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        raise ValueError("ROC requires both positive and negative samples")

    order = np.argsort(scores, kind='mergesort')[::-1]
    y = y_true[order]
    s = scores[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    distinct_idx = np.where(np.diff(s))[0]
    thr_idx = np.r_[distinct_idx, len(s) - 1]

    tpr = np.r_[0.0, tps[thr_idx] / pos, 1.0]
    fpr = np.r_[0.0, fps[thr_idx] / neg, 1.0]
    thresholds = np.r_[np.inf, s[thr_idx], -np.inf]

    if max_resolution not in (None, False, 0):
        n_target = max_resolution
        if n_target < 2:
            raise ValueError(f"max_resolution must be >=2 or None/False/0, got {max_resolution}")
        if len(fpr) > n_target:
            uniq_fpr, uniq_idx = np.unique(fpr, return_index=True)
            uniq_tpr = tpr[uniq_idx]
            uniq_thr = thresholds[uniq_idx]

            thr_interp = uniq_thr.astype(np.float64).copy()
            finite = np.isfinite(thr_interp)
            if finite.any():
                first_finite = thr_interp[finite][0]
                last_finite = thr_interp[finite][-1]
                thr_interp[np.isposinf(thr_interp)] = first_finite
                thr_interp[np.isneginf(thr_interp)] = last_finite
            else:
                thr_interp[:] = 0.0

            fpr_grid = np.linspace(uniq_fpr[0], uniq_fpr[-1], n_target)
            tpr = np.interp(fpr_grid, uniq_fpr, uniq_tpr)
            thresholds = np.interp(fpr_grid, uniq_fpr, thr_interp)
            thresholds[0] = np.inf
            thresholds[-1] = -np.inf
            fpr = fpr_grid

    auc = np.trapz(tpr, fpr)
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc}

# endregion

#* region Aggregation/ optimization -------------------------------------
# -----------------------------------------------------------------------
def min_clips_pool(clip_pred, clip_score=None, min_val=DEFAULT_MIN_CLIPS):
    """ Pool clip predictions by a minimum count or fraction of positive clips."""
    score = float(np.sum(np.asarray(clip_pred, dtype=np.int64) == 1))
    if isinstance(min_val, int):
        target = min_val
    elif isinstance(min_val, float):
        target = round(len(clip_pred) * min_val)
    else:
        raise TypeError(f"min_val must be int or float, got {type(min_val).__name__}")
    return int(score >= target), score


def iter_thresholds(threshold_range=DEFAULT_THRESHOLD_RANGE) -> np.ndarray:
    """ Expand one fixed-grid threshold range into a 1D float array."""
    if threshold_range is None:
        threshold_range = DEFAULT_THRESHOLD_RANGE

    if isinstance(threshold_range, np.ndarray):
        thresholds = np.asarray(threshold_range, dtype=np.float64).reshape(-1)
    elif isinstance(threshold_range, (list, tuple)) and len(threshold_range) == 3:
        start, stop, step = [float(v) for v in threshold_range]
        if step <= 0:
            raise ValueError(f"threshold step must be positive, got {step}")
        thresholds = np.arange(start, stop + (step * 0.5), step, dtype=np.float64)
    else:
        thresholds = np.asarray(list(threshold_range), dtype=np.float64).reshape(-1)

    if thresholds.size == 0:
        raise ValueError("threshold_range produced no thresholds")
    return np.unique(np.round(thresholds, 10))


def clip_metrics_for_threshold(y_true, y_prob, threshold: float) -> dict:
    """Evaluate one clip threshold without saving side effects."""
    y_pred = apply_threshold(y_prob, threshold)
    metrics = binary_metrics(y_true, y_pred)
    metrics.update({'threshold': float(threshold),
                    'f1': _f1_from_cm(metrics['confusion_matrix']),
                    'balanced_acc': _balanced_accuracy(metrics),
                    })
    return metrics


def video_level_arrays(raw: dict, threshold: float, **kwargs):
    """ Pool clip predictions to video-level arrays using the standard analyzer logic."""
    y_true, y_pred, scores, score_name = get_eval_arrays(raw, threshold=threshold)

    if 'meta_video' not in raw or len(y_true) != len(y_pred):
        raise KeyError("Results file meta data are defective")

    meta_video = np.asarray(raw['meta_video'])
    if len(meta_video) != len(y_true):
        raise ValueError(f"meta_video length mismatch: {len(meta_video)} vs {len(y_true)}")

    clip_scores = scores if score_name == 'y_prob' else None
    pool_func = kwargs.get('pool_func', min_clips_pool)
    min_val = kwargs.get('min_val', DEFAULT_MIN_CLIPS)

    video_to_idx = {}
    for i, vid in enumerate(meta_video):
        video_to_idx.setdefault(str(vid), []).append(i)

    vid_true, vid_pred, vid_score = [], [], []
    excluded_videos = []
    for vid, idx in video_to_idx.items():
        clip_true = y_true[idx]
        uniq_true = np.unique(clip_true)
        if len(uniq_true) != 1:
            excluded_videos.append(vid)
            continue

        clip_pred = y_pred[idx]
        clip_score = clip_scores[idx] if clip_scores is not None else None
        if pool_func is min_clips_pool:
            pooled = pool_func(clip_pred, clip_score, min_val=min_val)
        else:
            pooled = pool_func(clip_pred, clip_score)

        if isinstance(pooled, tuple):
            video_pred, video_score = pooled
        else:
            video_pred = pooled
            _, video_score = min_clips_pool(clip_pred, clip_score, min_val=min_val)

        vid_true.append(int(uniq_true[0]))
        vid_pred.append(int(video_pred))
        vid_score.append(float(video_score))

    if len(vid_true) == 0:
        raise ValueError("No valid videos left for analysis after excluding inconsistent GT videos")

    return (np.asarray(vid_true, dtype=np.int64),
            np.asarray(vid_pred, dtype=np.int64),
            np.asarray(vid_score, dtype=np.float64),
            excluded_videos,)


def objective_from_row(row: dict, mode='balanced_acc', **kwargs) -> tuple[float, tuple]:
    """ Return objective value plus tie-break key for one sweep row."""
    if mode == 'balanced_acc':
        return float(row['balanced_acc']), (float(row['balanced_acc']), float(row['f1']), -float(row['FPR']), float(row['threshold']))
    if mode == 'f1':
        return float(row['f1']), (float(row['f1']), float(row['balanced_acc']), -float(row['FPR']), float(row['threshold']))
    if mode == 'recall_floor':
        target_recall = kwargs.get('target_recall', None)
        if target_recall is None:
            raise ValueError("recall_floor mode requires target_recall")
        valid = float(row['recall']) >= float(target_recall)
        objective = 1.0 if valid else 0.0
        tie_key = (objective, -float(row['FPR']), float(row['balanced_acc']), float(row['threshold']))
        return objective, tie_key
    raise ValueError(f"Unsupported optimization mode: {mode}")


def finalize_threshold_sweep(test_results, analysis_mode, thresholds, rows, mode, **kwargs) -> dict:
    """Choose the best sweep row and normalize one optimization result payload."""
    raw, res_path = resolve_input(test_results)
    best_row = None
    best_key = None
    for row in rows:
        objective, tie_key = objective_from_row(row, mode=mode, **kwargs)
        row['objective'] = float(objective)
        if best_key is None or tie_key > best_key:
            best_key, best_row = tie_key, row

    return {'raw_results_path': str(res_path) if res_path is not None else None,
            'analysis_mode': analysis_mode,
            'optimization_mode': mode,
            'threshold_range': [float(v) for v in thresholds],
            'model_path': raw.get('model_path', None),
            'test_cache': raw.get('test_cache', None),
            'best_threshold': best_row['threshold'],
            'best_objective': best_row['objective'],
            'best_metrics': {k: serialize_json_data(v) for k, v in best_row.items() if k != 'objective'},
            'results_table': serialize_json_data(rows),
            }

# endregion

#* region Public API   --------------------------------------------------
# -----------------------------------------------------------------------
def analyze_clip_scores(test_results: Path | str | dict, **kwargs):
    """ Analyze clip/window probability scores only and save clip ROC outputs."""
    raw_res, results_path = resolve_input(test_results)
    y_true = validate_raw_results(raw_res)
    y_prob = raw_res.get('y_prob', None)
    score_name = 'y_prob'
    scores = y_prob
    if y_prob is None:
        y_pred = raw_res.get('y_pred', None)
        if y_pred is None:
            raise KeyError("raw results missing y_prob (or legacy y_pred)")
        scores = np.asarray(y_pred, dtype=np.float64)
        score_name = 'y_pred'
    else:
        scores = np.asarray(y_prob, dtype=np.float64)

    summary = {'raw_results_path': str(results_path) if results_path is not None else None,
               'raw_results_type': score_name,
               'analysis_mode': 'clip_scores',
               'model_path': raw_res.get('model_path', None),
               'test_cache': raw_res.get('test_cache', None),
               'num_clips': len(y_true),
               'support_clips': support_counts(y_true),
               }
    roc, roc_info = roc_summary(y_true, scores, score_name, max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES))
    summary.update(roc_info)

    tst_name = results_path.stem if results_path is not None else 'clip_test'
    out_name = kwargs.get('output_name', f"{tst_name}_clip-scores")
    out_base = resolve_output_path(results_path, out_name, kwargs.get('out_path', None))
    roc_mode_code = kwargs.get('roc_mode_code', 'clp')
    roc_base = (out_base.parent / get_exporting_name(raw_res.get('model_path', None), raw_res.get('test_cache', None),
               'roc', unit=roc_mode_code, short=True) if out_base is not None else None)
    summary['output_dir'] = str(out_base.parent) if out_base is not None else None

    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_plot'] = roc_base.with_suffix('.png').name if roc_base is not None else None
    summary['roc_csv'] = _roc_csv_name(roc, roc_base, save_roc_csv)
    if kwargs.get('save_roc', True) and roc is not None and (roc_base is not None or bool(kwargs.get('show_roc', False))):
        plot_roc_curve(roc, save_to=roc_base, save_csv=bool(save_roc_csv), show=bool(kwargs.get('show_roc', False)),
                       print_policy=kwargs.get('print_policy', PRINT_POLICY),
                       title=kwargs.get('roc_title', "\n".join(get_test_title_lines(raw_res.get('model_path', None),
                                                                                    raw_res.get('test_cache', None)))), )
    return summary


def analyze_clip_predictions(test_results:Path|str|dict, **kwargs): #552
    """Analyze thresholded clip/window predictions only and save clip summary output."""
    raw_res, results_path = resolve_input(test_results)
    threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))
    y_true, y_pred, _, score_name = get_eval_arrays(raw_res, threshold=threshold)
    summary = binary_metrics(y_true, y_pred)
    summary.update({'raw_results_path': str(results_path) if results_path is not None else None,
                    'raw_results_type': score_name,
                    'analysis_mode': 'clip',
                    'model_path': raw_res.get('model_path', None),
                    'test_cache': raw_res.get('test_cache', None),
                    'num_clips': len(y_true),
                    'support_clips': support_counts(y_true),
                    'analysis_config': {'threshold': threshold},
                    })
    out_name = kwargs.get('output_name', get_exporting_name(raw_res.get('model_path', None),
                                                            raw_res.get('test_cache', None),
                                                                  'summary', unit='clip'))
    out_base = resolve_output_path(results_path, out_name, kwargs.get('out_path', None))
    roc_mode_code = kwargs.get('roc_mode_code', 'clp')
    roc_base = (out_base.parent / get_exporting_name(raw_res.get('model_path', None),
                                                     raw_res.get('test_cache', None),
                                                         'roc', unit=roc_mode_code, short=True)
                if out_base is not None else None
                )
    thr_dir = (resolve_threshold_dir(out_base.parent, threshold,
                                     overwrite=bool(kwargs.get('overwrite', False)),
                                     threshold_dir=kwargs.get('threshold_dir', None))
                if out_base is not None else None
               )
    summary['output_dir'] = str(out_base.parent) if out_base is not None else None
    summary['threshold_dir'] = thr_dir.name if thr_dir is not None else None
    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_plot'] = roc_base.with_suffix('.png').name if roc_base is not None else None
    summary['roc_csv'] = _roc_csv_name(True, roc_base, save_roc_csv)

    if thr_dir is not None:
        save_analyze_summary(summary,
                              thr_dir / out_base.with_suffix('.json').name,
                              overwrite=bool(kwargs.get('overwrite', False)),
                              print_policy=kwargs.get('print_policy', PRINT_POLICY))
    else:
        print("[INFO] Analysis complete\n Summary file wasn't saved; (please provide out_path)")
    if kwargs.get('print', True):
        from evaluation_cli import print_test_report
        print_test_report(summary)
    return summary


def analyze_clip_test(test_results: Path | str | dict, **kwargs):
    """ Run clip score analysis and threshold-dependent clip prediction analysis."""
    raw_res, results_path = resolve_input(test_results)
    threshold_dir = kwargs.get('threshold_dir', None)
    if threshold_dir is None and results_path is not None:
        out_name = kwargs.get('output_name',
                              get_exporting_name(raw_res.get('model_path', None),
                                                 raw_res.get('test_cache', None),
                                                       'summary', unit='clip'))
        out_base = resolve_output_path(results_path, out_name, kwargs.get('out_path', None))
        if out_base is not None:
            threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))
            threshold_dir = resolve_threshold_dir(out_base.parent, threshold,
                                                 overwrite=bool(kwargs.get('overwrite', False)))
    score_kwargs = dict(kwargs)
    score_kwargs['print'] = False
    analyze_clip_scores(test_results, **score_kwargs)
    pred_kwargs = dict(kwargs)
    if threshold_dir is not None:
        pred_kwargs['threshold_dir'] = threshold_dir
    return analyze_clip_predictions(test_results, **pred_kwargs)


def analyze_video_scores(test_res: Path | str | dict, **kwargs):
    """Analyze video-level pooled scores only and save video ROC outputs."""
    raw_res, res_path = resolve_input(test_res)
    threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))
    y_true, _, _, score_name = get_eval_arrays(raw_res, threshold=threshold)
    video_kwargs = dict(kwargs)
    video_kwargs.pop('threshold', None)
    vid_true, vid_pred, vid_score, excluded_videos = video_level_arrays(raw_res, threshold=threshold, **video_kwargs)

    summary = {'raw_results_path': str(res_path) if res_path is not None else None,
               'raw_results_type': score_name if score_name == 'y_prob' else 'video_score',
               'analysis_mode': 'video_scores',
               'model_path': raw_res.get('model_path', None),
               'test_cache': raw_res.get('test_cache', None),
               'num_clips': len(y_true),
               'support_clips': support_counts(y_true),
               'num_videos': len(vid_true),
               'support_video': support_counts(vid_true),
               'num_videos_excluded': len(excluded_videos),
               'excluded_videos': excluded_videos,
               'analysis_config': {'threshold': threshold,
                                   'pool_func': getattr(kwargs.get('pool_func', min_clips_pool),
                                                        '__name__', str(kwargs.get('pool_func', min_clips_pool))),
                                   'min_val': kwargs.get('min_val', DEFAULT_MIN_CLIPS), },
               }
    roc, roc_info = roc_summary(vid_true, vid_score, 'video_score', max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES))
    summary.update(roc_info)
    for vid in excluded_videos:
        print_color(f"[WARN] Inconsistent GT in video {vid}; excluded from video test", 'o')

    tst_name = res_path.stem if res_path is not None else 'video_test'
    out_name = kwargs.get('output_name', f"{tst_name}_video-scores")
    out_base = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
    roc_base = (out_base.parent / get_exporting_name(raw_res.get('model_path', None),
                                                     raw_res.get('test_cache', None), 'roc', unit='vid', short=True)
                                                         if out_base is not None else None)
    summary['output_dir'] = str(out_base.parent) if out_base is not None else None
    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_plot'] = roc_base.with_suffix('.png').name if roc_base is not None else None
    summary['roc_csv'] = _roc_csv_name(roc, roc_base, save_roc_csv)

    if kwargs.get('save_roc', True) and roc is not None and (roc_base is not None or bool(kwargs.get('show_roc', False))):
        plot_roc_curve(roc, save_to=roc_base, save_csv=bool(save_roc_csv), show=bool(kwargs.get('show_roc', False)),
                       print_policy=kwargs.get('print_policy', PRINT_POLICY),
                       title=kwargs.get('roc_title', "\n".join(get_test_title_lines(raw_res.get('model_path', None),
                                                                                    raw_res.get('test_cache', None)))), )
    return summary


def analyze_video_predictions(test_res: Path | str | dict, **kwargs):
    """ Analyze thresholded video predictions only and save video summary output."""
    raw_res, res_path = resolve_input(test_res)
    threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))
    build_kwargs = dict(kwargs)
    build_kwargs.pop('threshold', None)
    y_true, _, _, score_name = get_eval_arrays(raw_res, threshold=threshold)
    vid_true, vid_pred, _, excluded_videos = video_level_arrays(raw_res, threshold=threshold, **build_kwargs)
    summary = binary_metrics(vid_true, vid_pred)
    summary.update({'raw_results_path': str(res_path) if res_path is not None else None,
                    'raw_results_type': score_name if score_name == 'y_prob' else 'video_score',
                    'analysis_mode': 'video',
                    'model_path': raw_res.get('model_path', None),
                    'test_cache': raw_res.get('test_cache', None),
                    'num_clips': len(y_true),
                    'support_clips': support_counts(y_true),
                    'num_videos': len(vid_true),
                    'support_video': support_counts(vid_true),
                    'num_videos_excluded': len(excluded_videos),
                    'excluded_videos': excluded_videos,
                    'analysis_config': {'threshold': threshold,
                                        'pool_func': getattr(kwargs.get('pool_func', min_clips_pool),
                                                             '__name__', str(kwargs.get('pool_func', min_clips_pool))),
                                        'min_val': kwargs.get('min_val', DEFAULT_MIN_CLIPS),},
                    })
    for vid in excluded_videos:
        print_color(f"[WARN] Inconsistent GT in video {vid}; excluded from video test", 'o')

    out_name = kwargs.get('output_name',
                          get_exporting_name(raw_res.get('model_path', None), raw_res.get('test_cache', None),
                                                   'summary', unit='video'))
    out_base = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
    roc_base = (out_base.parent / get_exporting_name(raw_res.get('model_path', None),
                                                     raw_res.get('test_cache', None), 'roc', unit='vid', short=True)
                                                        if out_base is not None else None)
    thr_dir = (resolve_threshold_dir(out_base.parent, threshold,
                                     overwrite=bool(kwargs.get('overwrite', False)),
                                     threshold_dir=kwargs.get('threshold_dir', None))
                                            if out_base is not None else None)
    summary['output_dir'] = str(out_base.parent) if out_base is not None else None
    summary['threshold_dir'] = thr_dir.name if thr_dir is not None else None
    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_plot'] = roc_base.with_suffix('.png').name if roc_base is not None else None
    summary['roc_csv'] = _roc_csv_name(True, roc_base, save_roc_csv)

    if thr_dir is not None:
        save_analyze_summary(summary,
                              thr_dir / out_base.with_suffix('.json').name,
                              overwrite=bool(kwargs.get('overwrite', False)),
                              print_policy=kwargs.get('print_policy', PRINT_POLICY))
    else:
        print("[WARN] Summary file wasn't saved; invalid or missing out_path")
    if kwargs.get('print', True):
        from evaluation_cli import print_test_report
        print_test_report(summary)
    return summary


def analyze_video_test(test_res: Path | str | dict, **kwargs):
    """ Run clip/video score analysis and threshold-dependent clip/video prediction analysis."""
    _, res_path = resolve_input(test_res)
    tst_name = res_path.stem if res_path is not None else 'video_test'
    out_name = kwargs.get('output_name', f"{tst_name}_video-summary")
    threshold_dir = kwargs.get('threshold_dir', None)
    if threshold_dir is None and res_path is not None:
        out_base = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
        if out_base is not None:
            threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))
            threshold_dir = resolve_threshold_dir(out_base.parent, threshold,
                                                  overwrite=bool(kwargs.get('overwrite', False)))

    score_kwargs = dict(kwargs)
    score_kwargs['print'] = False
    analyze_clip_scores(test_res, **score_kwargs)
    analyze_video_scores(test_res, **score_kwargs)

    clip_kwargs = dict(kwargs)
    clip_kwargs['print'] = False
    clip_kwargs['output_name'] = companion_summary_name(out_name, 'clip')
    if threshold_dir is not None:
        clip_kwargs['threshold_dir'] = threshold_dir
    analyze_clip_predictions(test_res, **clip_kwargs)
    pred_kwargs = dict(kwargs)
    if threshold_dir is not None:
        pred_kwargs['threshold_dir'] = threshold_dir
    return analyze_video_predictions(test_res, **pred_kwargs)


def optimize_clip_threshold(test_results: Path | str | dict, threshold_range=DEFAULT_THRESHOLD_RANGE, mode='balanced_acc', **kwargs):
    """ Sweep clip thresholds and return the best clip-level operating point."""
    raw_res, _ = resolve_input(test_results)
    y_true, y_prob = require_prob_scores(raw_res)
    thresholds = iter_thresholds(threshold_range)

    rows = []
    for threshold in thresholds:
        row = clip_metrics_for_threshold(y_true, y_prob, threshold)
        rows.append(row)
    return finalize_threshold_sweep(test_results, 'clip', thresholds, rows, mode, **kwargs)


def optimize_video_threshold(test_results: Path | str | dict, threshold_range=DEFAULT_THRESHOLD_RANGE, mode='balanced_acc', **kwargs):
    """Sweep clip thresholds and score them after video-level pooling."""
    raw_res, _ = resolve_input(test_results)
    require_prob_scores(raw_res)
    thresholds = iter_thresholds(threshold_range)

    rows = []
    for threshold in thresholds:
        vid_true, vid_pred, vid_score, excluded_videos = video_level_arrays(raw_res, threshold=threshold, **kwargs)
        row = binary_metrics(vid_true, vid_pred)
        row.update({'threshold': float(threshold),
                    'f1': _f1_from_cm(row['confusion_matrix']),
                    'balanced_acc': _balanced_accuracy(row),
                    'num_videos': int(len(vid_true)),
                    'support_video': support_counts(vid_true),
                    'num_videos_excluded': int(len(excluded_videos)),
                    'video_score_mean': float(np.mean(vid_score)) if len(vid_score) > 0 else None,
                    })
        rows.append(row)
    return finalize_threshold_sweep(test_results, 'video', thresholds, rows, mode, **kwargs)

# endregion

#547(1,3,) -> threshold dependency resolution -> 936
# 936-> 865(3,1,1)-># 835(2,1, 1)-> 833-> 822-> pm 748(2,,)
# meta info refactoring 789(2,,)   // 854 
