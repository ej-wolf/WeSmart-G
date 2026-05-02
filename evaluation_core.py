import json
from pathlib import Path
import numpy as np

from common.my_local_utils import print_color, serialize_json_data, resolve_output_path
from visual_util import plot_roc_curve

DEFAULT_ROC_RES = 100
DEFAULT_MIN_CLIPS = 2
DEFAULT_EVAL_THRESHOLD = 0.5
DEFAULT_REG_MODEL = Path("work_dirs/json_models/win-study/260414-1721_J-RWL_25ft_3w-1o5-stream-tst/best_model.148.pt")


# -----------------------------------------------------------------------
#* IO / raw-results helpers
# -----------------------------------------------------------------------
def _load_raw_results_npz(npz_path: str | Path) -> dict:
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


def _resolve_input(test_results):
    """ Normalize public analyzer input into raw results dict and optional source path."""
    if isinstance(test_results, (str, Path)):
        src_path = Path(test_results)
        raw = _load_raw_results_npz(src_path)
    elif isinstance(test_results, dict):
        src_path = test_results.get('path', None)
        if isinstance(src_path, str):
            src_path = Path(src_path)
        raw = test_results
    else:
        raise TypeError(f"Can't load test npz file:{test_results}")
    return raw, src_path


def _validate_raw_results(raw: dict) -> np.ndarray:
    """Validate raw payload and return `y_true` as a 1D int64 array."""
    if not isinstance(raw, dict):
        raise TypeError("raw results must be a dict")
    if 'y_true' not in raw:
        raise KeyError("raw results missing required key: y_true")

    y_true = np.asarray(raw['y_true'], dtype=np.int64)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be a 1D array, got {y_true.shape}")
    return y_true


def _get_eval_arrays(raw: dict, threshold=DEFAULT_EVAL_THRESHOLD) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return y_true, derived y_pred, scoring array, and score type."""
    y_true = _validate_raw_results(raw)
    if threshold is None:
        threshold = DEFAULT_EVAL_THRESHOLD
    y_prob = raw.get('y_prob', None)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.ndim != 1 or len(y_prob) != len(y_true):
            raise ValueError(f"y_prob length mismatch: {y_prob.shape} vs {len(y_true)}")
        y_pred = (y_prob >= float(threshold)).astype(np.int64)
        return y_true, y_pred, y_prob, 'y_prob'

    # Backward compatibility for older raw NPZ files.
    y_pred = raw.get('y_pred', None)
    if y_pred is None:
        raise KeyError("raw results missing y_prob (or legacy y_pred)")
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_pred.ndim != 1 or len(y_pred) != len(y_true):
        raise ValueError(f"y_pred length mismatch: {y_pred.shape} vs {len(y_true)}")
    return y_true, y_pred, np.asarray(y_pred, dtype=np.float64), 'y_pred'


# -----------------------------------------------------------------------
#* Core metrics / report helpers
# -----------------------------------------------------------------------
def _support_counts(y_true):
    """Return class counts as [class_0_count, class_1_count]."""
    y_true = np.asarray(y_true, dtype=np.int64)
    return [int(np.sum(y_true == 0)), int(np.sum(y_true == 1))]


def _support_pair(obj) -> tuple[int, int] | None:
    """Normalize current list and older dict/json support formats."""
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


def _cm_dict(cm) -> dict[str, int] | None:
    """Convert a 2x2 confusion matrix to a named dict."""
    if cm is None or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        return None
    return {'tn': int(cm[0][0]), 'fp': int(cm[0][1]), 'fn': int(cm[1][0]), 'tp': int(cm[1][1])}


def _save_analyze_summary(summary, out_path: Path | str):
    """Write one normalized summary JSON to `out_path`."""
    # TODO: Consider generalizing this into a common JSON report writer.
    if not Path(out_path).parent.is_dir():
        print(f"[WARN] Bad out_path; {Path(out_path).name} was not saved")
        return out_path

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
    events_info = summary.get('events_info', None)
    timeline_csvs = summary.get('timeline_csvs', None)
    roc_csv = summary.get('roc_csv', None)

    save_summary = {'raw_results_path': summary.get('raw_results_path', None),
                    'model': summary.get('model_path', ''),
                    'output_dir': output_dir,
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

    with Path(out_path).open('w') as f:
        json.dump(serialize_json_data(save_summary), f, indent=2)
    print("Analysis complete")
    print_color(f"  Summary saved to   :{out_path}", 'b')
    return out_path


def _roc_summary(y_true, scores, score_name, max_resolution=DEFAULT_ROC_RES):
    """Return ROC curve data plus the summary fields derived from it."""
    try:
        roc = roc_from_scores(y_true, scores, max_resolution=max_resolution)
        return roc, {'roc_auc': roc['auc']}
    except Exception as e:
        return None, {'roc_auc': None, 'roc_error': str(e)}


def _min_clips_pool(clip_pred, clip_score=None, min_val=DEFAULT_MIN_CLIPS):
    """Pool clip predictions by a minimum count or fraction of positive clips."""
    score = float(np.sum(np.asarray(clip_pred, dtype=np.int64) == 1))
    if isinstance(min_val, int):
        target = min_val
    elif isinstance(min_val, float):
        target = round(len(clip_pred) * min_val)
    else:
        raise TypeError(f"min_val must be int or float, got {type(min_val).__name__}")
    return int(score >= target), score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the basic binary metrics and 2x2 confusion matrix."""
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    n = len(y_true)
    return {
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'accuracy': (tn + tp) / n if n > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


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


# -----------------------------------------------------------------------
#* Clip evaluation
# -----------------------------------------------------------------------
def analyze_clip_test(test_results: Path | str | dict, **kwargs):
    """Analyze raw clip probabilities and build one clip-level summary report."""
    raw_res, results_path = _resolve_input(test_results)
    threshold = kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD)
    y_true, y_pred, scores, score_name = _get_eval_arrays(raw_res, threshold=threshold)

    summary = binary_metrics(y_true, y_pred)
    summary.update({'raw_results_path': str(results_path),
                    'raw_results_type': score_name,
                    'analysis_mode': 'clip',
                    'model_path': raw_res.get('model_path', None),
                    'test_cache': raw_res.get('test_cache', None),
                    'num_clips': len(y_true),
                    'support_clips': _support_counts(y_true),
                    })

    roc, roc_info = _roc_summary(y_true, scores, score_name, max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES))
    summary.update(roc_info)

    tst_name = results_path.stem if results_path is not None else 'clip_test'
    out_name = kwargs.get('output_name', f"{tst_name}_clip-summary")
    out_base = resolve_output_path(results_path, out_name, kwargs.get('out_path', None))
    # out_path = out_base.with_suffix('.json') if out_base is not None else None
    summary['output_dir'] = str(out_base.parent) if out_base is not None else None
    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_csv'] = (out_base.with_suffix('.csv').name if save_roc_csv and roc is not None and out_base is not None else
                          ('N/A' if save_roc_csv in {False, None} else None))

    if out_base is not None:
        out_path = _save_analyze_summary(summary, out_base.with_suffix('.json'))
    else:
        print("[INFO] Analysis complete\n Summary file wasn't saved; (please provide out_path)")

    if roc is not None and (out_base is not None or bool(kwargs.get('show_roc', False))):
        plot_roc_curve( roc, save_to=out_base, save_csv=bool(save_roc_csv), show=bool(kwargs.get('show_roc', False)),
                             title=kwargs.get('roc_title', f"ROC Curve for {tst_name} ({score_name})"), )
    if kwargs.get('print', True):
        print_test_report(summary)
    return summary


# -----------------------------------------------------------------------
#* Video evaluation
# -----------------------------------------------------------------------
def analyze_video_test(test_res: Path | str | dict, **kwargs):
    """Analyze clip probabilities at video level and build one video summary report."""
    raw_res, res_path = _resolve_input(test_res)
    threshold = kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD)
    y_true, y_pred, scores, score_name = _get_eval_arrays(raw_res, threshold=threshold)

    if 'meta_video' not in raw_res or len(y_true) != len(y_pred):
        raise KeyError("Results file meta data are defective")

    meta_video = np.asarray(raw_res['meta_video'])
    if len(meta_video) != len(y_true):
        raise ValueError(f"meta_video length mismatch: {len(meta_video)} vs {len(y_true)}")

    clip_scores = scores if score_name == 'y_prob' else None
    pool_func = kwargs.get('pool_func', _min_clips_pool)
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
            print_color(f"[WARN] Inconsistent GT in video {vid}; excluded from video test", 'o')
            excluded_videos.append(vid)
            continue

        clip_pred = y_pred[idx]
        clip_score = clip_scores[idx] if clip_scores is not None else None
        if pool_func is _min_clips_pool:
            pooled = pool_func(clip_pred, clip_score, min_val=min_val)
        else:
            pooled = pool_func(clip_pred, clip_score)

        if isinstance(pooled, tuple):
            videof_pred, video_score = pooled
        else:
            video_pred = pooled
            _, video_score = _min_clips_pool(clip_pred, clip_score, min_val=min_val)

        vid_true.append(int(uniq_true[0]))
        vid_pred.append(int(video_pred))
        vid_score.append(float(video_score))

    if len(vid_true) == 0:
        raise ValueError("No valid videos left for analysis after excluding inconsistent GT videos")

    vid_true = np.asarray(vid_true, dtype=np.int64)
    vid_pred = np.asarray(vid_pred, dtype=np.int64)
    vid_score = np.asarray(vid_score, dtype=np.float64)

    summary = binary_metrics(vid_true, vid_pred)
    summary.update({
        'raw_results_path': str(res_path),
        'raw_results_type': score_name if score_name == 'y_prob' else 'video_score',
        'analysis_mode': 'video',
        'model_path': raw_res.get('model_path', None),
        'test_cache': raw_res.get('test_cache', None),
        'num_clips': len(y_true),
        'support_clips': _support_counts(y_true),
        'num_videos': len(vid_true),
        'support_video': _support_counts(vid_true),
        'num_videos_excluded': len(excluded_videos),
        'excluded_videos': excluded_videos,
    })

    roc, roc_info = _roc_summary(vid_true, vid_score, 'video_score', max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES))
    summary.update(roc_info)

    tst_name = res_path.stem if res_path is not None else 'video_test'
    out_name = kwargs.get('output_name', f"{tst_name}_video-summary")
    out_base = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
    summary['output_dir'] = str(out_base.parent) if out_base is not None else None
    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_csv'] = (out_base.with_suffix('.csv').name if save_roc_csv and roc is not None and out_base is not None else
                          ('N/A' if save_roc_csv in {False, None} else None)    )

    if out_base is not None:
        out_path = _save_analyze_summary(summary, out_base.with_suffix('.json'))
    else:
        print("[WARN] Summary file wasn't saved; invalid or missing out_path")

    if roc is not None and (out_base is not None or bool(kwargs.get('show_roc', False))):
        plot_roc_curve(roc, save_to=out_base, save_csv=bool(save_roc_csv), show=bool(kwargs.get('show_roc', False)),
                       title=kwargs.get('roc_title', f"Video ROC for {tst_name}"),)
    if kwargs.get('print', True):
        print_test_report(summary)
    return summary


def print_test_report(results, **kwargs):
    """Print a compact aligned CLI view of one saved or in-memory summary."""
    def _fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{precision}f}"
        if isinstance(v, np.integer):
            return str(int(v))
        return str(v)

    if isinstance(results, (str, Path)):
        with Path(results).open('r') as f:
            summary = json.load(f)
    elif isinstance(results, dict):
        summary = results
    else:
        raise TypeError("results must be dict or path to summary json")

    precision = kwargs.get('precision', 4)
    label_w = kwargs.get('label_width', 20)
    analysis_mode = summary.get('analysis_mode', None)

    cm = summary.get('confusion_matrix', None)
    if analysis_mode == 'stream' and cm is None:
        cm_clips = summary.get('cm_clips', None)
        if isinstance(cm_clips, dict):
            cm = [[cm_clips.get('tn', 0), cm_clips.get('fp', 0)],
                  [cm_clips.get('fn', 0), cm_clips.get('tp', 0)]]

    testing_set = summary.get('testing_set', {})
    support = summary.get('support_video', summary.get('support_clips', None))
    if support is None:
        support = testing_set.get('videos_support', testing_set.get('clips_support', None))
    support_pair = _support_pair(support)
    support_str = f"{support_pair[0]}/ {support_pair[1]}" if support_pair is not None else 'N/A'
    num_samples = summary.get('num_videos', summary.get('num_clips', None))
    if num_samples is None:
        num_samples = testing_set.get('videos_num', testing_set.get('clips_num', None))


    rows = [("Predictions file", Path(summary.get('raw_results_path', '')).name),
            ("Num_samples", num_samples),
            ("GT_counts 0/1", support_str),
            ("accuracy", summary.get('accuracy', None)),
            ("recall", summary.get('recall', None)),
            ("FPR", summary.get('FPR', None)),
            ("AUC", summary.get('roc_auc', summary.get('ROC AUC', None))),
            ]
    if analysis_mode == 'stream':
        pass

    rows.extend([("False positive time", summary.get('false_positive_time', None)),
                 ("Miss time", summary.get('miss_time', None)),
                 ("GT events", summary.get('gt_events_num', None)),
                 ("Pred_events", summary.get('pred_events_num', None)),
                 ("Detected events", summary.get('detected_events', None)),
                 ("Missed events", summary.get('missed_events', None)),
                 ("False events", summary.get('false_events_num', None)),
                 ("Detection lag", summary.get('detection_lag', None)),
                 ("Onset delay", summary.get('mean_onset_delay', None)),
                 ("Offset error", summary.get('mean_offset_err', None)),
                 ])

    print("\n===== Test Summary =====")
    for k, v in rows:
        if v is not None:
            print(f"{k:<{label_w}}: {_fmt(v)}")

    if cm is not None and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        print(f"{'Confusion Matrix':<{label_w}}: pred-0  pred-1\n"
              f"{'True: 0':<{label_w}}[[{cm[0][0]:>5}, {cm[0][1]:>5}]\n"
              f"{'True: 1':<{label_w}} [{cm[1][0]:>5}, {cm[1][1]:>5}]]\n\n")

    return summary


def run_regression_suite(phase='refactor', **kwargs):
    """Run clip/video regression outputs from cache NPZ files."""
    from torch_clip_model import run_testing

    model_path = Path(kwargs.get('model_path', DEFAULT_REG_MODEL))
    out_root = Path(kwargs.get('out_root', "work_dirs/json_models/testing"))/phase/'evaluation_core'
    show_roc = bool(kwargs.get('show_roc', False))
    save_roc_csv = bool(kwargs.get('roc_csv', True))
    threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))

    cases = (('train_clip', Path("data/cache/win-study/J-RWL_25ft_3w-1o5_train.npz"), False),
             ('train_video', Path("data/cache/win-study/J-RWL_25ft_3w-1o5_train.npz"), True),
             ('test_clip', Path("data/cache/win-study/J-RWL_25ft_3w-1o5_test.npz"), False),
             ('test_video', Path("data/cache/win-study/J-RWL_25ft_3w-1o5_test.npz"), True),)

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

#547(1,3,)
