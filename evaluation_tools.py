from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
#* local imports
from common.my_local_utils import print_color

DEFAULT_ROC_RES = 100
DEFAULT_MIN_CLIPS = 2

# -----------------------------------------------------------------------
#* Local helpers
# -----------------------------------------------------------------------
def _scalar_or_none(arr):
    """ Convert scalar-like numpy values to Python scalars;
        other values remain unchanged."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(()).item()
    return arr

def _load_raw_results_npz(npz_path:str|Path) -> dict:
    """Load a raw test-results NPZ file into a normalized dictionary."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    raw = {'source_path': str(npz_path),
           'model_path' : _scalar_or_none(data['model_path']) if 'model_path' in data.files else None,
           'test_cache' : _scalar_or_none(data['test_cache']) if 'test_cache' in data.files else None,
           'threshold'  : _scalar_or_none(data['threshold'])  if 'threshold'  in data.files else None,
           'hidden_dim' : _scalar_or_none(data['hidden_dim']) if 'hidden_dim' in data.files else None,
           'cache_index': data['cache_index'] if 'cache_index' in data.files else None,
           'y_true': data['y_true'].astype(np.int64),
           'y_pred': data['y_pred'].astype(np.int64),
           'y_prob': data['y_prob'].astype(np.float32) if 'y_prob' in data.files else None,}

    if 'video_name' in data.files:
        raw['video_name'] = data['video_name']
    if 'time_stamp' in data.files:
        raw['time_stamp'] = data['time_stamp']
    if 'meta_video' in data.files:
        raw['meta_video'] = data['meta_video']
    if 'meta_t_start' in data.files:
        raw['meta_t_start'] = data['meta_t_start']
    if 'meta_t_end' in data.files:
        raw['meta_t_end'] = data['meta_t_end']
    return raw

def _validate_raw_results(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """ Validate raw payload and return y_true/y_pred as int64 1D arrays."""
    if not isinstance(raw, dict):
        raise TypeError("raw results must be a dict")

    missing = [k for k in ('y_true', 'y_pred') if k not in raw]
    if missing:
        raise KeyError(f"raw results missing required keys: {missing}")

    y_true = np.asarray(raw['y_true'], dtype=np.int64)
    y_pred = np.asarray(raw['y_pred'], dtype=np.int64)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true/y_pred must be 1D arrays, got {y_true.shape} and {y_pred.shape}")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}")

    return y_true, y_pred


def _resolve_input(test_results):
    """  Normalize public analyzer input into raw results dict and optional source path."""
    if   isinstance(test_results, (str, Path)):
        src_path = Path(test_results)
        raw = _load_raw_results_npz(src_path)
    elif isinstance(test_results, dict):
        # src_path = None
        src_path = test_results.get('path', None)
        raw = test_results
    else:
        raise TypeError(f"Can't load test npz file:{test_results}")
    return raw, src_path


def _resolve_output_pah(src_path, output_name, out_path=None):
    """ Resolve output JSON path from explicit path or source file location. """

    out_path = out_path or (src_path.parent if src_path else None)
    if out_path is None:
        return None
    out_path = Path(out_path)
    if out_path.suffix == '':
        return out_path/output_name
    return out_path


def _support_counts(y_true):
    """ Return class counts as [class_0_count, class_1_count]. """
    y_true = np.asarray(y_true, dtype=np.int64)
    return [int(np.sum(y_true == 0)), int(np.sum(y_true == 1))]


def _support_pair(obj) -> tuple[int, int] | None:
    """ Normalize current list and older dict/json support formats."""
    try:
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return int(obj.get(0, obj.get('0'))), int(obj.get(1, obj.get('1')))
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return (int(obj[0]), int(obj[1])) if len(obj) == 2 else None
        else:
            return None
    except (TypeError, ValueError):
        return None


def _save_analyze_summary(summary, out_path):
    """ Save summary JSON if path is given, otherwise print a short notice. """

    testing_set = {'test_cache': summary.get('test_cache', None),
                   'clips_num': summary.get('num_clips', None),
                   'clips_support': summary.get('support_clips', None),}
    if summary.get('analysis_unit', None) == 'video':
        testing_set['videos_num'] =  summary.get('num_videos', None)
        testing_set['videos_support'] = summary.get('support_video', None)

    save_summary = { 'raw_results_path': summary.get('raw_results_path', None),
                     'raw_results_type': summary.get('raw_results_type', None),
                     'raw_results_unit' : summary['analysis_unit'], #  summary.get('raw_result_unit', None),
                     'model'      :summary.get('model_path',''),
                     'testing_set': testing_set,
                     'confusion_matrix': summary.get('confusion_matrix', None),
                     'accuracy': summary.get('accuracy', None),
                     'recall': summary.get('recall', None),
                     'FPR': summary.get('FPR', None),
                     'ROC AUC': summary.get('roc_auc', None),
                     }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=False, exist_ok=True)
        with out_path.open('w') as f:
            json.dump(save_summary, f, indent=2)
        print("Analysis complete")
        print_color(f"  Summary saved to   :{out_path}", 'b')
    else:
        print("[INFO] Analysis complete\n Summary file wasn't saved; (please provide out_path)")
    return out_path


def _roc_summary(y_true, scores, score_name, max_resolution=DEFAULT_ROC_RES):
    """ Compute ROC outputs and the matching summary fields. """
    try:
        roc = roc_from_scores(y_true, scores, max_resolution=max_resolution)
        return roc, {'roc_auc':roc['auc'], 'roc_score_type': score_name}
    except Exception as e:
        return None, {'roc_auc':None, 'roc_score_type': score_name, 'roc_error': str(e)}


def _min_clips_pool(clip_pred, clip_score=None, min_val=DEFAULT_MIN_CLIPS):
    """ pool by minimum number or fraction of positive clips. """

    score = float(np.sum(np.asarray(clip_pred, dtype=np.int64) == 1))
    if   isinstance(min_val, int):
        target = min_val
    elif isinstance(min_val, float):
        target = round(len(clip_pred)*min_val)
    else:
        raise TypeError(f"min_val must be int or float, got {type(min_val).__name__}")
    return int(score >= target), score

# -----------------------------------------------------------------------
#* Main and API functions
# -----------------------------------------------------------------------

def analyze_clip_test(test_results:Path|str|dict, **kwargs):
    """ Build an evaluation summary from raw test results and dumps it into JSON.
        :param test_results: Accepts either Path/str to raw test-results NPZ,
                             or in-memory raw-results dict.
        Optional parameters in kwargs:
        out_path :  (Path/str) for the results JSON file, if not provided, use input path.
                    or infer path from the results dict
        show_roc:  If True draws ROC image
        print   :  if True print results to consol
    """
    #*Normalize arguments
    raw_res, results_path = _resolve_input(test_results)
    y_true, y_pred = _validate_raw_results(raw_res)
    clip_support = _support_counts(y_true)

    #* Start Analysis
    summary = binary_metrics(y_true, y_pred)
    #* Prefer probability scores when available; otherwise use hard predictions.
    y_prob = raw_res.get('y_prob', None)
    if y_prob is not None:
        scores = np.asarray(y_prob, dtype=np.float64)
        score_name = 'y_prob'
    else:
        scores = np.asarray(y_pred, dtype=np.float64)
        score_name = 'y_pred'

    summary.update({'raw_results_path': str(results_path),
                    'raw_results_type': score_name,
                    'analysis_unit': 'clip',
                    'model_path': raw_res.get('model_path', None),
                    'test_cache': raw_res.get('test_cache', None),
                    'num_clips': len(y_true),
                    'support_clips': clip_support,
                    })

    roc, roc_info = _roc_summary(y_true, scores, score_name,
                                 max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES))
    summary.update(roc_info)

    #* out_path can be either target directory or full target JSON path.
    tst_name = results_path.stem if results_path is not None else ''
    out_name = kwargs.get('output_name', f"{tst_name}_clip-summary.json")
    out_path = _resolve_output_pah(results_path, out_name, kwargs.get('out_path', None))
    out_path = _save_analyze_summary(summary, out_path)

    if roc is not None:
        plot_roc_curve(roc, save_to=out_path, show=bool(kwargs.get('show_roc', False)),
                       title=kwargs.get('roc_title', f"ROC Curve for {tst_name} ({score_name})"), )

    if kwargs.get('print', True):
        print_test_report(summary)
    return summary


def analyze_video_test(test_res:Path|str|dict, **kwargs):
    """ Build a video-level evaluation summary from clip-level raw test results.
        Videos with inconsistent clip GT labels are skipped with a warning.
        Optional kwargs:
        out_path     : output JSON path or directory
        output_name  : output JSON file name if out_path is a directory
        pool_func    : function that maps clip predictions to one video prediction
        min_val      : threshold for the default video pool rule
        show_roc     : if True draws ROC
        print        : if True print report to console
    """
    #* Normalize arguments
    raw_res, res_path = _resolve_input(test_res)
    y_true, y_pred = _validate_raw_results(raw_res)
    # clip_support = _support_counts(y_true)

    if 'meta_video' not in raw_res or len(y_true) != len(y_true):
        raise KeyError("Results file meta data are defective")

    meta_video = np.asarray(raw_res['meta_video'])
    if len(meta_video) != len(y_true):
        raise ValueError(f"meta_video length mismatch: {len(meta_video)} vs {len(y_true)}")

    y_prob = raw_res.get('y_prob', None)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if len(y_prob) != len(y_true):
            raise ValueError(f"y_prob length mismatch: {len(y_prob)} vs {len(y_true)}")

    pool_func = kwargs.get('pool_func', _min_clips_pool)
    min_val = kwargs.get('min_val', kwargs.get('threshold', DEFAULT_MIN_CLIPS))
    video_to_idx = {}
    for i, vid in enumerate(meta_video):
        video_to_idx.setdefault(str(vid), []).append(i)

    vid_names, vid_true, vid_pred, vid_score = [], [], [], []
    excluded_videos = []
    for vid, idx in video_to_idx.items():
        clip_true = y_true[idx]
        uniq_true = np.unique(clip_true)
        if len(uniq_true) != 1:
            print_color(f"[WARN] Inconsistent GT in video {vid}; excluded from video test", 'o')
            excluded_videos.append(vid)
            continue

        clip_pred = y_pred[idx]
        clip_score = y_prob[idx] if y_prob is not None else None
        if pool_func is _min_clips_pool:
            pooled = pool_func(clip_pred, clip_score, min_val=min_val)
        else:
            pooled = pool_func(clip_pred, clip_score)
        if isinstance(pooled, tuple):
            video_pred, video_score = pooled
        else:
            video_pred = pooled
            _, video_score = _min_clips_pool(clip_pred, clip_score, min_val=min_val)

        vid_names.append(vid)
        vid_true.append(int(uniq_true[0]))
        vid_pred.append(int(video_pred))
        vid_score.append(float(video_score))

    if len(vid_true) == 0:
        raise ValueError("No valid videos left for analysis after excluding inconsistent GT videos")

    vid_true = np.asarray(vid_true, dtype=np.int64)
    vid_pred = np.asarray(vid_pred, dtype=np.int64)
    vid_score = np.asarray(vid_score, dtype=np.float64)
    # video_support = _support_counts(vid_pred)

    print_color(_support_counts(y_true), 'r')
    summary = binary_metrics(vid_true, vid_pred)
    summary.update({'raw_results_path': str(res_path) ,
                    'raw_results_type': 'y_prob' if y_prob is not None else 'y_pred',
                    'analysis_unit': 'video',
                    'model_path': raw_res.get('model_path', None),
                    'test_cache': raw_res.get('test_cache', None),
                    'num_clips': len(y_true),
                    'support_clips': _support_counts(y_true),  # clip_support,
                    'num_videos': len(vid_true),
                    'support_video': _support_counts(vid_true),
                    'num_videos_excluded': len(excluded_videos),
                    'excluded_videos': excluded_videos, })

    roc, roc_info = _roc_summary(vid_true, vid_score, 'video_score',
                                 max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES))
    summary.update(roc_info)

    # * out_path can be either target directory or full target JSON path.
    tst_name = res_path.stem if res_path is not None else 'video_test'
    out_name = kwargs.get('output_name', f"{tst_name}_video-summary.json")
    out_path = _resolve_output_pah(res_path, out_name, kwargs.get('out_path', None))
    out_path = _save_analyze_summary(summary, out_path)

    if roc is not None:
        plot_roc_curve(roc, save_to=out_path, show=bool(kwargs.get('show_roc', False)),
                            title=kwargs.get('roc_title', f"Video ROC for {tst_name}"),)

    if kwargs.get('print', True):
        print_test_report(summary)
    return summary #345

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """ Compute binary classification metrics from true/predicted labels."""

    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    n = len(y_true)

    return {'confusion_matrix': [[tn, fp], [fn, tp]],
            'accuracy': (tn + tp)/n if n > 0 else 0.0,
            'recall'  : tp/(tp + fn) if (tp + fn) > 0 else 0.0,
            'FPR'     : fp/(fp + tn) if (fp + tn) > 0 else 0.0,}


def roc_from_scores(y_true: np.ndarray, scores: np.ndarray, max_resolution=DEFAULT_ROC_RES):
    """ Compute ROC arrays and AUC from binary labels and scores.
    :param y_true:  Grand truth labels
    :param scores:  Predictions, in form of binary labels of probabilities
    :param int max_resolution: Max ROC points. If None, 0, or False -> no limiting
    :return: dict with 'fpr', 'tpr' and 'thresholds' array and calculated AUC
    """
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

    #* Optional resolution limiting / smoothing.
    if max_resolution not in (None, False, 0):
        n_target = max_resolution
        if n_target < 2:
            raise ValueError(f"max_resolution must be >=2 or None/False/0, got {max_resolution}")

        if len(fpr) > n_target:
            # Unique FPR grid is required for interpolation.
            uniq_fpr, uniq_idx = np.unique(fpr, return_index=True)
            uniq_tpr = tpr[uniq_idx]
            uniq_thr = thresholds[uniq_idx]

            # Interpolate thresholds with finite values only, then restore edge sentinels.
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


#* outputs, handel print and plotting
# --------------------------------------------------

def print_test_report(results, **kwargs):
    """ Print a compact aligned report of testing summary.
        :param results:  dict or JSON path for testing summary
        Optional kwargs:
        - precision: float print precision
        - label_width: left column width
    """
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.{precision}f}"
        return str(v)

    if isinstance(results, (str, Path)):
        res_path = Path(results)
        with res_path.open('r') as f:
            summary = json.load(f)
    elif isinstance(results, dict):
        summary = results
    else:
        raise TypeError("results must be dict or path to summary json")

    precision = kwargs.get('precision', 4)
    label_w = kwargs.get('label_width', 20)

    cm:list|None = summary.get('confusion_matrix', None)
    testing_set = summary.get('testing_set', {})
    support = summary.get('support_video', summary.get('support_clips', None))
    if support is None:
        support = testing_set.get('videos_support', testing_set.get('clips_support', None))
    support_pair = _support_pair(support)
    support_str = f"{support_pair[0]}/ {support_pair[1]}" if support_pair is not None else 'N/A'
    num_samples = summary.get('num_videos', summary.get('num_clips', None))
    if num_samples is None:
        num_samples = testing_set.get('videos_num', testing_set.get('clips_num', None))

    rows = [("Predictions file", Path(summary.get('raw_results_path', '')).name ),
            ("Num_samples", num_samples),
            ("GT_counts 0/1", support_str),
            ("accuracy", summary.get('accuracy', None)),
            ("recall", summary.get('recall', None)),
            ("FPR", summary.get('FPR', None)),
            ("AUC", summary.get('roc_auc', None))
            ]

    print("\n===== Test Summary =====")
    for k, v in rows:
        if v is not None:
            print(f"{k:<{label_w}}: {_fmt(v)}")

    if cm is not None and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        print(f"{'Confusion Matrix':<{label_w}}: pred-0  pred-1\n"
              f"{'True: 0':<{label_w}}[[{cm[0][0]:>5}, {cm[0][1]:>5}]\n"
              f"{'True: 1':<{label_w}} [{cm[1][0]:>5}, {cm[1][1]:>5}]]\n\n")

    return summary


def plot_roc_curve(roc: dict, **kwargs):
    """ Render ROC plot from `roc_from_scores` output.
    Expects dict with keys: `fpr`, `tpr`, `thresholds`, `auc`.
    If `save_to` is provided, saves both PNG and CSV (`fpr;tpr;thresholds`) with same stem.
    :param roc:         Data for the plotting (Expects keys: `fpr`, `tpr`, `thresholds`, `auc`)
    :param kwargs:
    """
    #* Normalize arguments
    fig_size = kwargs.get('figsize', (6, 5))
    dpi = int(kwargs.get('dpi', 120))
    save_to = kwargs.get('save_to', None)
    if 'show' in kwargs:
        show = bool(kwargs['show'])
    else:
        show = save_to is None
    if save_to is None and not show:
        return

    fpr = np.asarray(roc['fpr'], dtype=np.float64)
    tpr = np.asarray(roc['tpr'], dtype=np.float64)
    thresholds = np.asarray(roc['thresholds'], dtype=np.float64)
    auc = float(roc['auc'])

    title = kwargs.get('title', "ROC Curve")
    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, alpha=0.7)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel('False Positive Rate');  plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_to is not None:
        save_to = Path(save_to)
        if save_to.suffix.lower() != '.png':
            save_to = save_to.with_suffix('.png')
        try:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_to, dpi=dpi)
            csv_path = save_to.with_suffix('.csv')
            roc_table = np.column_stack([fpr, tpr, thresholds])
            np.savetxt(csv_path, roc_table, delimiter=';', header='fpr;tpr;thresholds', comments='')
            print_color (f"  ROC plot saved to  :{save_to}\n"
                              f"  ROC table saved to :{csv_path}m",'b')
        except Exception as e:
            print_color(f"Failed to save ROC outputs to {save_to}: {e}", 'r')

    if show:
        plt.show()
    plt.close()

#548(5,3,2)
if __name__ == '__main__':
    pass
    # analyze_test_results('work_dirs/json_models/train_260323-0314_RWF_tms_f18/test_raw_model_260323-202938.npz')

#326(13,1,1)
