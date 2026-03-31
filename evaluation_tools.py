from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
#* local imports
from common.my_local_utils import print_color

DEFAULT_ROC_RES = 100

# -----------------------------------------------------------------------
#* Local helpers
# -----------------------------------------------------------------------

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """ Compute binary classification metrics from true/predicted labels."""
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    n = len(y_true)

    support = {}
    for v in y_true:
        k = int(v)
        support[k] = support.get(k, 0) + 1

    return {'num_samples': n,
            'support': support,
            'accuracy': (tn + tp)/n if n > 0 else 0.0,
            'recall'  : tp/(tp + fn) if (tp + fn) > 0 else 0.0,
            'FPR': fp/(fp + tn) if (fp + tn) > 0 else 0.0,
            'confusion_matrix': [[tn, fp], [fn, tp]],}

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

# -----------------------------------------------------------------------
#* Main and API functions
# -----------------------------------------------------------------------

def analyze_test_results(test_results:Path|str|dict, **kwargs):
    """ Build an evaluation summary from raw test results and dumps it into JSON.
        :param test_results: Accepts either Path/str to raw test-results NPZ,
                             or in-memory raw-results dict.
        Optional parameters in kwargs:
        out_path :  (Path/str) for the results JSON file, if not provided, use input path.
                    or infer path from the results dict
        show_roc:  If True draws ROC image
        print   :  if True print results to consol
    """

    if isinstance(test_results, (str, Path)):
        src_path = Path(test_results)
        raw = _load_raw_results_npz(src_path)
    elif isinstance(test_results, dict):
        src_path = None
        raw = test_results
    else:
        raise TypeError(" test_results must be dict or path to raw results npz")

    y_true, y_pred = _validate_raw_results(raw)
    summary = _binary_metrics(y_true, y_pred)
    summary.update({'model_path': raw.get('model_path', None),
                    'test_cache': raw.get('test_cache', None),
                    'threshold' : raw.get('threshold' , None),
                    'hidden_dim': raw.get('hidden_dim', None),
                    # 'raw_results_path': str(src_path) if src_path is not None else raw.get('path', None),})
                    'raw_results_path': str(src_path) or  raw.get('path', None),})

    # Prefer probability scores when available; otherwise use hard predictions.
    y_prob = raw.get('y_prob', None)
    if y_prob is not None:
        scores = np.asarray(y_prob, dtype=np.float64)
        score_name = 'y_prob'
    else:
        scores = np.asarray(y_pred, dtype=np.float64)
        score_name = 'y_pred'

    roc = None
    try:
        roc = roc_from_scores(y_true, scores, max_resolution=kwargs.get('max_resolution', 100))
        summary.update({'roc_auc': roc['auc'], 'roc_score_type': score_name,})
    except Exception as e:
        summary.update({'roc_auc': None, 'roc_score_type': score_name, 'roc_error': str(e),})

    tst_name = src_path.stem if src_path is not None else ''
    # out_path can be either target directory or full target JSON path.
    out_name = kwargs.get('output_name', f"{tst_name}_summary.json")
    src_dir = src_path.parent if src_path else None
    out_path = kwargs.get('out_path', None) or src_dir

    if out_path is not None:
        out_path = Path(out_path)
        if out_path.suffix == '':
            out_path.mkdir(parents=False, exist_ok=True)
            out_path = out_path/out_name
        with out_path.open('w') as f:
            json.dump(summary, f, indent=2)
            print(f"Test analysis complete")
            print_color(f"  Summary saved to   :{out_path}", 'b')
    else:
        print("[INFO] Test analysis complete\n Summary file wasn't saved; (please provide out_path)")

    if roc is not None:
        # Keep ROC outputs next to the summary output.
        roc_path = out_path.with_stem(tst_name)
        plot_roc_curve(roc, save_to=out_path, show=bool(kwargs.get('show_roc', False)),
                       title=kwargs.get('roc_title', f"ROC Curve for {tst_name} ({score_name})"),)
        # csv_path = roc_path.with_suffix('.csv')
        # roc_table = np.column_stack([roc['fpr'], roc['tpr'], roc['thresholds']])
        np.savetxt(roc_path.with_suffix('.csv'),
                   np.column_stack([roc['fpr'], roc['tpr'], roc['thresholds']]),
                   delimiter=';', header='FPR;TPR;Thresholds')


    if kwargs.get('print', True):
        print_test_report(summary)
    return summary


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
    support:list = summary.get('support', None)
    support_str = f"{support[0]}/ {support[1]}" if support is not None else 'N/A'

    rows = [("Results file", Path(summary.get('raw_results_path', '')).name ),
            ("Num_samples", summary.get('num_samples', None)),
            ("GT_counts 0/1", support_str),
            ("accuracy", summary.get('accuracy', None)),
            ("recall", summary.get('recall', None)),
            ("FPR", summary.get('FPR', None)),
            ("AUC", summary.get('roc_auc', None))
            # ("model_path", summary.get('model_path', None)),
            # ("test_cache", summary.get('test_cache', None)),
            ]

    print("\n==== Test Summary ====")
    for k, v in rows:
        if v is not None:
            print(f"{k:<{label_w}}: {_fmt(v)}")

    if cm is not None and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        # print(f"{'confusion_matrix':<{label_w}}:")
        # print(f"{'':<{label_w}}  pred0  pred1")
        # print(f"{'true0':<{label_w}}  {cm[0][0]:>5}  {cm[0][1]:>5}")
        # print(f"{'true1':<{label_w}}  {cm[1][0]:>5}  {cm[1][1]:>5}")
        print(f"{'Confusion Matrix':<{label_w}}: pred-0  pred-1\n"
              f"{'True: 0':<{label_w}}[[{cm[0][0]:>5}, {cm[0][1]:>5}]\n"
              f"{'True: 1':<{label_w}} [{cm[1][0]:>5}, {cm[1][1]:>5}]]\n")
        print()

    return summary


def plot_roc_curve(roc: dict, **kwargs):
    """ Render ROC plot from `roc_from_scores` output.
    Expects dict with keys: `fpr`, `tpr`, `thresholds`, `auc`.
    If `save_to` is provided, saves both PNG and CSV (`fpr;tpr;thresholds`) with same stem.
    :param roc:         Data for the plotting (Expects keys: `fpr`, `tpr`, `thresholds`, `auc`)
    :param kwargs:
    """

    fpr = np.asarray(roc['fpr'], dtype=np.float64)
    tpr = np.asarray(roc['tpr'], dtype=np.float64)
    thresholds = np.asarray(roc['thresholds'], dtype=np.float64)
    auc = float(roc['auc'])

    fig_size = kwargs.get('figsize', (6, 5))
    dpi = int(kwargs.get('dpi', 120))
    save_to = kwargs.get('save_to', None)
    if 'show' in kwargs:
        show = bool(kwargs['show'])
    else:
        show = save_to is None

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
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to, dpi=dpi)
        csv_path = save_to.with_suffix('.csv')
        roc_table = np.column_stack([fpr, tpr, thresholds])
        np.savetxt(csv_path, roc_table, delimiter=';', header='fpr;tpr;thresholds', comments='')
        print_color(f"  ROC plot saved to  :{save_to}\n"
                         f"  ROC table saved to :{csv_path}m",'b')
        print(f"AUC: {auc:.6f}")
    else:
        print(f"ROC plot was not saved\nAUC: {auc:.6f}")

    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    pass
    # analyze_test_results('work_dirs/json_models/train_260323-0314_RWF_tms_f18/test_raw_model_260323-202938.npz')

#326(13,1,1)
