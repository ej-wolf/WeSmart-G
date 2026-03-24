from pathlib import Path
import json
import numpy as np


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute basic binary-classification metrics from y_true and y_pred."""
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
            'accuracy': (tn + tp) / n if n > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'confusion_matrix': [[tn, fp], [fn, tp]],}


def _scalar_or_none(arr):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(()).item()
    return arr


def _load_raw_results_npz(npz_path: str | Path) -> dict:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    raw = {'source_path': str(npz_path),
           'model_path': _scalar_or_none(data['model_path']) if 'model_path' in data.files else None,
           'test_cache': _scalar_or_none(data['test_cache']) if 'test_cache' in data.files else None,
           'threshold': _scalar_or_none(data['threshold']) if 'threshold' in data.files else None,
           'hidden_dim': _scalar_or_none(data['hidden_dim']) if 'hidden_dim' in data.files else None,
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
    """Validate raw results payload and return y_true/y_pred as int64 arrays."""
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


def analyze_test_results(test_results:str|Path|dict, **kwargs):
    """Analyze raw test NPZ output created by run_testing and save summary metrics."""
    if isinstance(test_results, (str, Path)):
        src_path = Path(test_results)
        raw = _load_raw_results_npz(src_path)
    elif isinstance(test_results, dict):
        src_path = None
        raw = test_results
    else:
        raise TypeError("test_results must be dict or path to raw results npz")

    y_true, y_pred = _validate_raw_results(raw)
    summary = _binary_metrics(y_true, y_pred)
    summary.update({'model_path': raw.get('model_path', None),
                    'test_cache': raw.get('test_cache', None),
                    'threshold' : raw.get('threshold' , None),
                    'hidden_dim': raw.get('hidden_dim', None),
                    'raw_results_path': str(src_path) if src_path is not None else raw.get('path', None),})

    #     out_path_arg = kwargs.get('out_path', None)
    # if out_path_arg is not None:
    #     out_path = Path(out_path_arg)
    #     if out_path.suffix == '':
    #         out_path = out_path.with_suffix('.json')
    #     out_path.parent.mkdir(parents=True, exist_ok=True)
    # else:
    #     if src_path is None:
    #         print("[INFO] Summary file wasn't saved; (please provide out_path)")
    #         return summary
    #     out_name = f"test_summary_{src_path.stem}.json"
    #     out_path = src_path.parent/out_name
    tst_name = src_path.stem if src_path is not None else ''
    # out_path = kwargs.get('out_path', None)
    out_name = kwargs.get('output_name', f"{tst_name}_summary.json")
    src_dir = src_path.parent if src_path else None
    out_path = kwargs.get('out_path', None) or src_dir
    # if out_path is not None:
    #     out_path = Path(out_path)
    #     if out_path.suffix == '':
    #         out_path = out_path/out_name
    # elif src_path is not None:
    #     out_path = src_path/out_name
    # else:
    #     out_path = None

    if out_path is not None:
        out_path = Path(out_path)
        if out_path.suffix == '':
            out_path.mkdir(parents=False, exist_ok=True)
            out_path = out_path/out_name
        with out_path.open('w') as f:
            json.dump(summary, f, indent=2)
            print(f"Test analysis complete\n Summary saved to {out_path}")
    else:
        print("[INFO] Test analysis complete\n Summary file wasn't saved; (please provide out_path)")
    return summary


def print_test_report(results, **kwargs):
    """Print aligned test summary from dict or from summary JSON path."""
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

    precision = int(kwargs.get('precision', 4))
    label_w = int(kwargs.get('label_width', 20))

    cm = summary.get('confusion_matrix', None)
    support = summary.get('support', None)

    rows = [("num_samples", summary.get('num_samples', None)),
            ("accuracy", summary.get('accuracy', None)),
            ("recall", summary.get('recall', None)),
            ("false_positive_rate", summary.get('false_positive_rate', None)),
            ("model_path", summary.get('model_path', None)),
            ("test_cache", summary.get('test_cache', None)),
            ("raw_results_path", summary.get('raw_results_path', None)), ]

    print("==== Test Summary ====")
    for k, v in rows:
        if v is not None:
            print(f"{k:<{label_w}}: {_fmt(v)}")

    if support is not None:
        s0 = support.get('0', support.get(0, 0))
        s1 = support.get('1', support.get(1, 0))
        print(f"{'support(0/1)':<{label_w}}: {s0}/{s1}")

    if cm is not None and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        print(f"{'confusion_matrix':<{label_w}}:")
        print(f"{'':<{label_w}}  pred0  pred1")
        print(f"{'true0':<{label_w}}  {cm[0][0]:>5}  {cm[0][1]:>5}")
        print(f"{'true1':<{label_w}}  {cm[1][0]:>5}  {cm[1][1]:>5}")

    return summary


def _load_summary(summary_in: str | Path | dict) -> tuple[dict, Path | None]:
    """Load summary from dict or JSON path."""
    if isinstance(summary_in, (str, Path)):
        s_path = Path(summary_in)
        with s_path.open('r') as f:
            return json.load(f), s_path
    if isinstance(summary_in, dict):
        return summary_in, None
    raise TypeError("summary must be dict or path to summary json")


def _roc_from_scores(y_true: np.ndarray, scores: np.ndarray):
    """Compute ROC coordinates and AUC from binary labels and scores."""
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
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, thresholds, auc


def plot_roc_curve(summary, **kwargs):
    """Draw ROC curve from analyze_test_results summary and return AUC details."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for plot_roc_curve") from e

    summary_dict, summary_path = _load_summary(summary)
    raw_path_val = summary_dict.get('raw_results_path', None)
    if not raw_path_val:
        raise ValueError("summary missing 'raw_results_path'; cannot compute ROC")

    raw_path = Path(raw_path_val)
    raw = _load_raw_results_npz(raw_path)
    y_true, y_pred = _validate_raw_results(raw)

    y_prob = raw.get('y_prob', None)
    if y_prob is not None:
        scores = np.asarray(y_prob, dtype=np.float64)
        score_name = 'y_prob'
    else:
        print("[WARN] y_prob not found in raw results. ROC uses y_pred (coarse curve).")
        scores = np.asarray(y_pred, dtype=np.float64)
        score_name = 'y_pred'

    fpr, tpr, thresholds, auc = _roc_from_scores(y_true, scores)

    fig_size = kwargs.get('figsize', (6, 5))
    dpi = int(kwargs.get('dpi', 120))
    show = bool(kwargs.get('show', False))
    title = kwargs.get('title', f"ROC Curve ({score_name})")

    out_path_arg = kwargs.get('out_path', None)
    if out_path_arg is not None:
        out_path = Path(out_path_arg)
        if out_path.suffix == '':
            out_path = out_path / f"{raw_path.stem}_roc.png"
    else:
        base_dir = summary_path.parent if summary_path is not None else raw_path.parent
        out_path = base_dir / f"{raw_path.stem}_roc.png"
    if out_path.suffix == '':
        out_path = out_path.with_suffix('.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    print(f"ROC plot saved to {out_path}\nAUC: {auc:.6f}")
    return {'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'score_type': score_name,
            'plot_path': str(out_path)}


if __name__ == '__main__':
    pass
    # analyze_test_results('work_dirs/json_models/train_260323-0314_RWF_tms_f18/test_raw_model_260323-202938.npz')
