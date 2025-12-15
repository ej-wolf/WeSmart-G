from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np

# import run_safe_test
from my_local_utils import get_epoch_pth

def _load_rawframe_labels(ann_file: str | Path) -> np.ndarray:
    """ Load labels from a Rawframe annotation file.
        Each line must be: "frame_dir total_frames label".
        Returns: an int array of shape [N].
    """
    ann_file = Path(ann_file)
    labels: List[int] = []
    with ann_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                lbl = int(parts[2])
            except ValueError:
                continue
            labels.append(lbl)
    if not labels:
        raise ValueError(f"No labels parsed from {ann_file}")
    return np.asarray(labels, dtype=np.int64)


def load_ann_file(ann_path:str|Path):
    ann_path = Path(ann_path)
    samples,labels = [], []
    with ann_path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # path label
            path_str, label_str = line.rsplit(' ', 1)  # safer if path has spaces
            samples.append((path_str, int(label_str)))
            labels += [int(label_str)]
    return samples, labels

def _compute_binary_confusion( y_true:Iterable[int], y_pred:Iterable[int], positive_label:int= 1,)\
                              -> Tuple[int, int, int, int]:
    """ Return TN, FP, FN, TP for a binary problem.
        positive_label is the index of the "violence" / positive class.
    """
    y_true = list(int(v) for v in y_true)
    y_pred = list(int(v) for v in y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    pos = int(positive_label)
    # Assume exactly two labels: {pos, neg}
    all_labels = sorted(set(y_true) | set(y_pred))
    if len(all_labels) != 2 or pos not in all_labels:
        raise ValueError(
            f"Binary confusion expects exactly two labels incl. {pos}, got {all_labels}"
        )
    neg = all_labels[0] if all_labels[0] != pos else all_labels[1]

    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == pos and p == pos:
            tp += 1
        elif t == pos and p == neg:
            fn += 1
        elif t == neg and p == pos:
            fp += 1
        elif t == neg and p == neg:
            tn += 1
    return tn, fp, fn, tp


def evaluate_test_scores(ann_file: str | Path, scores_path: str | Path, *,
                         positive_label:int = 1, threshold: float|None = None,
                         ) -> Dict[str, object]:
    """ Compute metrics from an MMAction2 test output and a Rawframe ann file.
    Parameters:
    ----------
    ann_file : path,   Rawframe annotation used for the test set (same order as in test).
    scores_path : path,  Pickle file produced by tools/test.py with ``--out``/``--dump``.
                  Expected to be a list/array of shape [N, C] or a list of C-dim arrays.
    positive_label : int (default 1), class index to treat as "positive" (violence).
    threshold :  If None (default), prediction is argmax over class scores.
                 If set, we use a binary rule for the positive class:
                ``pred = positive_label if score[pos] >= threshold else other_class``.

    :returns dict { 'num_samples': int,
                    'support': {label: count, ...},
                    'accuracy': float,
                    'recall': float,
                    'false_positive_rate': float,
                    'confusion_matrix': [[tn, fp], [fn, tp]]}
    """
    ann_file = Path(ann_file)
    scores_path = Path(scores_path)

    # y_true = _load_rawframe_labels(ann_file)
    _, y_true = load_ann_file(ann_file)

    with scores_path.open("rb") as f:
        raw_scores = pickle.load(f)

    scores = np.asarray(raw_scores)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    if scores.shape[0] != len(y_true):
        raise ValueError(f"Mismatch: {scores.shape[0]} score entries vs {len(y_true)} labels.")

    #* Determine predictions
    if threshold is None:
        #* Multiclass argmax
        y_pred = np.argmax(scores, axis=1)
    else:
        #* Binary decision on the positive class
        pos = int(positive_label)
        all_labels = sorted(set(int(v) for v in y_true))
        if len(all_labels) != 2 or pos not in all_labels:
            raise ValueError("Thresholded mode expects exactly two classes including "
                            f"positive_label={pos}, got {all_labels}" )
        neg = all_labels[0] if all_labels[0] != pos else all_labels[1]
        pos_scores = scores[:, pos]
        y_pred = np.where(pos_scores >= float(threshold), pos, neg)

    tn, fp, fn, tp = _compute_binary_confusion(y_true, y_pred, positive_label)
    n = tn + fp + fn + tp

    accuracy = (tn + tp)/n if n > 0 else 0.0
    recall = tp/(tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp/(fp + tn) if (fp + tn) > 0 else 0.0

    # Simple per-class support
    support: Dict[int, int] = {}
    for v in y_true:
        v = int(v)
        support[v] = support.get(v, 0) + 1

    return { 'num_samples': int(n), 'support': support,
             'accuracy': float(accuracy), 'recall': float(recall),
             'false_positive_rate': float(fpr),
             'confusion_matrix': [[tn, fp], [fn, tp]],
             }


def launch_wrapper(**kwargs):
    prfx = Path(kwargs.get('relative_path', '../..' ))
    # out_dir = prfx/kwargs.get('out_dir', kwargs['checkpoint_path'])
    score_file = (prfx/kwargs.get('out_dir', Path(kwargs['checkpoint_path']).parent/'test_eval')/
                       kwargs.get('score_file', "test_scores.pkl"))

    cmd = [ 'python', kwargs.get('wrapper','run_safe_test.py'), #* the wrapper script
            str(prfx/kwargs['config_path']),
            str(prfx/kwargs['checkpoint_path']),
            kwargs.get('pkl_flag', '--dump'), str(score_file),
            "--launcher", "none",]

    subprocess.run(cmd, check=True, cwd=kwargs.get('run_in','.'))


def set_with_defaults(run_tag:str, **kwargs)-> dict:


    pth = get_epoch_pth(Path(os.getcwd())/'work_dirs'/run_tag, kwargs.get('pth', 'best'))
    return {'wrapper': kwargs.get('testing_file','run_safe_test.py'),
            'config_path': f"configs/{run_tag}.py",
            'checkpoint_path': f"work_dirs/{run_tag}/{pth}"}



def print_metrics(metrics:dict):
    print(f"Samples  :{metrics['num_samples']};  Labels {metrics['support']}\n"
          f"Accuracy : {metrics['accuracy']:6f}\n"
          f"Recall   : {metrics['recall']:6f}\n"
          f"Confusion matrix: {metrics['confusion_matrix'][0]}\n"
          f"                  {metrics['confusion_matrix'][1]}")



if __name__ == "__main__":


    test_params = {'wrapper': 'run_safe_test.py',
                   'config_path': "configs/TRN/trn_r50_bbfrm_02.py",
                   'checkpoint_path': "work_dirs/tsn_R50_bbrfm/epoch_25.pth",
                   }
    test_params = {'wrapper': 'run_safe_test.py',
                   'config_path': "configs/tsm_R50_MMA_RWF.py",
                   'checkpoint_path': "work_dirs/tsm_R50_MMA_RLVS/best_acc_top1_epoch_5.pth",
                   }

    # test_params = set_with_defaults('tsm_r50_bbfrm')

    # launch_wrapper (**test_params)

    metrics = evaluate_test_scores( #ann_file="data/cache/all_label.txt",  # same as val_dataloader.ann_file in config
        ann_file="data/json_frames/all_label.txt",
        scores_path="work_dirs/tsm_r50_bbfrm/test_eval/test_scores.pkl"
        )

    # metrics = evaluate_test_scores( ann_file="data/video/RWF-2000/val.txt",  # same as val_dataloader.ann_file in config
    #     scores_path="work_dirs/tsm_R50_MMA_RLVS/test_eval/test_scores.pkl",
    #     positive_label=1, threshold=None, )

    print_metrics(metrics)

#256( ,2,15)
