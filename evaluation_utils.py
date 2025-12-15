from __future__ import annotations
from pathlib import Path
import numpy as np
import pickle

# from typing import Dict, Iterable, List, Tuple
from typing import Iterable

#*  Basic helpers

# def load_ann_file(ann_path:str|Path) -> Tuple[List[str], List[int]]:
def load_ann_file(ann_path:str|Path) -> tuple(list[int], dict[str, int]):
    """ Load an MMAction-format annotation file.
        <rel/path> <label>
    Returns:  {video_names[str]: labels[int]}  - 'relative paths'
    """
    ann_path = Path(ann_path)
    video_ann: dict[str, int] = {}
    labels: list[int] = []

    with ann_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Use rsplit to be robust to spaces in the path
            path_str, label_str = line.rsplit(" ", 1)
            video_ann[path_str] = int(label_str)
            labels += [int(label_str)]

    return labels, video_ann

def _compute_binary_confusion(y_pred: Iterable[int], y_gt: Iterable[int],
                              positive_label:int=1) -> tuple[int, int, int, int]:
    """ Compute TN, FP, FN, TP for a binary problem.
    :param y_gt, y_pred: iterable of ints, Ground-truth and predicted correspondingly
    :param positive_label : int   label/s to be  considered the positive class.
    """
    y_gt   = np.asarray(list(y_gt)  , dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    pos = int(positive_label)
    neg_mask = (y_gt != pos)
    pos_mask = (y_gt == pos)

    tn = int(np.sum((y_pred == y_gt) & neg_mask))
    fp = int(np.sum((y_pred != y_gt) & neg_mask))
    fn = int(np.sum((y_pred != y_gt) & pos_mask))
    tp = int(np.sum((y_pred == y_gt) & pos_mask))

    return tn, fp, fn, tp


# --- Core evaluation: y_pred + y_gt -> metrics

def evaluate_test_scores(y_pred: Iterable[int],  y_gt: Iterable[int],
                         *,  positive_label: int = 1,) -> dict[str, object]:
    """ Evaluate binary metrics from predicted and ground-truth labels.

    :param y_pred : iterable of int,    Predicted labels, shape [N].
    :param y_gt   : iterable of int,    Ground-truth labels, shape [N].
    :param positive_label: int, Which class is considered "positive" (e.g. violence).
    Return  dict {'num_samples': int,
                  'support': {label: count, ...},
                  'accuracy': float,
                  'recall': float,
                  'false_positive_rate': float,
                  'confusion_matrix': [[tn, fp], [fn, tp]]}
    """
    y_gt = np.asarray(list(y_gt), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)

    if y_gt.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_gt {y_gt.shape} vs y_pred {y_pred.shape}")

    tn, fp, fn, tp = _compute_binary_confusion(y_gt, y_pred, positive_label)
    n = tn + fp + fn + tp

    accuracy = (tn + tp)/n  if  n > 0 else 0.0
    recall   = tp/(tp + fn) if (tp + fn) > 0 else 0.0
    fpr      = fp/(fp + tn) if (fp + tn) > 0 else 0.0

    # Per-class support
    support: dict[int, int] = {}
    for v in y_gt:
        v = int(v)
        support[v] = support.get(v, 0) + 1

    return {'num_samples': int(n),
            'support': support,
            'accuracy': float(accuracy),
            'recall': float(recall),
            'false_positive_rate': float(fpr),
            'confusion_matrix': [[tn, fp], [fn, tp]],}



#* Scores -> predictions

def get_pred(scores: np.ndarray|Iterable, *, positive_label:int=1,
             threshold:float|None=None,) -> np.ndarray:
    """ Turn raw score vectors into predicted labels.
    :param scores: Array-like, [N, C] or list of C-dim arrays
                   Output loaded from the ``scores_path`` pickle.
    :param positive_label : int Index of the positive class (used in thresholded mode).
    :param threshold: If None, use argmax over classes.
                      If set, apply a binary rule on the positive class:
                      pred = positive_label if score[pos] >= threshold else other_class.
    Returns:  np.ndarray, shape[N]  Predicted labels.
    """
    scores_arr = np.asarray(scores)
    if scores_arr.ndim == 1:
        scores_arr = scores_arr.reshape(-1, 1)

    if threshold is None: #* Multiclass argmax
        y_pred = np.argmax(scores_arr, axis=1)
    else:
        pos = int(positive_label)
        # In thresholded mode we expect binary classification,
        #* but we infer the actual label set from the predictions later.
        pos_scores = scores_arr[:, pos]
        #* The negative label is defined as "the other" label.
        neg = 0 if pos != 0 else 1
        y_pred = np.where(pos_scores >= float(threshold), pos, neg)

    return y_pred.squeeze().astype(int)


def evaluate_video_files(ann_file: str | Path, scores_path: str | Path,
                         *, positive_label:int=1, threshold:float|None=None) -> dict[str, object]:
    """ Convenience wrapper: ann_file + scores pickle -> metrics + predictions.
    :param ann_file: path-like;  Annotation file
    :param scores_path: path-like; Pickle produced by tools/test.py
    :param positive_label, threshold: params passed to get_pred
    Returns:  dict { 'metrics': <metrics dict from evaluate_test_scores>,
                    ' video_predictions': {video_name: pred_label}}
    """

    y_gt,_ = load_ann_file(Path(ann_file))

    with Path(scores_path).open("rb") as f:
        raw_scores = pickle.load(f)

    y_pred = get_pred(raw_scores, positive_label=positive_label, threshold=threshold)
    if len(y_pred) != len(y_gt):
        raise ValueError(f"Mismatch: {len(y_pred)} score entries vs {len(y_gt)} labels.")

    metrics = evaluate_test_scores(y_pred, y_gt, positive_label=positive_label)
    return  metrics


def evaluate_frame_files(ann_file:str|Path, scores_path:str|Path, *, positive_label:int = 1,
                         threshold:float|None=None) -> dict[str, object]:
    """ Compute metrics from an MMAction2 test output and a Rawframe ann file.   """
    ann_file = Path(ann_file)
    scores_path = Path(scores_path)

    y_gt, _ = load_ann_file(ann_file)

    with scores_path.open("rb") as f:
        scores = pickle.load(f)

    y_pred = get_pred(scores, positive_label=positive_label, threshold=threshold)

    if len(y_pred) != len(y_gt):
        raise ValueError(f"Mismatch: {len(y_pred)} score entries vs {len(y_gt)} labels.")

    return evaluate_test_scores(y_pred, y_gt)

if __name__ == "__main__":

    from testing import print_metrics
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Evaluate MMAction test scores.")
    # parser.add_argument("ann_file", type=str, help="Annotation file for the test set")
    # parser.add_argument("scores_path", type=str, help="Pickle with raw scores")
    # parser.add_argument("--positive_label", type=int, default=1)
    # parser.add_argument("--threshold", type=float, default=None)
    # args = parser.parse_args()

    # result = evaluate_from_files(
    #     ann_file=args.ann_file,
    #     scores_path=args.scores_path,
    #     positive_label=args.positive_label,
    #     threshold=args.threshold,
    # )

    met_vid = evaluate_video_files(
        ann_file   ="data/video/joint_val.txt",
        scores_path="work_dirs/tsm_R50_MMA_JOINT/results.pkl" )

    met_frm = evaluate_frame_files(
        ann_file   = "data/json_frames/all_label.txt",
        scores_path="work_dirs/tsm_r50_bbfrm/test_eval/test_scores.pkl"   )

    print_metrics(met_vid)
    # print_metrics(met_frm)

#299 -> 292(1,15,10) -> 199(2,1,4)
