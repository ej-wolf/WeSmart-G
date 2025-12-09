from __future__ import annotations

import pickle
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import run_safe_test

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

    y_true = _load_rawframe_labels(ann_file)

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


def test_trn_checkpoint(checkpoint_path: str | Path,
                        config_path: str | Path,
                        ann_file: str | Path,
                        out_dir: str | Path,
                        *,
                        positive_label: int = 1,
                        threshold: float | None = None,
                        python_exec: str = "python",
                        mmaction_root: str | Path = "extern/mmaction2",
                        use_out_flag: str = "--out",
                        ) -> Dict[str, object]:
    """ Run MMAction2 tools/test.py and compute basic metrics.
        A thin wrapper around the official test script. It assumes that
        ``tools/test.py`` accepts an ``--out`` (or ``--dump``) argument which
        saves raw prediction scores as a pickle file.

    Parameters
    ----------
    checkpoint_path: path Trained model checkpoint (.pth).
    config_path: path    
        Config used for testing (same as training, but with proper test split).
    ann_file : path
        Rawframe annotation file for the test set (same order as in test).
    out_dir : path
        Directory where the temporary scores pickle will be stored.
    positive_label : int, default 1
        Index of the positive class (violence).
    threshold : float or None, default None
        Optional decision threshold on the positive class. If None, argmax
        over class scores is used.
    0python_exec : str, default "python"
        Python executable to use when calling tools/test.py.
    mmaction_root : path, default "extern/mmaction2"
        Root of the MMAction2 repo (where tools/test.py lives).
    use_out_flag : str, default "--out"
        Flag passed to tools/test.py to save scores. For some versions this
        might need to be "--dump" instead.

    :returns  dict passing the forward the  evaluate_scores_from_test_output return structure
    """

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    ann_file = Path(ann_file)
    out_dir = Path(out_dir)
    mmaction_root = Path(mmaction_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir/"test_scores.pkl"

    test_script = mmaction_root/"tools/test.py"
    wrapper = Path("run_safe_test.py")
    # test_script = "tools/test.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"Cannot find tools/test.py under {mmaction_root}")

    cmd = [ python_exec, str(wrapper),
            str(config_path), str(checkpoint_path),
            use_out_flag, str(scores_path),
            "--launcher", "none",]

    # cmd2 = (f"python {test_script}  {config_path}  {checkpoint_path}"
    #         f" {use_out_flag} {scores_path}  --launcher none")

    # Run MMAction2's test script in its own repo root
    # subprocess.run(cmd, check=True, cwd=str(mmaction_root))
    subprocess.run(cmd, check=True, cwd=".")

    # Now compute metrics from the saved scores
    # return evaluate_test_scores(        ann_file=ann_file,
    #     scores_path=scores_path,
    #     positive_label=positive_label,
    #     threshold=threshold,
    # )
    return evaluate_test_scores(ann_file, scores_path, positive_label=positive_label, threshold=threshold,)

def lunch_wrapper(**kwargs):

    prfx = Path(kwargs.get('relative_path', '../..' ))

    cmd = [ 'python', kwargs.get('wrapper','run_safe_test.py'),
            str(prfx/kwargs['config_path']), str(prfx/kwargs['checkpoint_path']),
            kwargs.get('use_out_flag', '--dumb'), str(prfx/kwargs['out_dir']/"test_scores.pkl"),
            "--launcher", "none",]
    # run_in = kwargs.get('run_in','.')
    subprocess.run(cmd, check=True, cwd=kwargs.get('run_in','.'))


if __name__ == "__main__":

    # metrics = test_trn_checkpoint(
    #           config_path = "../../configs/TRN/trn_r50_bbfrm_02.py",
    #           checkpoint_path="../../work_dirs/R50_bbrfm_01/best_acc_top1_epoch_5.pth",
    #           ann_file="data/cache/all_label.txt",
    #           out_dir ="../../work_dirs/R50_bbrfm_01/test_eval",
    #           positive_label=1,      # whatever is your “violence” class index
    #           threshold=None,        # or some float if you want explicit thresholding
    #           mmaction_root="extern/mmaction2",
    #           use_out_flag="--dump",  # change to "--dump\out" if needed
    #           )

    test_params = {'wrapper': 'run_safe_test.py',
                   'config_path': "configs/TRN/trn_r50_bbfrm_02.py",
                   'checkpoint_path': "work_dirs/R50_bbrfm_01/best_acc_top1_epoch_5.pth",
                   'out_dir' :"work_dirs/R50_bbrfm_01/test_eval" }



    lunch_wrapper (**test_params)

    metrics = evaluate_test_scores( ann_file="data/cache/all_label.txt",  # same as val_dataloader.ann_file in config
        scores_path="work_dirs/R50_bbrfm_01/test_eval/test_scores.pkl",
        positive_label=1,  # or 0, depending on your label mapping
        threshold=None,  # argmax over scores (default)
        )

    print(metrics)
