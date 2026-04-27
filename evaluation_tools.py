from pathlib import Path
import csv
import json
import numpy as np
#* local imports
from common.my_local_utils import print_color, serialize_json_data, resolve_output_path
from visual_util import plot_roc_curve

DEFAULT_ROC_RES = 100
DEFAULT_MIN_CLIPS = 2
STREAM_MATCH_MIN_OVERLAP = 1e-9
STREAM_MATCH_MAX_LAG = None
STREAM_MAX_EVENT_GAP = 1


# -----------------------------------------------------------------------
#* IO Helpers
# TODO: Consider Splitting the IO helpers into a small evaluation_io module.
# -----------------------------------------------------------------------
def _load_raw_results_npz(npz_path:str|Path) -> dict:
    """ Load a raw test-results NPZ file into a normalized dictionary."""
    def _scalar_or_none(arr):
        """ Convert scalar-like numpy values to plain Python scalars."""
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
    raw = {'model_path' : _scalar_or_none(data['model_path']) if 'model_path' in data.files else None,
           'test_cache' : _scalar_or_none(data['test_cache']) if 'test_cache' in data.files else None,
           'y_true': data['y_true'].astype(np.int64),
           'y_pred': data['y_pred'].astype(np.int64),
           'y_prob': data['y_prob'].astype(np.float32) if 'y_prob' in data.files else None,}

    if 'meta_video' in data.files:
        raw['meta_video'] = data['meta_video']
    if 'meta_t_start' in data.files:
        raw['meta_t_start'] = data['meta_t_start']
    if 'meta_t_end' in data.files:
        raw['meta_t_end'] = data['meta_t_end']
    if 'meta_n_frames' in data.files:
        raw['meta_n_frames'] = data['meta_n_frames']
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
        src_path = test_results.get('path', None)
        if isinstance(src_path, str):
            src_path = Path(src_path)
        raw = test_results
    else:
        raise TypeError(f"Can't load test npz file:{test_results}")
    return raw, src_path


# -----------------------------------------------------------------------
#* Metric / Report Helpers
# TODO: Consider splitting metric/report helpers into a separate module.
# -----------------------------------------------------------------------
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


def _cm_dict(cm) -> dict[str, int]|None:
    """ Convert a 2x2 confusion matrix to a named dict. """
    if cm is None or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        return None
    return {'tn': int(cm[0][0]), 'fp': int(cm[0][1]), 'fn': int(cm[1][0]), 'tp': int(cm[1][1])}


def _save_analyze_summary(summary, out_path:Path|str):
    """Write one normalized summary JSON to `out_path`."""
    # TODO: Consider generalizing this into a common JSON report writer.
    #   Reason: most of the logic here is generic save/ordering/serialization;
    #   the evaluation-specific part is mainly the summary payload shaping.

    if not Path(out_path).parent.is_dir():
        print(f"[WARN] Bad out_path; {out_path.name} was not saved ")
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
    if output_dir is None and out_path is not None:
        output_dir = str(Path(out_path).parent)
    events_info = summary.get('events_info', None)
    timeline_csvs = summary.get('timeline_csvs', None)
    roc_csv = summary.get('roc_csv', None)

    save_summary = {'raw_results_path': summary.get('raw_results_path', None),
                    'model': summary.get('model_path', ''),
                    'output_dir': output_dir,
                    'events_info': Path(events_info).name if events_info else None,
                    'timeline_csvs': serialize_json_data(timeline_csvs),
                    'analysis_mode': analysis_mode,
                    'testing_set': testing_set,
                    'accuracy': summary.get('accuracy', None),
                    'recall': summary.get('recall', None),
                    'FPR': summary.get('FPR', None),
                    'ROC AUC': summary.get('roc_auc', None),
                    'roc_type': summary.get('raw_results_type', None),
                    'roc_csv': Path(roc_csv).name if roc_csv else None,
                    }

    if analysis_mode != 'stream':
        save_summary['confusion_matrix'] = summary.get('confusion_matrix', None)

    if analysis_mode == 'stream':
        extra_keys = ('cm_clips',  'false_positive_time', 'miss_time',
                      'gt_events_num', 'pred_events_num', 'detected_events',
                      'missed_events', 'false_events_num', 'mean_onset_delay',
                      'mean_offset_err', 'match_config', 'detection_lag',)
        for key in extra_keys:
            if key in summary:
                save_summary[key] = serialize_json_data(summary.get(key))

    with out_path.open('w') as f:
        json.dump(serialize_json_data(save_summary), f, indent=2)
    print("Analysis complete")
    print_color(f"  Summary saved to   :{out_path}", 'b')
    return out_path


def _roc_summary(y_true, scores, score_name, max_resolution=DEFAULT_ROC_RES):
    """Return ROC curve data plus the summary fields derived from it."""
    try:
        roc = roc_from_scores(y_true, scores, max_resolution=max_resolution)
        return roc, {'roc_auc':roc['auc']}
    except Exception as e:
        return None, {'roc_auc':None, 'roc_error': str(e)}


# -----------------------------------------------------------------------
#* Evaluation-specific Helpers
# -----------------------------------------------------------------------
def _min_clips_pool(clip_pred, clip_score=None, min_val=DEFAULT_MIN_CLIPS):
    """Pool clip predictions by a minimum count or fraction of positive clips."""

    score = float(np.sum(np.asarray(clip_pred, dtype=np.int64) == 1))
    if   isinstance(min_val, int):
        target = min_val
    elif isinstance(min_val, float):
        target = round(len(clip_pred)*min_val)
    else:
        raise TypeError(f"min_val must be int or float, got {type(min_val).__name__}")
    return int(score >= target), score


def _build_stream_events(mask, t_frm, step, max_event_gap=STREAM_MAX_EVENT_GAP):
    """Merge consecutive positive timeline rows into event segments."""
    mask = np.asarray(mask, dtype=bool)
    t_frm = np.asarray(t_frm, dtype=np.float64)
    events = []
    start_idx = None
    last_pos_idx = None
    gap_count = 0

    for idx, is_pos in enumerate(mask):
        if is_pos and start_idx is None:
            start_idx = idx
            last_pos_idx = idx
            gap_count = 0
        elif is_pos and start_idx is not None:
            last_pos_idx = idx
            gap_count = 0
        elif not is_pos and start_idx is not None:
            gap_count += 1
            if gap_count <= max_event_gap:
                continue

            end_idx = last_pos_idx
            events.append({'start': float(t_frm[start_idx]),
                           'end': float(t_frm[end_idx] + step),
                           't_frm_start': float(t_frm[start_idx]),
                           't_frm_end': float(t_frm[end_idx]),
                           'idx_start': int(start_idx),
                           'idx_end': int(end_idx),})
            start_idx = None
            last_pos_idx = None
            gap_count = 0

    if start_idx is not None:
        end_idx = last_pos_idx
        events.append({'start': float(t_frm[start_idx]),
                       'end': float(t_frm[end_idx] + step),
                       't_frm_start': float(t_frm[start_idx]),
                       't_frm_end': float(t_frm[end_idx]),
                       'idx_start': int(start_idx),
                       'idx_end': int(end_idx),})
    return events


def _event_overlap(gt_event, pred_event):
    """ Return overlap duration and overlap ratio normalized by GT duration."""
    start = max(float(gt_event['start']), float(pred_event['start']))
    end = min(float(gt_event['end']), float(pred_event['end']))
    overlap = max(0.0, end - start)
    gt_dur = max(float(gt_event['end']) - float(gt_event['start']), 1e-12)
    return overlap, overlap/gt_dur


def _match_stream_events(gt_events, pred_events, min_overlap, max_lag):
    """ Greedy match of predicted stream events to GT events."""
    used_pred = set()
    detected_events, missed_events = [], []

    for gt_idx, gt_event in enumerate(gt_events):
        candidates = []
        for pred_idx, pred_event in enumerate(pred_events):
            if pred_idx in used_pred:
                continue
            overlap, overlap_ratio = _event_overlap(gt_event, pred_event)
            if overlap <= 0 or overlap_ratio < min_overlap:
                continue

            onset_lag = float(pred_event['start']) - float(gt_event['start'])
            if max_lag is not None and onset_lag > max_lag:
                continue

            offset_err = float(pred_event['end']) - float(gt_event['end'])
            candidates.append((abs(onset_lag), -overlap_ratio, abs(offset_err), pred_idx,
                               overlap, overlap_ratio, offset_err))

        if not candidates:
            missed_events.append(dict(gt_event))
            continue

        candidates.sort()
        onset_lag, _, _, pred_idx, overlap, overlap_ratio, offset_err = candidates[0]
        used_pred.add(pred_idx)
        pred_event = pred_events[pred_idx]
        detected_events.append({'gt_idx': int(gt_idx),
                                'pred_idx': int(pred_idx),
                                'gt_event': dict(gt_event),
                                'pred_event': dict(pred_event),
                                'onset_lag': float(onset_lag),
                                'offset_err': float(offset_err),
                                'overlap': float(overlap),
                                'overlap_ratio': float(overlap_ratio),})

    false_events = [dict(pred_event) for idx, pred_event in enumerate(pred_events) if idx not in used_pred]
    return detected_events, missed_events, false_events


def process_stream_inputs(test_res, **kwargs):
    """ Load and normalize raw stream results into one grouped analysis context.
    The returned context keeps the global arrays once and groups each source
    video/stream by sorted clip indices for later per-video analysis.
    """
    raw_res, res_path = _resolve_input(test_res)
    y_true, y_pred = _validate_raw_results(raw_res)

    required_meta = ('meta_video', 'meta_t_start', 'meta_t_end')
    missing = [key for key in required_meta if key not in raw_res]
    if missing:
        raise KeyError(f"Stream analysis requires meta fields: {missing}")

    meta_video = np.asarray(raw_res['meta_video'])
    meta_t_start = np.asarray(raw_res['meta_t_start'], dtype=np.float64)
    meta_t_end = np.asarray(raw_res['meta_t_end'], dtype=np.float64)
    if not (len(meta_video) == len(meta_t_start) == len(meta_t_end) == len(y_true)):
        raise ValueError("Stream metadata length mismatch")

    if 'meta_n_frames' in raw_res:
        meta_n_frames = np.asarray(raw_res['meta_n_frames'], dtype=np.int64)
        if len(meta_n_frames) != len(y_true):
            raise ValueError("meta_n_frames length mismatch")
    else:
        meta_n_frames = np.full(len(y_true), -1, dtype=np.int64)

    y_prob = raw_res.get('y_prob', None)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if len(y_prob) != len(y_true):
            raise ValueError(f"y_prob length mismatch: {len(y_prob)} vs {len(y_true)}")
        scores = y_prob
        score_name = 'y_prob'
    else:
        scores = np.asarray(y_pred, dtype=np.float64)
        score_name = 'y_pred'

    match_min_overlap = kwargs.get('match_min_overlap', STREAM_MATCH_MIN_OVERLAP)
    match_max_lag = kwargs.get('match_max_lag', STREAM_MATCH_MAX_LAG)
    max_event_gap = kwargs.get('max_event_gap', STREAM_MAX_EVENT_GAP)

    if match_min_overlap < 0:
        raise ValueError(f"match_min_overlap must be non-negative, got {match_min_overlap}")
    if match_max_lag is not None and match_max_lag < 0:
        raise ValueError(f"match_max_lag must be non-negative or None, got {match_max_lag}")
    if max_event_gap < 0:
        raise ValueError(f"max_event_gap must be non-negative, got {max_event_gap}")

    video_to_idx = {}
    for idx, video_name in enumerate(meta_video):
        video_to_idx.setdefault(str(video_name), []).append(idx)

    video_groups = []
    for video_name, idx in sorted(video_to_idx.items()):
        order = np.lexsort((meta_t_start[idx], meta_t_end[idx]))
        idx = np.asarray(idx, dtype=np.int64)[order]
        video_groups.append({'video': video_name, 'idx': idx})
    return {'raw_res': raw_res,
            'res_path': res_path,
            'y_true': y_true,
            'y_pred': y_pred,
            'scores': scores,
            'score_name': score_name,
            'meta_video': meta_video,
            'meta_t_start': meta_t_start,
            'meta_t_end': meta_t_end,
            'meta_n_frames': meta_n_frames,
            'match_min_overlap': match_min_overlap,
            'match_max_lag': match_max_lag,
            'max_event_gap': max_event_gap,
            'video_groups': video_groups,
            }


def analyze_single_stream(stream_ctx, video_group):
    """Analyze one grouped stream/video timeline and return its report payload.

    The returned payload contains per-video metrics, event lists, and the
    minimal timeline rows later written to CSV.
    """

    def _timeline_step():
        """ Infer the timeline step from clip end-times, with duration fallback."""
        times = np.asarray(t_frm, dtype=np.float64)
        if len(times) >= 2:
            diffs = np.diff(np.sort(times))
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                return float(np.median(diffs))

        t_star = np.asarray(win_t_start, dtype=np.float64)
        t_end = np.asarray(win_t_end, dtype=np.float64)
        if len(t_star) == len(t_end) and len(t_end) > 0:
            durations = t_end - t_star
            durations = durations[durations > 0]
            if len(durations) > 0:
                return float(np.median(durations))
        return 1.0

    def _build_window_timeline_rows():
        """Build minimal per-window timeline rows with no redundant derived columns."""
        rows = []
        for win_idx, (t_s, t_e, frm, gt_lbl, y_prb, y_prd) in enumerate(
                zip(win_t_start, win_t_end, win_n_frames, win_labels, win_scores, win_pred)):
            rows.append({'win_idx': win_idx, 't_frm': float(t_e), 't_start': float(t_s), 'n_frm': int(frm),
                         'gt_label': int(gt_lbl), 'y_prob': float(y_prb), 'y_pred': int(y_prd), })
        return rows

    idx = video_group['idx']
    video_name = video_group['video']
    win_labels = stream_ctx['y_true'][idx]
    win_pred = stream_ctx['y_pred'][idx]
    win_scores = stream_ctx['scores'][idx]
    win_t_start = stream_ctx['meta_t_start'][idx]
    win_t_end = stream_ctx['meta_t_end'][idx]
    win_n_frames = stream_ctx['meta_n_frames'][idx]
    t_frm = win_t_end
    match_min_overlap = stream_ctx['match_min_overlap']
    match_max_lag = stream_ctx['match_max_lag']
    max_event_gap = stream_ctx['max_event_gap']

    timeline_step = _timeline_step()
    cm_metrics = binary_metrics(win_labels, win_pred)
    cm_clips = _cm_dict(cm_metrics['confusion_matrix'])
    fp_time = cm_clips['fp'] * timeline_step
    miss_time = cm_clips['fn'] * timeline_step

    gt_events = _build_stream_events(win_labels == 1, t_frm, timeline_step, max_event_gap=max_event_gap)
    pred_events = _build_stream_events(win_pred == 1, t_frm, timeline_step, max_event_gap=max_event_gap)
    detected_events, missed_events, false_events = _match_stream_events(gt_events, pred_events,
                                                                         min_overlap=match_min_overlap, max_lag=match_max_lag)

    onset_delays = [match['onset_lag'] for match in detected_events]
    offset_errs = [abs(match['offset_err']) for match in detected_events]

    return {'video': video_name,
            'clips_num': len(win_labels),
            'cm_clips': cm_clips,
            'accuracy': cm_metrics.get('accuracy', None),
            'recall': cm_metrics.get('recall', None),
            'FPR': cm_metrics.get('FPR', None),
            'false_positive_time': fp_time,
            'miss_time': miss_time,
            'detection_lag': onset_delays[0] if onset_delays else None,
            'mean_onset_delay': np.mean(onset_delays) if onset_delays else None,
            'mean_offset_err': np.mean(offset_errs) if offset_errs else None,
            'gt_events_num': len(gt_events),
            'pred_events_num': len(pred_events),
            'detected_events_num': len(detected_events),
            'missed_events_num': len(missed_events),
            'false_events_num': len(false_events),
            'gt_events': gt_events,
            'pred_events': pred_events,
            'detected_events': detected_events, # 'matched_ events'
            'missed_events': missed_events,
            'false_events': false_events,
            'timeline_rows': _build_window_timeline_rows(),
            'onset_delays': onset_delays,
            'offset_errs': offset_errs,
            }


def get_stream_summary(stream_ctx, video_results, **kwargs):
    """Aggregate per-video stream reports into one stream-level summary."""
    y_true = stream_ctx['y_true']
    y_pred = stream_ctx['y_pred']
    scores = stream_ctx['scores']
    score_name = stream_ctx['score_name']
    raw_res = stream_ctx['raw_res']
    res_path = stream_ctx['res_path']
    match_min_overlap = stream_ctx['match_min_overlap']
    match_max_lag = stream_ctx['match_max_lag']
    max_event_gap = stream_ctx['max_event_gap']

    all_onset_delays = []
    all_offset_errs = []
    agg_cm = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    total_fp_time =  total_miss_time = 0.0
    total_gt_events = total_pred_events =  total_detect_events\
        = total_missed_events = total_false_events = 0

    for report in video_results:
        for key in agg_cm:
            agg_cm[key] += report['cm_clips'][key]
        total_fp_time += report['false_positive_time']
        total_miss_time += report['miss_time']
        total_gt_events += report['gt_events_num']
        total_pred_events += report['pred_events_num']
        total_detect_events += report['detected_events_num'] # matched_events_num
        total_missed_events += report['missed_events_num']
        total_false_events += report['false_events_num']
        all_onset_delays.extend(report['onset_delays'])
        all_offset_errs.extend(report['offset_errs'])

    summary = binary_metrics(y_true, y_pred)
    summary.update({'raw_results_path': str(res_path),
                    'raw_results_type': score_name,
                    'analysis_mode': 'stream',
                    'model_path': raw_res.get('model_path', None),
                    'test_cache': raw_res.get('test_cache', None),
                    'num_clips': len(y_true),
                    'support_clips': _support_counts(y_true),
                    'num_videos': len(video_results),
                    'cm_clips': agg_cm,
                    'false_positive_time': total_fp_time,
                    'miss_time': total_miss_time,
                    'detection_lag': all_onset_delays[0] if all_onset_delays else None,
                    'mean_onset_delay': np.mean(all_onset_delays) if all_onset_delays else None,
                    'mean_offset_err': np.mean(all_offset_errs) if all_offset_errs else None,
                    'gt_events_num': total_gt_events,
                    'pred_events_num': total_pred_events,
                    'detected_events': total_detect_events,
                    'missed_events': total_missed_events,
                    'false_events_num': total_false_events,
                    'match_config': {'min_overlap': match_min_overlap,
                                     'max_lag': match_max_lag,
                                     'max_event_gap': max_event_gap,},
                    })

    roc, roc_info = _roc_summary(y_true, scores, score_name,
                                 max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES))
    summary.update(roc_info)
    return summary, roc

# -----------------------------------------------------------------------
#* Main and API functions
# -----------------------------------------------------------------------

def analyze_clip_test(test_results:Path|str|dict, **kwargs): #52
    """ Analyze raw clip predictions and build one clip-level summary report.
    :param test_results: Raw prediction, either as a NPZ file or an in-memory raw-results dict.
    kwargs params:
    out_path:            Output summary path or output dir.
    output_name:         Stem for the output files, if given overrides the auto generated name.
    max_resolution:      Maximum resolution of  ROC curve .
    roc_csv:             If True, save the ROC data to CSV file
    show_roc:            If True, display the ROC figure.
    roc_title:           Optional custom title for the ROC plot.
    print:               If True, print the compact CLI report.
    :return:             Summary dict for clip-level evaluation.
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
                    'analysis_mode': 'clip',
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
    out_path = resolve_output_path(results_path, out_name, kwargs.get('out_path', None))
    summary['output_dir'] = str(out_path.parent) if out_path is not None else None
    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_csv'] = (out_path.with_suffix('.csv').name
                          if save_roc_csv and roc is not None and out_path is not None else
                          ('N-/A' if save_roc_csv in {False, None} else None))
    if out_path is not None:
        out_path = _save_analyze_summary(summary, out_path)
    else:
        print("[INFO] Analysis complete\n Summary file wasn't saved; (please provide out_path)")

    if roc is not None and (out_path is not None or bool(kwargs.get('show_roc', False))):
        plot_roc_curve(roc, save_to=out_path, save_csv=bool(save_roc_csv), show=bool(kwargs.get('show_roc', False)),
                       title=kwargs.get('roc_title', f"ROC Curve for {tst_name} ({score_name})"), )

    if kwargs.get('print', True):
        print_test_report(summary)
    return summary


def analyze_video_test(test_res:Path|str|dict, **kwargs): #100
    """ Analyze clip predictions at video level and build one video summary report.
    Videos with inconsistent GT labels across their clips are excluded with a
    warning before video-level metrics are computed.
    :param test_res:    Raw prediction, either as a NPZ file or an in-memory raw-results dict.
    kwargs options
    All the options as in  analyze_clip_test.
    pool_func:          Function used to pool clip predictions into one video prediction.
    min_val/threshold: Threshold for the default _min_clips_pool rule. (threshold is Legacy alias)
    :return:            Summary dict for clip-level evaluation.
    """
    #* Normalize arguments
    raw_res, res_path = _resolve_input(test_res)
    y_true, y_pred = _validate_raw_results(raw_res)

    if 'meta_video' not in raw_res or len(y_true) != len(y_pred):
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

    print_color(_support_counts(y_true), 'r')
    summary = binary_metrics(vid_true, vid_pred)
    summary.update({'raw_results_path': str(res_path) ,
                    'raw_results_type': 'y_prob' if y_prob is not None else 'y_pred',
                    'analysis_mode': 'video',
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
    out_path = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
    summary['output_dir'] = str(out_path.parent) if out_path is not None else None
    save_roc_csv = kwargs.get('roc_csv', True)
    summary['roc_csv'] = (out_path.with_suffix('.csv').name
                          if save_roc_csv and roc is not None and out_path is not None else
                          ('N/A' if save_roc_csv in {False, None} else None))
    if out_path is not None:
        out_path = _save_analyze_summary(summary, out_path)
    else:
        print("[WARN] Summary file wasn't saved; invalid or missing out_path")


    if roc is not None and (out_path is not None or bool(kwargs.get('show_roc', False))):
        plot_roc_curve(roc, save_to=out_path, save_csv=bool(save_roc_csv), show=bool(kwargs.get('show_roc', False)),
                            title=kwargs.get('roc_title', f"Video ROC for {tst_name}"),)

    if kwargs.get('print', True):
        print_test_report(summary)
    return summary


def analyze_stream_test(test_res:Path|str|dict, **kwargs): #282
    """ Analyze raw clip predictions as one or more temporal streams.
    The stream analysis groups clips by source video, builds GT/predicted event
    segments on each stream, and finally matches detected events to GT events, writes
    per-video timeline CSV files, and builds stream summary / events outputs.
    :param test_res:    Raw prediction, either as a NPZ file or an in-memory raw-results dict.
    kwargs options :
    All the options as in  analyze_clip_test.
    details_name:        Filename for the stream events JSON when it is saved.
    match_min_overlap:  Minimum GT-duration overlap ratio required for a detected event.
    match_max_lag:      Maximum allowed onset lag for matching a predicted event to GT.
    max_event_gap:      Maximum consecutive negative windows allowed inside one merged event.
    events_json:        If True, save the stream events JSON file.
    pool_func:          Function used to pool clip predictions into one video prediction.
    min_val:            Threshold for the default `_min_clips_pool` rule.
    :return:            Dict with the aggregate summary and the per-video reports.
    """
    csv_fieldnames = ['win_idx', 't_frm', 't_start', 'n_frm', 'gt_label', 'y_prob', 'y_pred']

    def _save_stream_timeline_csv(csv_path):
        """Write one per-video timeline CSV with only window-local columns."""
        csv_path = Path(csv_path).with_suffix('.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(timeline_rows)
        print_color(f"  Timeline CSV saved to  :{csv_path}", 'b')
        return csv_path

    stream_ctx = process_stream_inputs(test_res, **kwargs)
    raw_res = stream_ctx['raw_res']
    res_path = stream_ctx['res_path']

    video_reports = []
    for video_group in stream_ctx['video_groups']:
        report = analyze_single_stream(stream_ctx, video_group)
        video_reports.append(report)
    summary, roc = get_stream_summary(stream_ctx, video_reports, **kwargs)

    tst_name = res_path.stem if res_path is not None else 'stream_test'
    out_name = kwargs.get('output_name', f"{tst_name}_stream-summary.json")
    summary_path = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
    details_name = kwargs.get('details_name', f"{tst_name}_stream-events.json")
    save_events_json = kwargs.get('events_json', True)
    save_roc_csv = kwargs.get('roc_csv', True)

    timeline_csvs = []
    if summary_path is not None and save_events_json not in {False, None}:
        details_path = summary_path.with_name(details_name)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = str(summary_path.parent)
        for report in video_reports:
            video_name = report['video']
            video_tag = str(video_name).replace('/', '_').replace('\\', '_').replace(' ', '_')
            timeline_rows = report.pop('timeline_rows')
            report.pop('onset_delays', None)
            report.pop('offset_errs', None)
            timeline_csv_path = summary_path.with_name(f"{tst_name}_{video_tag}_timeline.csv")
            timeline_csv_path = _save_stream_timeline_csv(timeline_csv_path)
            report['timeline_csv'] = timeline_csv_path.name
            timeline_csvs.append(timeline_csv_path.name)

        summary['output_dir'] = output_dir
        summary['events_info'] = details_path.name
        summary['timeline_csvs'] = timeline_csvs
        details_payload = {'raw_results_path': str(res_path),
                           'model_path': raw_res.get('model_path', None),
                           'output_dir': output_dir,
                           'events_info': details_path.name,
                           'analysis_mode': 'stream',
                           'test_cache': raw_res.get('test_cache', None),
                           'match_config': summary['match_config'],
                           'videos': video_reports,}
        with details_path.open('w') as f:
            json.dump(serialize_json_data(details_payload), f, indent=2)
        print_color(f"  Stream events saved to :{details_path}", 'b')
    else:
        details_path = None
        summary['output_dir'] = str(summary_path.parent) if summary_path is not None else None
        summary['events_info'] = 'N/A' if save_events_json in {False, None} else None
        summary['timeline_csvs'] = []

    summary['roc_csv'] = (summary_path.with_suffix('.csv').name
                          if save_roc_csv and roc is not None and summary_path is not None else
                          ('N/A' if save_roc_csv in {False, None} else None))
    if summary_path is not None:
        _save_analyze_summary(summary, summary_path)
    else:
        print("[INFO] Analysis complete\n Summary file wasn't saved; (please provide out_path)")

    if roc is not None and (summary_path is not None or bool(kwargs.get('show_roc', False))):
        plot_roc_curve(roc, save_to=summary_path, save_csv=bool(save_roc_csv), show=bool(kwargs.get('show_roc', False)),
                       title=kwargs.get('roc_title', f"Stream ROC for {tst_name} ({stream_ctx['score_name']})"))

    if kwargs.get('print', True):
        print_test_report(summary)

    return {'summary': summary, 'videos': video_reports}

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the basic binary metrics and 2x2 confusion matrix."""

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
            #* Unique FPR grid is required for interpolation.
            uniq_fpr, uniq_idx = np.unique(fpr, return_index=True)
            uniq_tpr = tpr[uniq_idx]
            uniq_thr = thresholds[uniq_idx]

            #* Interpolate thresholds with finite values only, then restore edge sentinels.
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


def print_test_report(results, **kwargs):
    """ Print a compact aligned CLI view of one saved or in-memory summary.
    :param results:        Summary dict or path to a saved summary JSON file.
    :return:               The normalized summary dict that was printed.
    kwargs options :
    precision:             floating-point precision for printing.
    label_width:           Width of the left label column in the printed table.
    """
    def _fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{precision}f}"
        if isinstance(v, np.integer):
            return str(int(v))
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
    analysis_mode = summary.get('analysis_mode', None)

    cm:list|None = summary.get('confusion_matrix', None)
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

    rows = [("Predictions file", Path(summary.get('raw_results_path', '')).name ),
            ("Num_samples", num_samples),
            ("GT_counts 0/1", support_str),
            ("accuracy", summary.get('accuracy', None)),
            ("recall", summary.get('recall', None)),
            ("FPR", summary.get('FPR', None)),
            ("AUC", summary.get('roc_auc', summary.get('ROC AUC', None)))
            ]

    if analysis_mode == 'stream':
        rows.extend([
            ("False positive time", summary.get('false_positive_time', None)),
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


#548(5,3,2) #926(3,2,)-> 1060(5,3,)->992(3,3,)->973(3,4)->908(?)->944(2,2,)->949
# ->1027...975
if __name__ == '__main__':
    pass
    set_file = Path("work_dirs/json_models/draft/stream-tst_J-RWL_25ft_3w-1o5/tst-02.npz")
    # analyze_test_results('work_dirs/json_models/train_260323-0314_RWF_tms_f18/test_raw_model_260323-202938.npz')
    tst_file = Path("work_dirs/json_models/draft/stream-tst_J-RWL_25ft_3w-1o5/tst-10.npz")
    a = analyze_stream_test(set_file)
    from visual_util import draw_confusion_matrix
    cm = a['summary']['confusion_matrix']
    draw_confusion_matrix(cm)
    pass
#326(13,1,1)->999(3,3,)
