""" Timestamp-based metrics for one or more stream timeline CSV files."""
import math
from bisect import bisect_left
from statistics import median

MAX_EVENT_GAP = 1.5
K_MIN = 2
K_MAX = 4
ALARM_COST = 60.0
ALARM_HALF_RATE = 4.0
FP_WEIGHT = 0.4

#* region Public API  ---------------------------------------------------
# -----------------------------------------------------------------------
def resolve_metric_config(values=None, warn_missing=False) -> dict:
    """Resolve metric settings using core defaults and validation rules."""
    defaults = {'event_gap': MAX_EVENT_GAP,
                'k_min': K_MIN, 'k_max': K_MAX,
                'fp_cost': ALARM_COST,
                'half_rate': ALARM_HALF_RATE,
                'w_fp': FP_WEIGHT}
    values = values or {}
    if not isinstance(values, dict):
        print("[WARN] metric config section is invalid; using core constants")
        values = {}

    resolved = {}
    for key, default in defaults.items():
        if key not in values:
            if warn_missing:
                print(f"[WARN] metric config missing {key}; using {default}")
            resolved[key] = default
            continue
        try:
            value = values[key]
            value = int(value) if key in {'k_min', 'k_max'} else float(value)
            valid = ((key == 'event_gap' and value >= 0) or
                     (key in {'k_min', 'k_max'} and value > 0) or
                     (key == 'fp_cost' and value >= 0) or
                     (key == 'half_rate' and value > 0) or
                     (key == 'w_fp' and 0 <= value <= 1))
            if not valid:
                raise ValueError('value is outside the allowed range')
            resolved[key] = value
        except (TypeError, ValueError) as error:
            print(f"[WARN] invalid metric config {key}={values[key]!r}: {error}; "
                  f"using {default}")
            resolved[key] = default

    if resolved['k_min'] > resolved['k_max']:
        print("[WARN] metric config k_min is greater than k_max; using core constants")
        resolved['k_min'], resolved['k_max'] = K_MIN, K_MAX
    resolved['w_recall'] = 1.0 - resolved['w_fp']
    return resolved


def get_timeline_timing(rows: list[dict], metadata: dict | None = None) -> dict:
    """ Resolve FPS, window span, and inference frequency for timeline rows."""

    metadata = metadata or {}
    spans = [row['t_frm'] - row['t_start'] for row in rows  if row['t_frm'] > row['t_start']]
    if not spans:
        raise ValueError("timeline has no positive window spans")

    times = sorted(row['t_frm'] for row in rows)
    diffs = [right - left for left, right in zip(times, times[1:]) if right > left]
    win_span = metadata.get('win_span')
    frq_i = metadata.get('frq_i')
    fps = metadata.get('fps')

    win_span = float(win_span) if win_span is not None else median(spans)
    if frq_i is not None:
        frq_i = float(frq_i)
    elif metadata.get('infer_t') is not None:
        frq_i = 1.0/float(metadata['infer_t'])
    else:
        frq_i = 1.0/median(diffs) if diffs else None
    if fps is None:
        frame_spans = [(row.get('n_frm'), row['t_frm'] - row['t_start']) for row in rows]
        frame_spans = [(count, span) for count, span in frame_spans
                       if isinstance(count, (int, float)) and count > 0 and span > 0]
        fps = (sum(count for count, _ in frame_spans)/sum(span for _, span in frame_spans)
               if frame_spans else None)
    else:
        fps = float(fps)

    if win_span <= 0 or frq_i is None or frq_i <= 0 or fps is None or fps <= 0:
        raise ValueError("timeline FPS, window span, and inference frequency must be positive")
    return {'fps': fps, 'window_span': win_span, 'frq_i': frq_i}


def evaluate_stream(timeline: dict, pred_col=None, threshold=None, **kwargs) -> dict:
    """ Evaluate one already-loaded timeline dictionary."""
    pred_col, threshold, params = _resolve_eval_params(pred_col, threshold, kwargs)
    return _evaluate_timeline(timeline, pred_col, threshold, params)


def eval_multi_streams(timelines, pred_col=None, threshold=None, **kwargs) -> dict:
    """ Evaluate multiple loaded timelines and aggregate their reports."""

    pred_col, threshold, params = _resolve_eval_params(pred_col, threshold, kwargs)
    stream_reports = []
    for t_line in timelines:
        try:
            stream_reports.append(_evaluate_timeline(t_line, pred_col, threshold, params))
        except Exception as error:
            stream_reports.append({'file' : str(t_line.get('path', 'N/A')), 'status': 'fail',
                                   'error': f"{type(error).__name__}: {error}"})

    valid_streams = [report for report in stream_reports if report['status'] == 'pass']
    stream_events = [stream['events'] for stream in valid_streams]
    gt_events   = sum(event['gt']    for event in stream_events)
    full_events = sum(event['full']  for event in stream_events)
    half_events = sum(event['half']  for event in stream_events)
    fp_events   = sum(event['false'] for event in stream_events)
    t_events    = sum(event['duration']['total'] for event in stream_events)
    t_neg = sum(stream['time']['t_neg'] for stream in valid_streams)
    t_fp  = sum(stream['time']['t_fp']  for stream in valid_streams)
    t_tn = t_neg - t_fp
    longest_event = max((event['duration']['longest'] for event in stream_events), default=0.0)
    lag_total = sum(event['avg_lag']*(event['full'] + event['half'])
                    for event in stream_events if event['avg_lag'] is not None)

    det_count = full_events + half_events
    avg_lag = lag_total/det_count if det_count else None

    recall = (full_events + 0.5*half_events)/gt_events if gt_events else None
    if t_neg > 0:
        tn_rate = t_tn/t_neg
        fp_rate = fp_events/(t_neg/3600.0)
        cc = 2.0**(-fp_rate/params['half_rate']) #* count_credit
        fp_scr = tn_rate*cc
        fp_burden = (t_fp + params['fp_cost']*fp_events)/t_neg
        burden_scr = 1.0/(1.0 + fp_burden)
    else:
        tn_rate = fp_rate = cc = fp_scr = fp_burden = burden_scr = None

    if recall is None or fp_scr is None:
        total_scr = None
    elif recall == 0.0 or fp_scr == 0.0:
        total_scr = 0.0
    else:
        total_scr = 1.0/(params['w_recall']/recall + params['w_fp']/fp_scr)

    failed_num = len(stream_reports) - len(valid_streams)
    status = 'fail' if not valid_streams else 'partial' if failed_num else 'pass'
    return {'status': status,
            'prediction': {'column': pred_col, 'threshold': threshold},
            'scores': {'recall': recall,
                       'fp': fp_scr, 'fp_burden': fp_burden, 'burden_scr': burden_scr,
                       'total': total_scr},
            'time'  : {'total': sum(stream['time']['total'] for stream in valid_streams),
                       't_neg': t_neg, 't_fp': t_fp, 't_tn': t_tn},
            'events': {'gt': gt_events,
                       'predicted': sum(event['predicted'] for event in stream_events),
                       'full': full_events, 'half': half_events,
                       'false': fp_events, 'fp_per_h': fp_rate,
                       'duration': {'total': t_events, 'longest': longest_event},
                       'avg_lag': avg_lag},
            'components': {'tn_rate': tn_rate, 'count_credit': cc},
            'params' : params,
            'streams': stream_reports}


def eval_multi_thresholds(timelines, thresholds=None, pred_cols=None, **kwargs) -> list[dict]:
    """Evaluate the same timelines at multiple thresholds or prediction columns."""

    def selectors(values, name):
        if values is None:
            return []
        values = ([values] if isinstance(values, str) or not hasattr(values, '__iter__')
                  else list(values))
        if not values:
            raise ValueError(f"{name} can not be empty")
        values = ([str(value) for value in values] if name == 'pred_cols'
                  else [float(value) for value in values])
        return list(dict.fromkeys(values))

    thresholds = selectors(thresholds, 'thresholds')
    pred_cols  = selectors(pred_cols, 'pred_cols')
    if bool(thresholds) == bool(pred_cols):
        raise ValueError("provide exactly one of thresholds or pred_cols")
    if any(not 0.0 < th < 1.0 for th in thresholds):
        raise ValueError("thresholds must be between 0 and 1")

    timelines = list(timelines)
    if thresholds:
        return [eval_multi_streams(timelines, threshold=th, **kwargs) for th in thresholds]
    else:
        return [eval_multi_streams(timelines, pred_col=prd, **kwargs) for prd in pred_cols]

# endregion

#* region Helpers  ------------------------------------------------------
# -----------------------------------------------------------------------
def _resolve_eval_params(pred_col, threshold, kwargs):
    """ Validate prediction selection and resolve the shared metric parameters."""
    if (pred_col is None) == (threshold is None):
        raise ValueError("provide exactly one of pred_col or threshold")
    if pred_col is not None:
        pred_col = str(pred_col)
    else:
        threshold = float(threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

    params = resolve_metric_config(kwargs)
    return pred_col, threshold, params


def _evaluate_timeline(timeline, pred_col, threshold, params):
    """ Calculate metrics for one validated timeline dictionary."""

    rows = sorted(timeline['rows'], key=lambda row: row['t_frm'])
    required = {'t_frm', 't_start', 'gt_label', 'n_frm'}
    missing = required.difference(timeline['fieldnames'])
    if missing:
        raise KeyError(f"timeline columns missing: {sorted(missing)}")
    if pred_col is not None and pred_col not in timeline['fieldnames']:
        raise KeyError(f"timeline prediction column not found: {pred_col}")
    if threshold is not None and 'y_prob' not in timeline['fieldnames']:
        raise KeyError("timeline probability column not found: y_prob")

    times = [row['t_frm'] for row in rows]
    diffs = [right - left for left, right in zip(times, times[1:])]
    if any(diff <= 0 for diff in diffs):
        raise ValueError("timeline t_frm values must be unique and increasing")
    timing = get_timeline_timing(rows, timeline['metadata'])
    last_step = median(diffs) if diffs else 1.0/timing['frq_i']
    durations = diffs + [last_step]

    gt = [row['gt_label'] for row in rows]
    pred = ([row[pred_col] for row in rows] if pred_col is not None
            else [int(row['y_prob'] >= threshold) for row in rows])
    if any(value not in {0, 1} for value in gt):
        raise ValueError("gt_label values must be binary 0/1")
    if any(value not in {0, 1} for value in pred):
        raise ValueError("prediction values must be binary 0/1")

    gt_events = _build_events(gt, times, durations, params['event_gap'])
    pred_events = _build_events(pred, times, durations, params['event_gap'])
    gt_event_durations = [event['end'] - event['start'] for event in gt_events]
    k_frames = min(params['k_max'], max(params['k_min'],
                   math.floor(timing['window_span']*timing['fps']/2.0)))
    timeline_end = times[-1] + durations[-1]

    lags = []
    full_count = half_count = 0
    for gt_event in gt_events:
        nominal_boundary = gt_event['start'] + k_frames/timing['fps']
        boundary_idx = bisect_left(times, nominal_boundary)
        full_boundary = times[boundary_idx] if boundary_idx < len(times) else timeline_end
        miss_boundary = max(gt_event['start'] + timing['window_span'], full_boundary)
        credit = 0.0
        overlapping_starts = [pred_event['start'] for pred_event in pred_events
                              if _events_overlap(gt_event, pred_event)]
        for pred_event in pred_events:
            if not _events_overlap(gt_event, pred_event):
                continue
            if pred_event['start'] <= full_boundary:
                credit = 1.0
                break
            if pred_event['start'] <= miss_boundary:
                credit = max(credit, 0.5)
        full_count += credit == 1.0
        half_count += credit == 0.5
        if credit > 0 and overlapping_starts:
            lags.append(min(overlapping_starts) - gt_event['start'])

    fp_events = sum(not any(_events_overlap(pred_event, gt_event) for gt_event in gt_events)
                    for pred_event in pred_events)
    total_time = sum(durations)
    t_negative = sum(duration for duration, label in zip(durations, gt) if label == 0)
    t_fp = sum(duration for duration, gt_value, pred_value in zip(durations, gt, pred)
               if gt_value == 0 and pred_value == 1)
    t_tn = t_negative - t_fp
    fp_rate = fp_events/(t_negative/3600.0) if t_negative else None
    return {'file': str(timeline['path']),
            'stream': timeline['metadata'].get('source'),
            'status': 'pass', 'timing': timing,
            'time': {'total': total_time, 't_neg': t_negative, 't_fp': t_fp, 't_tn': t_tn},
            'events': {'gt': len(gt_events), 'predicted': len(pred_events),
                       'full': full_count, 'half': half_count,
                       'false': fp_events, 'fp_per_h': fp_rate,
                       'duration': {'total': sum(gt_event_durations),
                                    'longest': max(gt_event_durations, default=0.0)},
                       'avg_lag': sum(lags)/len(lags) if lags else None}}


def _build_events(mask, times, durations, max_gap):
    """ Build merged positive event intervals from timestamped row states."""
    events, active = [], None
    for positive, start, duration in zip(mask, times, durations):
        if not positive:
            continue
        end = start + duration
        if active is not None and start - active['end'] <= max_gap:
            active['end'] = max(active['end'], end)
        else:
            active = {'start': start, 'end': end}
            events.append(active)
    return events


def _events_overlap(first, second):
    """Return whether two event intervals have positive temporal overlap."""
    return min(first['end'], second['end']) > max(first['start'], second['start'])

# endregion
#315(,6,3) 279(2,,)
#multi-th: 322(4,,) -> 309(2,,)
