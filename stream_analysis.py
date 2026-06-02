from pathlib import Path
import csv
import json
import numpy as np

from common.my_local_utils import print_color, serialize_json_data, resolve_output_path
from project_utils import get_exporting_name, get_test_title_lines
from visual_util import plot_timeline
from evaluation_core import (DEFAULT_ROC_RES, DEFAULT_EVAL_THRESHOLD, PRINT_POLICY, DEFAULT_THRESHOLD_RANGE,
                             cm_dict, companion_summary_name, get_eval_arrays, resolve_input, roc_summary,
                             save_analyze_summary, support_counts,
                             analyze_clip_predictions, analyze_clip_scores, binary_metrics, iter_thresholds,
                             resolve_threshold_dir, resolve_unique_output_file, require_prob_scores, )

STREAM_MATCH_MIN_OVERLAP = 1e-9
STREAM_MATCH_MAX_LAG = None
STREAM_MAX_EVENT_GAP = 1
DEFAULT_REG_MODEL = Path("work_dirs/json_models/win-study/260414-1721_J-RWL_25ft_3w-1o5-stream-tst/best_model.148.pt")


def build_stream_events(mask, t_frm, step, max_event_gap=STREAM_MAX_EVENT_GAP):
    """ Merge consecutive positive timeline rows into event segments."""
    mask = np.asarray(mask, dtype=bool)
    t_frm = np.asarray(t_frm, dtype=np.float64)
    events = []
    start_idx, last_pos_idx = None, None
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
                           'end'  : float(t_frm[end_idx] + step),
                           't_frm_start': float(t_frm[start_idx]),
                           't_frm_end'  : float(t_frm[end_idx]),
                           'idx_start': int(start_idx),
                           'idx_end': int(end_idx),
                           })
            start_idx, last_pos_idx = None, None
            gap_count = 0

    if start_idx is not None:
        end_idx = last_pos_idx
        events.append({'start':float(t_frm[start_idx]),
                       'end' : float(t_frm[end_idx] + step),
                       't_frm_start': float(t_frm[start_idx]),
                       't_frm_end': float(t_frm[end_idx]),
                       'idx_start': int(start_idx),
                       'idx_end': int(end_idx),
                       })
    return events #46


def event_overlap(gt_event, pred_event):
    """Return overlap duration and overlap ratio normalized by GT duration."""
    start = max(float(gt_event['start']), float(pred_event['start']))
    end = min(float(gt_event['end']), float(pred_event['end']))
    overlap = max(0.0, end - start)
    gt_dur = max(float(gt_event['end']) - float(gt_event['start']), 1e-12)
    return overlap, overlap / gt_dur


def match_stream_events(gt_events, pred_events, min_overlap, max_lag):
    """Greedy match of predicted stream events to GT events."""
    used_pred = set()
    detected_events, missed_events = [], []

    for gt_idx, gt_event in enumerate(gt_events):
        candidates = []
        for pred_idx, pred_event in enumerate(pred_events):
            if pred_idx in used_pred:
                continue
            overlap, overlap_ratio = event_overlap(gt_event, pred_event)
            if overlap <= 0 or overlap_ratio < min_overlap:
                continue

            onset_lag = float(pred_event['start']) - float(gt_event['start'])
            if max_lag is not None and onset_lag > max_lag:
                continue

            offset_err = float(pred_event['end']) - float(gt_event['end'])
            candidates.append((abs(onset_lag), -overlap_ratio, abs(offset_err), pred_idx, overlap, overlap_ratio, offset_err))

        if not candidates:
            missed_events.append(dict(gt_event))
            continue

        candidates.sort()
        onset_lag, _, _, pred_idx, overlap, overlap_ratio, offset_err = candidates[0]
        used_pred.add(pred_idx)
        pred_event = pred_events[pred_idx]
        detected_events.append({
            'gt_idx': int(gt_idx),
            'pred_idx': int(pred_idx),
            'gt_event': dict(gt_event),
            'pred_event': dict(pred_event),
            'onset_lag': float(onset_lag),
            'offset_err': float(offset_err),
            'overlap': float(overlap),
            'overlap_ratio': float(overlap_ratio),
        })

    false_events = [dict(pred_event) for idx, pred_event in enumerate(pred_events) if idx not in used_pred]
    return detected_events, missed_events, false_events


def process_stream_inputs(test_res, threshold: float, **kwargs):
    """Load and normalize raw stream results into one grouped analysis context."""
    raw_res, res_path = resolve_input(test_res)
    y_true, y_pred, scores, score_name = get_eval_arrays(raw_res, threshold=threshold)

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
            'threshold': float(threshold),
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
    """ Analyze one grouped stream/video timeline and return its report payload."""

    def _timeline_step():
        """ Infer the timeline step from clip end-times, with duration fallback."""
        times = np.asarray(t_frm, dtype=np.float64)
        if len(times) >= 2:
            diffs = np.diff(np.sort(times))
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                return float(np.median(diffs))

        t_start = np.asarray(win_t_start, dtype=np.float64)
        t_end = np.asarray(win_t_end, dtype=np.float64)
        if len(t_start) == len(t_end) and len(t_end) > 0:
            durations = t_end - t_start
            durations = durations[durations > 0]
            if len(durations) > 0:
                return float(np.median(durations))
        return 1.0

    def _build_window_timeline_rows():
        """Build minimal per-window timeline rows."""
        rows = []
        for win_idx, (t_s, t_e, frm, gt_lbl, y_prb, y_prd) in enumerate(
                      zip(win_t_start, win_t_end, win_n_frames, win_labels, win_scores, win_pred)):
            rows.append({'win_idx': win_idx,
                         't_frm' : float(t_e),
                         't_start': float(t_s),
                         'n_frm'  : int(frm),
                         'gt_label': int(gt_lbl),
                         'y_prob': float(y_prb),
                         'y_pred': int(y_prd),
                         })
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

    timeline_step = _timeline_step()
    cm_metrics = binary_metrics(win_labels, win_pred)
    cm_clips = cm_dict(cm_metrics['confusion_matrix'])
    fp_time = cm_clips['fp'] * timeline_step
    miss_time = cm_clips['fn'] * timeline_step

    gt_events = build_stream_events(win_labels == 1, t_frm, timeline_step, max_event_gap=stream_ctx['max_event_gap'])
    pred_events = build_stream_events(win_pred == 1, t_frm, timeline_step, max_event_gap=stream_ctx['max_event_gap'])
    detected_events, missed_events, false_events = match_stream_events(
                                                        gt_events, pred_events,
                                                        min_overlap=stream_ctx['match_min_overlap'],
                                                        max_lag=stream_ctx['match_max_lag'],
                                                        )
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
            'detected_events': detected_events,
            'missed_events': missed_events,
            'false_events': false_events,
            'timeline_rows': _build_window_timeline_rows(),
            'onset_delays': onset_delays,
            'offset_errs': offset_errs,
            }


def _build_stream_event_summary(stream_ctx, video_results):
    """Aggregate per-video stream reports into one threshold-dependent stream summary."""
    all_onset_delays = []
    all_offset_errs = []
    agg_cm = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    total_fp_time = total_miss_time = 0.0
    total_gt_events = total_pred_events = total_detect_events = 0
    total_missed_events = total_false_events = 0

    for report in video_results:
        for key in agg_cm:
            agg_cm[key] += report['cm_clips'][key]
        total_fp_time += report['false_positive_time']
        total_miss_time += report['miss_time']
        total_gt_events += report['gt_events_num']
        total_pred_events += report['pred_events_num']
        total_detect_events += report['detected_events_num']
        total_missed_events += report['missed_events_num']
        total_false_events += report['false_events_num']
        all_onset_delays.extend(report['onset_delays'])
        all_offset_errs.extend(report['offset_errs'])

    summary = binary_metrics(stream_ctx['y_true'], stream_ctx['y_pred'])
    summary.update({'raw_results_path': str(stream_ctx['res_path']),
                    'raw_results_type': stream_ctx['score_name'],
                    'analysis_mode': 'stream',
                    'model_path': stream_ctx['raw_res'].get('model_path', None),
                    'test_cache': stream_ctx['raw_res'].get('test_cache', None),
                    'num_clips' : len(stream_ctx['y_true']),
                    'support_clips': support_counts(stream_ctx['y_true']),
                    'num_videos': len(video_results),
                    'cm_clips': agg_cm,
                    'false_positive_time': total_fp_time,
                    'miss_time': total_miss_time,
                    'detection_lag': all_onset_delays[0] if all_onset_delays else None,
                    'mean_onset_delay': np.mean(all_onset_delays) if all_onset_delays else None,
                    'mean_offset_err': np.mean(all_offset_errs) if all_offset_errs else None,
                    'gt_events_num'  : total_gt_events,
                    'pred_events_num': total_pred_events,
                    'detected_events': total_detect_events,
                    'missed_events'  : total_missed_events,
                    'false_events_num':total_false_events,
                    'match_config': {'min_overlap': stream_ctx['match_min_overlap'],
                                     'max_lag': stream_ctx['match_max_lag'],
                                     'max_event_gap': stream_ctx['max_event_gap'], },
                    'analysis_config':{'threshold': float(stream_ctx['threshold']),
                                       'match_min_overlap': stream_ctx['match_min_overlap'],
                                       'match_max_lag': stream_ctx['match_max_lag'],
                                       'max_event_gap': stream_ctx['max_event_gap'], },
                    })
    return summary


def get_stream_summary(stream_ctx, video_results, **kwargs):
    """Aggregate per-video stream reports and attach clip/window ROC info."""
    summary = _build_stream_event_summary(stream_ctx, video_results)
    roc, roc_info = roc_summary(stream_ctx['y_true'], stream_ctx['scores'], stream_ctx['score_name'],
                                max_resolution=kwargs.get('max_resolution', DEFAULT_ROC_RES), )
    summary.update(roc_info)
    return summary, roc


def _event_metrics(detected_events, missed_events, false_events):
    """Return event precision/recall/F1 from aggregated event counts."""
    detected_events = int(detected_events)
    missed_events = int(missed_events)
    false_events = int(false_events)
    precision = detected_events/(detected_events + false_events ) if (detected_events + false_events ) > 0 else 0.0
    recall    = detected_events/(detected_events + missed_events) if (detected_events + missed_events) > 0 else 0.0
    f1 = 2.0*precision*recall/(precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _stream_objective(row: dict, mode='event_f1') -> tuple[float, tuple]:
    """ Return one stream optimization objective value and tie-break key."""
    if mode == 'event_f1':
        objective = float(row['event_f1'])
        tie_key = (objective, float(row['event_recall']), -float(row['false_events']), float(row['threshold']))
        return objective, tie_key
    if mode == 'sensitivity_first':
        objective = float(row['event_recall'])
        tie_key = (objective, -float(row['false_events']), float(row['event_f1']), float(row['threshold']))
        return objective, tie_key
    if mode == 'low_false_alarms':
        objective = -float(row['false_events'])
        tie_key = (objective, float(row['event_recall']), float(row['event_f1']), float(row['threshold']))
        return objective, tie_key
    raise ValueError(f"Unsupported stream optimization mode: {mode}")


def optimize_stream_threshold(test_results: Path | str | dict, threshold_range=DEFAULT_THRESHOLD_RANGE, mode='event_f1', **kwargs):
    """ Sweep thresholds for stream event analysis and return the best operating point."""
    raw_res, res_path = resolve_input(test_results)
    require_prob_scores(raw_res)
    thresholds = iter_thresholds(threshold_range)

    rows = []
    for threshold in thresholds:
        stream_ctx = process_stream_inputs(test_results, threshold=float(threshold), **kwargs)
        video_reports = []
        for video_group in stream_ctx['video_groups']:
            video_reports.append(analyze_single_stream(stream_ctx, video_group))
        summary = _build_stream_event_summary(stream_ctx, video_reports)
        event_precision, event_recall, event_f1 = _event_metrics(
                                                        summary.get('detected_events', 0),
                                                        summary.get('missed_events', 0),
                                                        summary.get('false_events_num', 0),
                                                                )
        row = {'threshold': float(threshold),
               'accuracy': float(summary.get('accuracy', 0.0)),
               'recall': float(summary.get('recall', 0.0)),
               'FPR': float(summary.get('FPR', 0.0)),
               'detected_events': int(summary.get('detected_events', 0)),
               'missed_events': int(summary.get('missed_events', 0)),
               'false_events': int(summary.get('false_events_num', 0)),
               'pred_events': int(summary.get('pred_events_num', 0)),
               'gt_events': int(summary.get('gt_events_num', 0)),
               'event_precision': float(event_precision),
               'event_recall': float(event_recall),
               'event_f1': float(event_f1),
               'false_positive_time': float(summary.get('false_positive_time', 0.0)),
               'miss_time': float(summary.get('miss_time', 0.0)),
               'num_videos': int(summary.get('num_videos', 0)),
               'support_clips': support_counts(stream_ctx['y_true']),
               }
        objective, tie_key = _stream_objective(row, mode=mode)
        row['objective'] = float(objective)
        row['_tie_key'] = tie_key
        rows.append(row)

    best_row = max(rows, key=lambda r: r['_tie_key'])
    for row in rows:
        row.pop('_tie_key', None)

    return {'raw_results_path': str(res_path) if res_path is not None else None,
            'analysis_mode': 'stream',
            'optimization_mode': mode,
            'threshold_range': [float(v) for v in thresholds],
            'model_path': raw_res.get('model_path', None),
            'test_cache': raw_res.get('test_cache', None),
            'best_threshold': best_row['threshold'],
            'best_objective': best_row['objective'],
            'best_metrics': {k: serialize_json_data(v) for k, v in best_row.items() if k != 'objective'},
            'results_table': serialize_json_data(rows),
            }


def run_stream_event_analysis(stream_ctx) -> tuple[list[dict], dict]:
    """Run per-video stream analysis and build the aggregated summary."""
    video_reports = []
    for video_group in stream_ctx['video_groups']:
        video_reports.append(analyze_single_stream(stream_ctx, video_group))
    summary = _build_stream_event_summary(stream_ctx, video_reports)
    return video_reports, summary


def export_stream_timeline(report, summary_path, raw_res, *, overwrite: bool) -> Path:
    """ Export stream timeline CSV and annotate the report with its filename."""

    csv_fieldnames = ['win_idx', 't_frm', 't_start', 'n_frm', 'gt_label', 'y_prob', 'y_pred']
    video_name = report['video']
    video_tag = str(video_name).replace('/', '_').replace('\\', '_').replace(' ', '_')
    mdl_tag, _ = get_test_title_lines(raw_res.get('model_path', None), raw_res.get('test_cache', None))
    timeline_rows = report.pop('timeline_rows')
    timeline_csv_path = summary_path.with_name(f"timeline_{mdl_tag}_{video_tag}.csv").with_suffix('.csv')
    timeline_csv_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_csv_path = resolve_unique_output_file(timeline_csv_path, overwrite=overwrite)

    with timeline_csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(timeline_rows)
    report['timeline_csv'] = timeline_csv_path.name
    return timeline_csv_path


def analyze_stream_events(test_res:Path|str|dict, **kwargs):
    """ Analyze thresholded clip/window predictions as stream events and timelines only.
    ToDo:   fix this functions , resolve_unique_output_file,  etc..."""

    plotting_mode = kwargs.get('plotting', None)
    overwrite = kwargs.get('overwrite', False)
    threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))

    if plotting_mode not in {None, 'save', 'disp'}:
        raise ValueError("plotting must be one of: None, 'save', 'disp'")

    def _plot_stream_timeline(csv_path:str):
        """ Optionally render the saved timeline CSV."""
        if plotting_mode is None:
            return None
        if plotting_mode == 'save':
            png_path = resolve_unique_output_file(Path(csv_path).with_name(f"timeline_{mdl_tag}_{video_tag}.png"), overwrite=overwrite)
            plot_timeline(csv_path, save_to=png_path, show=False, threshold=threshold,
                          print_policy=kwargs.get('print_policy', PRINT_POLICY))
            return png_path.name
        plot_timeline(csv_path, show=True, threshold=threshold,
                      print_policy=kwargs.get('print_policy', PRINT_POLICY))
        return None

    ctx_kwargs = dict(kwargs)
    ctx_kwargs.pop('threshold', None)
    stream_ctx = process_stream_inputs(test_res, threshold=threshold, **ctx_kwargs)
    res_path = stream_ctx['res_path']
    raw_res  = stream_ctx['raw_res']
    model_path = raw_res.get('model_path', None)
    test_cache = raw_res.get('test_cache', None)
    mdl_tag, _ = get_test_title_lines(model_path, test_cache)

    video_reports, summary = run_stream_event_analysis(stream_ctx)

    tst_name = res_path.stem if res_path is not None else 'stream_test'
    out_name = kwargs.get('output_name', get_exporting_name(model_path, test_cache, 'summary', unit='stream'))
    out_base = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
    roc_base = (out_base.parent/get_exporting_name(model_path, test_cache, 'roc',
                                                   unit=kwargs.get('roc_mode_code', 'stm'), short=True)
                if out_base is not None else None)
    thr_dir = (resolve_threshold_dir(out_base.parent, threshold, overwrite=overwrite, threshold_dir=kwargs.get('threshold_dir', None))
               if out_base is not None else None)
    summary_path = thr_dir/out_base.with_suffix('.json').name if out_base is not None and thr_dir is not None else None

    details_name = kwargs.get('details_name', f"{get_exporting_name(model_path, test_cache, 'events')}.json")

    timeline_csvs, timeline_plots = [], []
    if kwargs.get('events_json', True) not in {False, None} and summary_path is not None:
        details_path = resolve_unique_output_file(summary_path.with_name(details_name), overwrite=overwrite)
        details_path.parent.mkdir(parents=True, exist_ok=True)

        for report in video_reports:
            video_name = report['video']
            video_tag = str(video_name).replace('/', '_').replace('\\', '_').replace(' ', '_')
            report.pop('onset_delays', None)
            report.pop('offset_errs' , None)
            timeline_csv_path = export_stream_timeline(report, summary_path, raw_res, overwrite=overwrite)
            timeline_plot_name = _plot_stream_timeline(timeline_csv_path)
            report['timeline_plot'] = timeline_plot_name
            timeline_csvs.append(timeline_csv_path.name)
            if timeline_plot_name is not None:
                timeline_plots.append(timeline_plot_name)

        summary['output_dir'] = str(out_base.parent)
        summary['threshold_dir'] = thr_dir.name
        summary['events_info'] = details_path.name
        summary['timeline_csvs'] = timeline_csvs
        summary['timeline_plots'] = timeline_plots
        details_payload = {'raw_results_path': str(res_path),
                           'model_path': model_path,
                           'output_dir': str(out_base.parent),
                           'threshold_dir': thr_dir.name,
                           'events_info': details_path.name,
                           'analysis_mode': 'stream',
                           'test_cache': test_cache,
                           'analysis_config': summary['analysis_config'],
                           'match_config': summary['match_config'],
                           'videos': video_reports,
                           }
        with details_path.open('w') as f:
            json.dump(serialize_json_data(details_payload), f, indent=2)
    else:
        summary['output_dir'] = str(out_base.parent) if out_base is not None else None
        summary['threshold_dir'] = thr_dir.name if thr_dir is not None else None
        summary['events_info'] = 'N/A'
        summary['timeline_csvs'] = []
        summary['timeline_plots'] = []

    if kwargs.get('roc_csv', True) and roc_base is not None:
        summary['roc_csv'] = roc_base.with_suffix('.csv').name
    else:
        summary['roc_csv'] = 'N/A'

    if summary_path is not None:
        save_analyze_summary(summary, summary_path, overwrite=overwrite,
                              print_policy=kwargs.get('print_policy', PRINT_POLICY))
    else:
        print("[INFO] Analysis complete\n Summary file wasn't saved; (please provide out_path)")

    if kwargs.get('print', True):
        from evaluation_cli import print_test_report
        print_test_report(summary)

    return {'summary': summary, 'videos': video_reports}


def analyze_stream_test(test_res:Path|str|dict, **kwargs):
    """ Run clip/window score analysis, clip predictions, and stream event analysis."""
    raw_res, _ = resolve_input(test_res)
    out_name = kwargs.get('output_name',
                          get_exporting_name(raw_res.get('model_path', None), raw_res.get('test_cache', None),
                                                   'summary', unit='stream'))
    threshold_dir = kwargs.get('threshold_dir', None)
    if threshold_dir is None:
        _, res_path = resolve_input(test_res)
        if res_path is not None:
            out_base = resolve_output_path(res_path, out_name, kwargs.get('out_path', None))
            if out_base is not None:
                threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))
                threshold_dir = resolve_threshold_dir(out_base.parent, threshold,
                                                     overwrite=bool(kwargs.get('overwrite', False)))
    score_kwargs = dict(kwargs)
    score_kwargs['print'] = False
    score_kwargs['roc_mode_code'] = 'stm'
    analyze_clip_scores(test_res, **score_kwargs)

    clip_kwargs = dict(kwargs)
    clip_kwargs['print'] = False
    clip_kwargs['output_name'] = companion_summary_name(out_name, 'clip')
    clip_kwargs['roc_mode_code'] = 'stm'
    if threshold_dir is not None:
        clip_kwargs['threshold_dir'] = threshold_dir
    analyze_clip_predictions(test_res, **clip_kwargs)

    event_kwargs = dict(kwargs)
    event_kwargs['roc_mode_code'] = 'stm'
    if threshold_dir is not None:
        event_kwargs['threshold_dir'] = threshold_dir
    return analyze_stream_events(test_res, **event_kwargs)


def run_regression_suite(phase='refactor', **kwargs):
    """ Run stream regression outputs from cache NPZ files."""
    from torch_clip_model import run_testing

    model_path = Path(kwargs.get('model_path', DEFAULT_REG_MODEL))
    out_root = Path(kwargs.get('out_root', "work_dirs/json_models/testing")) / phase / 'stream_analysis'
    show_roc = bool(kwargs.get('show_roc', False))
    save_roc_csv = bool(kwargs.get('roc_csv', True))
    save_events_json = bool(kwargs.get('events_json', True))
    threshold = float(kwargs.get('threshold', DEFAULT_EVAL_THRESHOLD))

    cases = (('F_141_0_0_0_0', Path("data/cache/stream-tst/F_141_0_0_0_0.npz"), 'F_141_raw'),
             ('cam6_11_5_y26', Path("data/cache/stream-tst/cam6_11_5_y26.npz"), 'cam6_11_5_y26_raw'), )
    outputs = {}
    for case_name, cache_path, raw_tag in cases:
        out_dir = out_root/case_name
        out_dir.mkdir(parents=True, exist_ok=True)
        res = run_testing(model_path, cache_path, out_dir=out_dir, output_tag=raw_tag, video_mode=True)
        report = analyze_stream_test(res['path'], out_path=out_dir, show_roc=show_roc, roc_csv=save_roc_csv,
                                     events_json=save_events_json, print=False, threshold=threshold, )
        outputs[case_name] = {'raw_results': res['path'], 'summary': report['summary']}
    return outputs
#628(3,3,)
