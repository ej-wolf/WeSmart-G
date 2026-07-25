"""Public workflows for prediction, timeline, and metric analysis."""
import json
from pathlib import Path
import yaml

from analysis_utils import (AUTO_META, attach_stream_meta, build_timelines,
                            load_report_file, load_timelines, print_metric_report,
                            print_report_table, print_test_report,
                            print_threshold_comparison,
                            resolve_stream_meta_path,
                            save_metric_report, save_timeline_csv)
from project_utils import get_test_title_lines
from stream_metric import eval_multi_thresholds, get_timeline_timing, resolve_metric_config


DEFAULT_METRIC_CONFIG = Path(__file__).resolve().parent/"configs/metrics/metric_config.yaml"


#* region Public API  ---------------------------------------------------
# -----------------------------------------------------------------------
def timelines_from_results(test_results) -> list[dict]:
    """Load raw prediction results and build one ordered timeline per stream."""
    from evaluation_core import resolve_input

    raw, _ = resolve_input(test_results)
    required = ('y_true', 'y_prob', 'meta_video', 'meta_t_start', 'meta_t_end')
    missing = [key for key in required if key not in raw or raw[key] is None]
    if missing:
        raise KeyError(f"stream timelines require raw-result fields: {missing}")

    timelines = build_timelines(raw['y_true'], raw['y_prob'], raw['meta_video'],
                                raw['meta_t_start'], raw['meta_t_end'],
                                raw.get('meta_n_frames'))
    model_tag, _ = get_test_title_lines(raw.get('model_path'), raw.get('test_cache'))
    for timeline in timelines:
        source = timeline['metadata']['source']
        timeline['metadata']['timeline'] = f"timeline_{model_tag}_{source}"
        timing = get_timeline_timing(timeline['rows'], timeline['metadata'])
        timeline['metadata'].update({'win_span': timing['window_span'],
                                     'fps': timing['fps'],
                                     'infer_t': 1.0/timing['frq_i'],
                                     'frq_i': timing['frq_i']})
    return timelines


def resolve_metric_params(config_path=None, values=None) -> dict:
    """Load optional YAML values and resolve metric parameters through the core."""
    if values is not None and not isinstance(values, dict):
        raise TypeError("metric parameter values must be a dictionary")

    config_path = Path(config_path) if config_path is not None else DEFAULT_METRIC_CONFIG
    loaded = {}
    if config_path.is_file():
        try:
            with config_path.open('r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
            if not isinstance(config, dict):
                print(f"[WARN] metric config is invalid: {config_path}")
            else:
                loaded = config.get('stream_metric', {})
                if not isinstance(loaded, dict):
                    print(f"[WARN] metric config section is invalid: {config_path}")
                    loaded = {}
        except Exception as error:
            print(f"[WARN] cannot read metric config {config_path}: {error}")
    else:
        print(f"[WARN] metric config not found: {config_path}")

    if values:
        loaded.update(values)
    return resolve_metric_config(loaded, warn_missing=True)


def analyze_timelines(timeline_input, thresholds=None, pred_cols=None, **kwargs):
    """Evaluate loaded or stored timelines at one or more operating points."""
    config_path = kwargs.pop('config_path', None)
    metric_values = kwargs.pop('metric_params', None)
    meta_info = kwargs.pop('meta_info', AUTO_META)
    output_path = kwargs.pop('output_path', None)
    print_results = bool(kwargs.pop('print_results', False))
    print_kwargs = kwargs.pop('print_kwargs', {})

    timelines, load_errors, timeline_files = load_timelines(timeline_input)
    if not timelines:
        details = load_errors[0]['error'] if load_errors else 'no timeline inputs'
        raise ValueError(f"no valid timelines were loaded: {details}")

    if thresholds is None and pred_cols is None:
        pred_cols = list(dict.fromkeys(
            column for timeline in timelines for column in timeline['fieldnames']
            if column == 'y_pred' or column.startswith('y_prd-')))
        if not pred_cols:
            raise ValueError("no prediction columns found; pass pred_cols or thresholds")

    params = resolve_metric_params(config_path, metric_values)
    reports = eval_multi_thresholds(timelines, thresholds=thresholds,
                                    pred_cols=pred_cols, **params)
    if load_errors:
        for report in reports:
            report['streams'].extend(dict(error) for error in load_errors)
            if report['status'] == 'pass':
                report['status'] = 'partial'

    meta_path = resolve_stream_meta_path(timeline_input, timeline_files, meta_info)
    reports = [attach_stream_meta(report, meta_path) for report in reports]
    result = reports[0] if len(reports) == 1 else reports

    if output_path is not None:
        save_metric_report(result, output_path)
    if print_results:
        if isinstance(result, list):
            print_threshold_comparison(result, **print_kwargs)
        else:
            print_metric_report(result, **print_kwargs)
    return result


def analyze_raw_results(test_results, mode='stream', **kwargs):
    """Analyze raw prediction results at clip, video, or stream level."""
    from evaluation_core import analyze_clip_test, analyze_video_test

    mode = str(mode).strip().lower()
    thresholds = kwargs.pop('thresholds', None)
    threshold = kwargs.pop('threshold', None)
    if thresholds is None and threshold is not None:
        thresholds = [threshold]
    elif thresholds is not None and not hasattr(thresholds, '__iter__'):
        thresholds = [thresholds]
    elif thresholds is not None:
        thresholds = list(thresholds)

    if mode in {'clip', 'video'}:
        output_path = kwargs.pop('output_path', None)
        print_results = kwargs.pop('print_results', None)
        for key in ('config_path', 'metric_params', 'meta_info', 'print_kwargs',
                    'pred_cols', 'timeline_dir', 'window_metrics'):
            kwargs.pop(key, None)
        if output_path is not None:
            kwargs['out_path'] = output_path
        if print_results is not None:
            kwargs['print'] = print_results

    if mode == 'clip':
        if not thresholds:
            return analyze_clip_test(test_results, **kwargs)
        reports = [analyze_clip_test(test_results, threshold=float(value), **kwargs)
                   for value in thresholds]
        return reports[0] if len(reports) == 1 else reports
    if mode == 'video':
        if not thresholds:
            return analyze_video_test(test_results, **kwargs)
        reports = [analyze_video_test(test_results, threshold=float(value), **kwargs)
                   for value in thresholds]
        return reports[0] if len(reports) == 1 else reports
    if mode != 'stream':
        raise ValueError("analysis mode must be 'clip', 'video', or 'stream'")

    pred_cols = kwargs.pop('pred_cols', None)
    timeline_dir = kwargs.pop('timeline_dir', None)
    window_metrics = bool(kwargs.pop('window_metrics', False))
    output_path = kwargs.pop('output_path', None)
    print_results = bool(kwargs.pop('print_results', False))
    print_kwargs = kwargs.pop('print_kwargs', {})
    plotting = kwargs.pop('plotting', False)
    timelines = timelines_from_results(test_results)

    timeline_files = []
    if timeline_dir is not None:
        timeline_dir = Path(timeline_dir)
        timeline_dir.mkdir(parents=True, exist_ok=True)
        for timeline in timelines:
            columns = {}
            for threshold in thresholds or []:
                name = f"y_prd-{int(round(float(threshold)*100))}"
                columns[name] = [int(row['y_prob'] >= float(threshold))
                                 for row in timeline['rows']]
            path = timeline_dir/f"{timeline['metadata']['timeline']}.csv"
            timeline_files.append(save_timeline_csv(timeline, path, columns))

    if plotting and timeline_files:
        from visual_util import plot_timeline
        plot_root = Path(timeline_dir)
        for threshold in thresholds or []:
            col_name = f"y_prd-{int(round(float(threshold)*100))}"
            plot_dir = plot_root/f"th-{int(round(float(threshold)*100))}"
            for timeline_path in timeline_files:
                plot_timeline(timeline_path, pred_column=col_name, threshold=float(threshold),
                              save_to=plot_dir/Path(timeline_path).with_suffix('.png').name,
                              show=False)

    stream_report = analyze_timelines(timelines, thresholds=thresholds,
                                      pred_cols=pred_cols, print_results=False,
                                      **kwargs)

    if output_path is not None:
        reports = stream_report if isinstance(stream_report, list) else [stream_report]
        output_path = Path(output_path)
        for report in reports:
            threshold_value = report.get('prediction', {}).get('threshold')
            if threshold_value is None:
                report_dir = output_path.parent
            else:
                report_dir = output_path.parent/f"th-{int(round(float(threshold_value)*100))}"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir/output_path.name
            events_path = report_dir/output_path.name.replace('_reports.json', '_events.json')
            with report_path.open('w', encoding='utf-8') as file:
                json.dump(report, file, indent=2)
            events = {'analysis_mode': 'stream_metric',
                      'prediction': report.get('prediction', {}),
                      'params': report.get('params', {}),
                      'streams': report.get('streams', [])}
            with events_path.open('w', encoding='utf-8') as file:
                json.dump(events, file, indent=2)

    if print_results:
        if isinstance(stream_report, list):
            print_threshold_comparison(stream_report, **print_kwargs)
        else:
            print_metric_report(stream_report, **print_kwargs)

    window_report = None
    if window_metrics:
        selected = list(thresholds or [])
        if not selected:
            raise ValueError("window_metrics requires at least one threshold")
        window_report = [analyze_clip_test(test_results, threshold=float(threshold))
                         for threshold in selected]
        if len(window_report) == 1:
            window_report = window_report[0]

    return {'metric': stream_report,
            'timelines': timelines,
            'timeline_files': timeline_files,
            'window_metrics': window_report}


def load_results(result_path):
    """Load a saved analysis report without recalculating it."""
    return load_report_file(result_path)


def print_results(result_path_or_report, **kwargs):
    """Load when needed and print one previously calculated report."""
    report = (load_report_file(result_path_or_report)
              if isinstance(result_path_or_report, (str, Path))
              else result_path_or_report)
    if isinstance(report, list):
        return print_threshold_comparison(report, **kwargs)
    if isinstance(report, dict) and 'streams' in report:
        return print_metric_report(report, **kwargs)
    if isinstance(report, dict) and {'fieldnames', 'rows'}.issubset(report):
        return print_report_table(report)
    if isinstance(report, dict):
        return print_test_report(report, **kwargs)
    raise ValueError("unsupported analysis report structure")


def plot_timeline(timeline_path, **kwargs):
    """Plot one existing timeline without running evaluation."""
    from visual_util import plot_timeline as render_timeline
    return render_timeline(timeline_path, **kwargs)

# endregion
