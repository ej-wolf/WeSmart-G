"""Reporting, persistence, and CLI helpers for stream metrics."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import yaml

from stream_metric import eval_multi_thresholds, resolve_metric_config
from json_stream_utils import DEFAULT_STREAM_META_PATH, SJ_META_INFO
from common.my_local_utils import _fmt


MAIN_DIR = Path(__file__).resolve().parent
DEFAULT_METRIC_CONFIG = MAIN_DIR/"configs/metrics/metric_config.yaml"


#* region Public API  ---------------------------------------------------
# -----------------------------------------------------------------------
def load_timeline_csv(csv_path: str | Path) -> dict:
    """Load a timeline CSV whose data table may follow a metadata block."""
    def clean_cell(value):
        return value.strip().lstrip('\ufeff').strip(" '\"")

    def scalar(value):
        value = clean_cell(value)
        try:
            return float(value)
        except ValueError:
            return value

    def typed_row(row):
        parsed = {}
        int_fields = {'win_idx', 'n_frm', 'gt_label', 'y_true'}
        float_fields = {'t_frm', 't_start', 'y_prob'}
        for k, v in row.items():
            if k in int_fields or k == 'y_pred' or k.startswith('y_prd-'):
                parsed[k] = int(float(v))
            elif k in float_fields:
                parsed[k] = float(v)
            else:
                parsed[k] = v
        return parsed

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"timeline CSV not found: {csv_path}")

    metadata, rows = {}, [] # metadata_rows = []
    fieldnames = None
    with csv_path.open('r', newline='', encoding='utf-8-sig') as f:
        for values in csv.reader(f):
            if not values or not any(value.strip() for value in values):
                continue
            cells = [clean_cell(value) for value in values]
            if fieldnames is None and {'t_frm', 't_start'}.issubset(cells):
                fieldnames = cells
                continue
            if fieldnames is None:
                # metadata_rows.append(values)
                if len(cells) >= 2 and cells[0]:
                    key = clean_cell(cells[0])
                    key = 'frq_i' if key == 'infer_frq' else key
                    metadata[key] = scalar(cells[1])
                continue

            if len(cells) < len(fieldnames):
                cells.extend([''] * (len(fieldnames) - len(cells)))
            rows.append(typed_row(dict(zip(fieldnames, cells))))

    if fieldnames is None:
        raise ValueError(f"timeline data header not found in {csv_path}")
    if not rows:
        raise ValueError(f"timeline CSV contains no data rows: {csv_path}")
    return {'path': csv_path,
            'metadata': metadata, # 'metadata_rows': metadata_rows,
            'fieldnames': fieldnames, 'rows': rows}


def load_metric_config(config_path=None) -> dict:
    """Load YAML settings and delegate defaults and validation to the core."""
    config_path = Path(config_path) if config_path is not None else DEFAULT_METRIC_CONFIG
    values = {}
    if config_path.is_file():
        try:
            with config_path.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            if not isinstance(config, dict):
                print(f"[WARN] metric config is invalid: {config_path}")
            else:
                values = config.get('stream_metric', {})
                if not isinstance(values, dict):
                    print(f"[WARN] metric config section is invalid: {config_path}")
                    values = {}
        except Exception as error:
            print(f"[WARN] cannot read metric config {config_path}: {error}")
    else:
        print(f"[WARN] metric config not found: {config_path}")
    return resolve_metric_config(values, warn_missing=True)


_AUTO_META = object()


def _resolve_stream_meta_path(stream_path, timeline_files, meta_info):
    if meta_info is not _AUTO_META:
        return None if meta_info is None else Path(meta_info)

    source = Path(stream_path) if not isinstance(stream_path, (list, tuple, set, frozenset)) else None
    if source is not None and source.is_dir():
        path = source / SJ_META_INFO
        if path.is_file():
            return path
    parents = {Path(path).parent for path in timeline_files}
    if len(parents) == 1:
        path = next(iter(parents)) / SJ_META_INFO
        if path.is_file():
            return path
    return DEFAULT_STREAM_META_PATH if DEFAULT_STREAM_META_PATH.is_file() else None


def _load_stream_meta(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if not path.is_file():
        print(f'[WARN] stream metadata not found: {path}')
        return []
    try:
        with path.open('r', encoding='utf-8') as file:
            payload = json.load(file)
        records = payload.get('streams', []) if isinstance(payload, dict) else []
        return records if isinstance(records, list) else []
    except Exception as error:
        print(f'[WARN] cannot read stream metadata {path}: {error}')
        return []


def _attach_stream_meta(result: dict, meta_path: Path | None) -> dict:
    def stem(value) -> str:
        name = Path(str(value)).name
        if name.lower().endswith('.json.zip'):
            return name[:-len('.json.zip')]
        return Path(name).stem

    def find_meta(stream: dict, records: list[dict]) -> dict | None:
        stream_stem = stem(stream.get('stream') or stream.get('file', ''))
        candidates = [record for record in records
                      if isinstance(record, dict) and record.get('stem') == stream_stem]
        stream_fps = stream.get('timing', {}).get('fps')
        if stream_fps is not None:
            matching = []
            for record in candidates:
                try:
                    if record.get('fps') is not None and math.isclose(
                            float(record['fps']), float(stream_fps),
                            rel_tol=1e-3, abs_tol=0.01):
                        matching.append(record)
                except (TypeError, ValueError):
                    continue
            candidates = matching
        # TODO: use yolo_threshold to disambiguate records when that setting
        # becomes part of the stream metric configuration.
        return candidates[0] if len(candidates) == 1 else None

    if meta_path is None:
        return result

    records = _load_stream_meta(meta_path)
    for stream in result.get('streams', []):
        stream.pop('stream_meta', None)
    result.pop('stream_meta', None)

    matched = []
    for stream in result.get('streams', []):
        if stream.get('status') != 'pass':
            continue
        metadata = find_meta(stream, records)
        if metadata is not None:
            stream['stream_meta'] = metadata
            matched.append(metadata)

    if matched:
        result['stream_meta'] = {
            'path': str(meta_path),
            'matched': len(matched),
            'person_dets': sum(int(item.get('person_dets', 0)) for item in matched),
            'max_dets_frame': max(int(item.get('max_dets_frame', 0)) for item in matched),
            'consecutive_det_frames': max(
                int(item.get('consecutive_det_frames',
                             item.get('max_consecutive_det_frames', 0)))
                for item in matched),
        }
    else:
        result['stream_meta'] = {'path': str(meta_path), 'matched': 0}
    return result


def run_stream_metrics(stream_path, pred_col=None, threshold=None, **kwargs) -> dict:
    """Find, load, and evaluate one or more timeline CSV files."""
    reports = run_multi_thresholds( stream_path,
              thresholds=None if threshold is None else [threshold],
              pred_cols=None if pred_col is None else [pred_col],
              **kwargs)
    return reports[0]


def run_multi_thresholds(stream_path, thresholds=None, pred_cols=None, **kwargs) -> list[dict]:
    """Load timelines once and evaluate multiple prediction operating points."""
    def input_files(value):
        values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        files = []
        for item in values:
            path = Path(item)
            if path.is_dir():
                files.extend(sorted(path.glob('timeline_*.csv')))
            else:
                files.append(path)
        return list(dict.fromkeys(files))

    config_path = kwargs.pop('config_path', None)
    meta_info = kwargs.pop('meta_info', _AUTO_META)
    timeline_files = input_files(stream_path)
    metric_params = load_metric_config(config_path)
    timelines, load_errors = [], []
    for csv_path in timeline_files:
        try:
            timelines.append(load_timeline_csv(csv_path))
        except Exception as error:
            load_errors.append({'file': str(csv_path), 'status': 'fail',
                                'error': f"{type(error).__name__}: {error}"})

    reports = eval_multi_thresholds(timelines, thresholds=thresholds, pred_cols=pred_cols,
                                    **metric_params)
    if load_errors:
        for report in reports:
            report['streams'].extend(dict(error) for error in load_errors)
            if report['status'] == 'pass':
                report['status'] = 'partial'
    meta_path = _resolve_stream_meta_path(stream_path, timeline_files, meta_info)
    return [_attach_stream_meta(report, meta_path) for report in reports]


def print_stream_metric(report: dict, **kwargs):
    """Print an aggregate stream-metric report and optional per-file details."""
    results_table = bool(kwargs.get('results_table', False))
    total_row = bool(kwargs.get('total_row', False))
    meta_info = kwargs.get('meta_info')
    meta_path = Path(meta_info) if meta_info is not None else None
    if meta_path is not None:
        _attach_stream_meta(report, meta_path)
    show_meta = meta_path is not None and meta_path.is_file()
    fp_unit = 'min' if kwargs.get('fp_unit', 'h') in {'min', 'minute'} else 'h'

    def fmt_duration(val):
        if  val is None:
            return 'N/A'
        elif val <= 60.0:
            return f'{val:.2f}'

        tenths = round(val*10.0)
        if tenths < 36000:
            m, s_tenths = divmod(tenths, 600)
            return f'{m:02d}:{s_tenths//10:02d}.{s_tenths % 10}'

        s_total = tenths//10 if val < 3600.0 else int(val)
        h, m_rem = divmod(s_total, 3600)
        m, s = divmod(m_rem, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    def stream_name(stream):
        source = stream.get('stream')
        if source:
            return str(source)[:30]
        name = Path(stream.get('file', 'N/A')).stem
        name = name.removeprefix('timeline_')
        match = re.match(r'.+?_W[^_]+_(.+)$', name)
        return (match.group(1) if match else name)[:30]

    def false_rate(stream):
        rate = stream.get('events', {}).get('fp_per_h')
        if rate is None:
            return 'N/A'
        decimals = 1 if fp_unit == 'h' else 2
        rate = rate if fp_unit == 'h' else rate/60.0
        return f'{rate:.{decimals}f}'

    def meta_value(stream, name):
        value = stream.get('stream_meta', {}).get(name)
        return 'N/A' if value is None else str(value)

    def false_detection_rate(stream):
        dets = stream.get('stream_meta', {}).get('person_dets')
        if dets is None or dets <= 0:
            return 'N/A'
        return f'{stream.get("events", {}).get("false", 0) * 1000/dets:.2f}'

    prediction = report.get('prediction', {})
    scores = report.get('scores', {})
    time_info = report.get('time', {})
    events = report.get('events', {})
    multi_thresholds = bool(kwargs.get('multi_thresholds', False))
    det = events['full'] + events['half']
    mis = events['gt'] - det

    print("\n=== Stream Metric ===")
    print(f"Status:      {report.get('status', 'N/A')}")
    print(f"Prediction:  {prediction.get('column') or 'y_prob'},  threshold={prediction.get('threshold')}")
    print(f"Duration:    Total = {fmt_duration(time_info['total'])}, GT events = {fmt_duration(events['duration']['total'])}")
    if not multi_thresholds:
        print(f"Events:      GT = {events.get('gt', 'N/A')};  detected = {det};  missed = {mis};"
              f"  false = {events.get('false', 'N/A')}")
        print(f"Recall:      {_fmt(scores.get('recall'))}")
        print(f"FP:          burden = {_fmt(scores.get('fp_burden'))};  score = {_fmt(scores.get('fp'))}")
        print(f"Total score: {_fmt(scores.get('total'))}")


    if not results_table:
        return report

    streams = report.get('streams', [])
    print("\n=== Streams ===")
    headers = ['Status', 'Stream', 'Total(s)', 'GT dur', 'Longest']
    if show_meta:
        headers += ['P dets', 'Max P/f', 'Max run']
    headers += ['GT', 'Detected', 'Lag (s)', 'Missed', 'False', f'False/{fp_unit}']
    if show_meta:
        headers += ['False/P (1k)']
    table = []

    for stream in streams:
        stream_events = stream.get('events', {})
        event_duration = stream_events.get('duration', {})
        if stream_events:
            detections = stream_events['full'] + stream_events['half']
            missed = stream_events['gt'] - detections
        else:
            detections = missed = 'N/A'
        row = [stream.get('status', 'N/A'),
               stream_name(stream),
               fmt_duration(stream.get('time', {}).get('total')),
               fmt_duration(event_duration.get('total')),
               fmt_duration(event_duration.get('longest')),]
        if show_meta:
            row += [meta_value(stream, 'person_dets'),
                    meta_value(stream, 'max_dets_frame'),
                    meta_value(stream, 'consecutive_det_frames')]
        row += [str(stream_events.get('gt', 'N/A')),
                str(detections),
                _fmt(stream_events.get('avg_lag'), d=2),
                str(missed),
                str(stream_events.get('false', 'N/A')),
                false_rate(stream),]
        if show_meta:
            row += [false_detection_rate(stream)]
        table.append(tuple(row))

    if total_row:
        event_duration = events.get('duration', {})
        row = ['Total', str(len(table)),
                fmt_duration(time_info.get('total')),
                fmt_duration(event_duration.get('total')),
                fmt_duration(event_duration.get('longest')),]
        if show_meta:
            row += [meta_value(report, 'person_dets'),
                    meta_value(report, 'max_dets_frame'),
                    meta_value(report, 'consecutive_det_frames')]
        row += [str(events.get('gt', 'N/A')),
                str(det),
                _fmt(events.get('avg_lag'), d=2),
                str(mis),
                str(events.get('false', 'N/A')),
                ('N/A' if events.get('fp_per_h') is None else
                 f"{(events['fp_per_h'] if fp_unit == 'h' else events['fp_per_h']/60.0):.{1 if fp_unit=='h' else 2}f}"),
                ]
        if show_meta:
            row+= ['N/A' if not report.get('stream_meta', {}).get('person_dets') else
                       f"{events.get('false', 0)*1000/report['stream_meta']['person_dets']:.2f}"]
        table.append(tuple(row))

    widths = [max(5, len(hd)) for hd in headers]
    for row in table:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    alignments = ['<', '<', '>', '>', '>']
    if show_meta:
        alignments += ['^', '^', '^']
    alignments += ['^']*7
    print(' | '.join(f'{header:{align}{width}}'
                     for header, align, width in zip(headers, alignments, widths)))
    print('-+-'.join('-' * width for width in widths))
    for row in table:
        if total_row and row is table[-1]:
            print('-+-'.join('-' * width for width in widths))
        print(' | '.join(f'{v:{a}{w}}' for v, a, w in zip(row, alignments, widths)))
    return report


def print_multi_thresholds(reports: list[dict], **kwargs):
    """Print optional per-stream tables followed by a threshold comparison."""
    fp_unit = 'min' if kwargs.get('fp_unit', 'h') in {'min', 'minute'} else 'h'

    def prediction_name(report):
        prediction = report.get('prediction', {})
        if prediction.get('column') is not None:
            return str(prediction['column'])
        threshold = prediction.get('threshold')
        return 'N/A' if threshold is None else f'th={threshold:g}'

    def event_counts(report):
        events = report.get('events', {})
        detected = events.get('full', 0) + events.get('half', 0)
        return detected, events.get('gt', 0) - detected

    def false_rate(report):
        rate = report.get('events', {}).get('fp_per_h')
        if rate is None:
            return None
        return rate if fp_unit == 'h' else rate/60.0

    def false_per_detections(report):
        detections = report.get('stream_meta', {}).get('person_dets')
        if not detections:
            return None
        return report.get('events', {}).get('false', 0)*1000/detections

    reports = list(reports)
    if kwargs.get('results_table', False):
        kwargs['multi_thresholds'] = True
        for index, report in enumerate(reports):
            if index:
                print()
            print_stream_metric(report, **kwargs)

    show_meta = any(report.get('stream_meta', {}).get('person_dets') is not None
                    for report in reports)
    headers = ['Prediction', 'Recall', 'FP score', 'FP burden', 'Score',
               'Detected', 'Missed', 'False', f'False/{fp_unit}']
    if show_meta:
        headers.append('False/P (1k)')

    table = []
    for report in reports:
        scores = report.get('scores', {})
        events = report.get('events', {})
        detected, missed = event_counts(report)
        fp_rate = false_rate(report)
        row = [prediction_name(report),
               _fmt(scores.get('recall')),
               _fmt(scores.get('fp')),
               _fmt(scores.get('fp_burden')),
               _fmt(scores.get('total')),
               str(detected), str(missed), str(events.get('false', 'N/A')),
               ('N/A' if fp_rate is None else
                f'{fp_rate:.{1 if fp_unit == "h" else 2}f}')]
        if show_meta:
            row.append(_fmt(false_per_detections(report)))
        table.append(row)

    print("\n=== Multi-Threshold Summary ===")
    widths = [max(5, len(header)) for header in headers]
    for row in table:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    alignments = ['<'] + ['>']*(len(headers) - 1)
    print(' | '.join(f'{header:{align}{width}}'
                     for header, align, width in zip(headers, alignments, widths)))
    print('-+-'.join('-'*width for width in widths))
    for row in table:
        print(' | '.join(f'{value:{align}{width}}'
                         for value, align, width in zip(row, alignments, widths)))
    return reports


def save_stream_metric(report: dict | list[dict], output_path) -> Path:
    """Save a stream-metric report as JSON and return the written path."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.json':
        output_path = output_path / 'stream_metric.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return output_path


# endregion


#* region CLI  ----------------------------------------------------------
# -----------------------------------------------------------------------
def main():
    """Run stream metric evaluation from the command line."""
    def parse_meta_info(value):
        return None if value.lower() == 'none' else Path(value)

    parser = argparse.ArgumentParser( description='Evaluate timeline (CSV files) with stream metrics')
    parser.add_argument('stream_path', type=Path, help='Timeline CSV, directory, or list is not supported by CLI')

    prediction = parser.add_mutually_exclusive_group(required=True)
    prediction.add_argument('--pred-col', nargs='+', help='One or more existing binary prediction columns')
    prediction.add_argument('-th','--threshold', nargs='+', type=float, help='One or more thresholds applied to y_prob')

    parser.add_argument('-o', '--output', type=Path,help='JSON output file or directory')
    parser.add_argument('-c', '--config', type=Path, default=None, help='Metric YAML configuration file')
    parser.add_argument('-tbl', '--results-table', action='store_true', help='Print the per-stream results table')
    parser.add_argument('-tr' , '--total-row', action='store_true', help='Print an aggregate row in the per-stream table')
    parser.add_argument('--fp-unit' , choices=('h', 'min'), default='h', help='False-positive rate time unit')
    parser.add_argument('--meta-info', nargs='?', type=parse_meta_info, const=None, default=DEFAULT_STREAM_META_PATH,
                        help='Streams meta info path; use without a value to disable metadata columns')
    args = parser.parse_args()

    selectors = args.threshold if args.threshold is not None else args.pred_col
    if len(selectors) == 1:
        report = run_stream_metrics(args.stream_path,
                            pred_col = args.pred_col[0] if args.pred_col is not None else None,
                            threshold= args.threshold[0] if args.threshold is not None else None,
                            config_path=args.config, meta_info=args.meta_info)
        print_stream_metric(report, results_table=args.results_table, total_row=args.total_row,
                            fp_unit=args.fp_unit, meta_info=args.meta_info)
    else:
        report = run_multi_thresholds(args.stream_path,
                            thresholds=args.threshold, pred_cols=args.pred_col,
                            config_path=args.config, meta_info=args.meta_info)
        print_multi_thresholds(report, results_table=args.results_table,
                               total_row=args.total_row, fp_unit=args.fp_unit,
                               meta_info=args.meta_info)
    if args.output is not None:
        print(f"Saved: {save_stream_metric(report, args.output)}")


# endregion
#561(,14,1) ->530(,13,1)

if __name__ == '__main__':
    main()
