""" Reporting and persistence helpers for stream metrics."""
import re
import csv, json, yaml
import math
from numbers import Integral, Real
from pathlib import Path

#* project imports
from stream_metric import eval_multi_thresholds, resolve_metric_config
from json_stream_utils import DEFAULT_STREAM_META, SJ_META_INFO, stream_stem
from common.my_local_utils import _fmt, get_unique_name


MAIN_DIR = Path(__file__).resolve().parent
DEFAULT_METRIC_CONFIG = MAIN_DIR/"configs/metrics/metric_config.yaml"


#* region Public API  ---------------------------------------------------
# -----------------------------------------------------------------------
def load_timeline_csv(csv_path: str | Path) -> dict:
    """ Load a timeline CSV whose data table may follow a metadata block."""
    def clean_cell(val):
        return val.strip().lstrip('\ufeff').strip(" '\"")

    def scalar(val):
        val = clean_cell(val)
        try:
            return float(val)
        except ValueError:
            return val

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

    metadata, rows = {}, [] # metadata_rows = []
    fieldnames = None

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"timeline CSV not found: {csv_path}")
    metadata['timeline'] = csv_path.stem

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

    return {'metadata': metadata, 'fieldnames': fieldnames, 'rows': rows} # 'metadata_rows': metadata_rows,


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
    metric_params  = load_metric_config(config_path)
    timelines, load_errors = [], []
    for csv_path in timeline_files:
        try:
            timelines.append(load_timeline_csv(csv_path))
        except Exception as error:
            load_errors.append({'timeline': csv_path.stem, 'stream': None, 'status': 'fail',
                                'error': f"{type(error).__name__}: {error}"})

    if thresholds is None and pred_cols is None:
        pred_cols = list(dict.fromkeys(
            column for timeline in timelines for column in timeline['fieldnames']
            if column == 'y_pred' or column.startswith('y_prd-')))
        if not pred_cols:
            raise ValueError("no prediction columns found; pass --pred-col or --threshold")

    reports = eval_multi_thresholds(timelines, thresholds=thresholds, pred_cols=pred_cols, **metric_params)
    if load_errors:
        for report in reports:
            report['streams'].extend(dict(error) for error in load_errors)
            if report['status'] == 'pass':
                report['status'] = 'partial'
    meta_path = _resolve_stream_meta_path(stream_path, timeline_files, meta_info)
    return [_attach_stream_meta(report, meta_path) for report in reports]


def save_stream_metric(report: dict | list[dict], output_path) -> Path:
    """Save a stream-metric report as CSV or JSON and return its path."""
    def save_csv(path):
        def full_stream_name(stream):
            source = stream.get('stream')
            if source:
                return str(source)
            name = str(stream.get('timeline', 'N/A')).removeprefix('timeline_')
            match = re.match(r'.+?_W[^_]+_(.+)$', name)
            return match.group(1) if match else name

        def format_cell(field, value):
            if not isinstance(value, Real) or isinstance(value, Integral):
                return value
            time_field = (field.startswith('t_') or
                          field.endswith(('_span', '_duration', '_lag')) or
                          field in {'longest_gt', 'event_gap', 'fp_cost'})
            return f'{value:.{2 if time_field else 3}f}'

        reports = report if isinstance(report, list) else [report]
        rows = []
        for result in reports:
            prediction = result.get('prediction', {})
            for stream in result.get('streams', []):
                timing = stream.get('timing', {})
                time_info = stream.get('time', {})
                events = stream.get('events', {})
                duration = events.get('duration', {})
                scores = stream.get('scores', {})
                meta = stream.get('stream_meta', {})
                rows.append({'stream': full_stream_name(stream),
                             'fps': timing.get('fps', ''),
                             'win_span': timing.get('window_span', ''),
                             'infer_frq': timing.get('frq_i', ''),
                             'threshold': ('' if prediction.get('threshold') is None
                                           else prediction['threshold']),
                             'y_pred': prediction.get('column') or '',
                             'status': stream.get('status', 'N/A'),
                             'timeline_file': str(stream.get('timeline', '')),
                             'src_span': meta.get('duration', ''),
                             'frames_count': meta.get('frames', ''),
                             'yolo_threshold': meta.get('yolo_threshold', ''),
                             'yolo_dets': meta.get('person_dets', ''),
                             'max_dets_frm': meta.get('max_dets_frame', ''),
                             'consc_det_frms': meta.get('consecutive_det_frames', ''),
                             't_total': time_info.get('total', ''),
                             'gt_events': events.get('gt', ''),
                             'gt_duration': duration.get('total', ''),
                             'longest_gt': duration.get('longest', ''),
                             'pred_events': events.get('predicted', ''),
                             'det_full': events.get('full', ''),
                             'det_half': events.get('half', ''),
                             'false': events.get('false', ''),
                             'recall': scores.get('recall', ''),
                             'avg_lag': events.get('avg_lag', ''),
                             't_fp': time_info.get('t_fp', ''),
                             't_tn': time_info.get('t_tn', ''),
                             'fp_per_h': events.get('fp_per_h', ''),
                             'fp_burden': scores.get('fp_burden', ''),
                             'notes': stream.get('error', '')})

        metadata = []
        passed = [row for row in rows if row['status'] == 'pass']
        for field in ('win_span', 'infer_frq'):
            values = [row[field] for row in passed]
            common = bool(values) and all(
                value != '' and math.isclose(float(value), float(values[0]),
                                             rel_tol=1e-6, abs_tol=1e-6)
                for value in values)
            if common:
                metadata.append((field, values[0]))
                for row in rows:
                    row.pop(field)

        params = reports[0].get('params', {}) if reports else {}
        metadata += list(params.items())
        fields = ['stream', 'fps']
        fields += [field for field in ('win_span', 'infer_frq') if field in rows[0]] if rows else []
        fields += ['threshold', 'y_pred', 'status', 'timeline_file',
                   'src_span', 'frames_count', 'yolo_threshold', 'yolo_dets',
                   'max_dets_frm', 'consc_det_frms', 't_total',
                   'gt_events', 'gt_duration', 'longest_gt',
                   'pred_events', 'det_full', 'det_half', 'false', 'recall', 'avg_lag',
                   't_fp', 't_tn', 'fp_per_h', 'fp_burden', 'notes']

        def sort_key(row):
            fps = row['fps']
            threshold = row['threshold']
            return (str(row['stream']).casefold(),
                    fps == '', float(fps) if fps != '' else math.inf,
                    threshold == '', float(threshold) if threshold != '' else math.inf,
                    str(row['y_pred']), str(row['timeline_file']))

        rows.sort(key=sort_key)
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['parameter', 'value'])
            writer.writerows((field, format_cell(field, value)) for field, value in metadata)
            writer.writerow([])
            table = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
            table.writeheader()
            table.writerows({field: format_cell(field, value) for field, value in row.items()}
                            for row in rows)

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if output_path.is_dir():
        output_path = output_path / 'stream_metric.csv'
        suffix = '.csv'
    elif not suffix:
        output_path = output_path.with_suffix('.csv')
        suffix = '.csv'
    elif suffix not in {'.csv', '.json'}:
        raise ValueError(f"unsupported stream metric output format: {suffix}")
    output_path = get_unique_name(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == '.csv':
        save_csv(output_path)
    else:
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    return output_path

# endregion


#* region Printing  ---------------------------------------------------
# -----------------------------------------------------------------------
def _fmt_duration(value):
    if value is None:
        return 'N/A'
    if value <= 60.0:
        return f'{value:.2f}'

    tenths = round(value*10.0)
    if tenths < 36000:
        minutes, seconds = divmod(tenths, 600)
        return f'{minutes:02d}:{seconds//10:02d}.{seconds % 10}'

    total_seconds = tenths//10 if value < 3600.0 else int(value)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _stream_name(stream):
    source = stream.get('stream')
    if source:
        return str(source)[:30]
    name = str(stream.get('timeline', 'N/A')).removeprefix('timeline_')
    match = re.match(r'.+?_W[^_]+_(.+)$', name)
    return (match.group(1) if match else name)[:30]


def _metric_values(data, fp_unit, lag_digits=2):
    events = data.get('events', {})
    duration = events.get('duration', {})
    scores = data.get('scores', {})
    meta = data.get('stream_meta', {})
    detected = (events.get('full', 0) + events.get('half', 0)
                if events else 'N/A')
    fp_rate = events.get('fp_per_h')
    if fp_rate is not None:
        fp_rate = fp_rate if fp_unit == 'h' else fp_rate/60.0
        fp_rate = f'{fp_rate:.{1 if fp_unit == "h" else 2}f}'
    person_dets = meta.get('person_dets')
    fp_per_det = ('N/A' if person_dets is None or person_dets <= 0 else
                  f'{events.get("false", 0)*1000/person_dets:.2f}')
    return {'stream': _stream_name(data),
            'total': _fmt_duration(data.get('time', {}).get('total')),
            'gt_dur': _fmt_duration(duration.get('total')),
            'longest': _fmt_duration(duration.get('longest')),
            'p_dets': 'N/A' if person_dets is None else str(person_dets),
            'max_p': 'N/A' if meta.get('max_dets_frame') is None else str(meta['max_dets_frame']),
            'max_run': ('N/A' if meta.get('consecutive_det_frames') is None else
                        str(meta['consecutive_det_frames'])),
            'gt': str(events.get('gt', 'N/A')),
            'detected': str(detected),
            'false': str(events.get('false', 'N/A')),
            'recall': _fmt(scores.get('recall'), d=2),
            'lag': _fmt(events.get('avg_lag'), d=lag_digits),
            'fp_burden': _fmt(scores.get('fp_burden'), d=2),
            'fp_rate': 'N/A' if fp_rate is None else fp_rate,
            'fp_per_det': fp_per_det}


def _print_table(headers, rows, alignments, separators=()):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(5, len(header)) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    separator = '-+-'.join('-'*width for width in widths)
    print(' ' + ' | '.join(f'{header:{align}{width}}'
                           for header, align, width in zip(headers, alignments, widths)))
    print(' ' + separator)
    for index, row in enumerate(rows):
        if index in separators:
            print(' ' + separator)
        print(' ' + ' | '.join(f'{value:{align}{width}}'
                               for value, align, width in zip(row, alignments, widths)))


def print_stream_metric(report: dict, **kwargs):
    """ Print an aggregate stream-metric report and optional per-file details."""
    results_table = bool(kwargs.get('results_table', False))
    total_row = bool(kwargs.get('total_row', False))
    meta_info = kwargs.get('meta_info')
    meta_path = Path(meta_info) if meta_info is not None else None
    if meta_path is not None:
        _attach_stream_meta(report, meta_path)
    show_meta = meta_path is not None and meta_path.is_file()
    fp_unit = 'min' if kwargs.get('fp_unit', 'h') in {'min', 'minute'} else 'h'

    prediction = report.get('prediction', {})
    scores = report.get('scores', {})
    time_info = report.get('time', {})
    events = report.get('events', {})
    multi_thresholds = bool(kwargs.get('multi_thresholds', False))
    det = events['full'] + events['half']
    mis = events['gt'] - det
    streams = report.get('streams', [])
    passed_streams = [stream for stream in streams if stream.get('status') == 'pass']
    failed_count = len(streams) - len(passed_streams)

    print("\n=== Stream Metric ===")
    print(f"Status:     pass [{len(passed_streams)}]   failed [{failed_count}]")
    print(f"Prediction:  {prediction.get('column') or 'y_prob'},  threshold={prediction.get('threshold')}")
    print(f"Duration:    Total = {_fmt_duration(time_info['total'])}, GT events = {_fmt_duration(events['duration']['total'])}")
    if not multi_thresholds:
        print(f"Events:      GT = {events.get('gt', 'N/A')};  detected = {det};  missed = {mis};"
              f"  false = {events.get('false', 'N/A')}")
        print(f"Recall:      {_fmt(scores.get('recall'))}")
        print(f"FP:          burden = {_fmt(scores.get('fp_burden'))};  score = {_fmt(scores.get('fp'))}")
        print(f"Total score: {_fmt(scores.get('total'))}")

    if not results_table:
        return report

    print("\n=== Streams ===")
    headers = ['Stream', 'Total(s)', 'GT dur', 'Longest']
    if show_meta:
        headers += ['P dets', 'Max P/f', 'Max run']
    headers += ['GT', 'Detected', 'False', 'Recall', 'Lag (s)', 'FP burden', f'False/{fp_unit}']
    if show_meta:
        headers += ['False/P (1k)']

    table = []
    for stream in passed_streams:
        values = _metric_values(stream, fp_unit)
        row = [values['stream'], values['total'], values['gt_dur'], values['longest']]
        if show_meta:
            row += [values['p_dets'], values['max_p'], values['max_run']]
        row += [values['gt'], values['detected'], values['false'], values['recall'],
                values['lag'], values['fp_burden'], values['fp_rate']]
        if show_meta:
            row += [values['fp_per_det']]
        table.append(tuple(row))

    if total_row:
        values = _metric_values(report, fp_unit)
        row = [f'Total [{len(table)}]', values['total'], values['gt_dur'], values['longest']]
        if show_meta:
            row += [values['p_dets'], values['max_p'], values['max_run']]
        row += [values['gt'], values['detected'], values['false'], values['recall'],
                values['lag'], values['fp_burden'], values['fp_rate']]
        if show_meta:
            row += [values['fp_per_det']]
        table.append(tuple(row))

    alignments = ['<', '>', '>', '>']
    if show_meta:
        alignments += ['^', '^', '^']
    alignments += ['^']*(7 + int(show_meta))
    _print_table(headers, table, alignments,
                 separators={len(table) - 1} if total_row else ())
    return report


def print_multi_thresholds(reports: list[dict], **kwargs):
    """Print multi-threshold summaries using the selected table layout."""
    fp_unit = 'min' if kwargs.get('fp_unit', 'h') in {'min', 'minute'} else 'h'

    def print_thrs_cmp():
        selector_header = ('Prediction' if any(rep.get('prediction', {}).get('column') for rep in reports)
                                        else 'Threshold')
        grouped, stream_order = {}, []
        for rep_i, rep in enumerate(reports):
            for stream in rep.get('streams', []):
                key = str(stream.get('timeline', 'N/A'))
                if key not in grouped:
                    grouped[key] = {}
                    stream_order.append(key)
                grouped[key][rep_i] = stream

        passed_keys = [key for key in stream_order if len(grouped[key]) == len(reports)
                                                   and  all(stream.get('status') == 'pass' for stream in grouped[key].values())]
        failed_count = len(stream_order) - len(passed_keys)
        show_stream_meta = any( stream.get('stream_meta', {}).get('person_dets') is not None
                                for key in passed_keys for stream in grouped[key].values())

        headers = ['Stream', selector_header, 'Total(s)', 'GT dur']
        if show_stream_meta:
            headers.append('P dets')
        headers += ['GT', 'Detected', 'False', 'Recall', 'Lag(s)', 'FP burden', f'False/{fp_unit}']
        if show_stream_meta:
            headers.append('False/P')

        tbl, separators = [], set()
        for key in passed_keys:
            if tbl:
                separators.add(len(tbl))
            by_report = grouped[key]
            base_stream = next((stream for stream in by_report.values()
                                if stream.get('status') == 'pass'),
                               next(iter(by_report.values())))
            base_values = _metric_values(base_stream, fp_unit, lag_digits=1)
            for rep_i, rep in enumerate(reports):
                stream = by_report.get(rep_i)
                pred = rep.get('prediction', {})
                selector = (str(pred.get('column')) if selector_header == 'Prediction'
                                                    else ('N/A' if pred.get('threshold') is None
                                                                else f"{pred['threshold']:.2f}"))
                first = rep_i == 0
                val = _metric_values(stream, fp_unit, lag_digits=1)
                row = [base_values['stream'] if first else '', selector,
                       base_values['total'] if first else '',
                       base_values['gt_dur'] if first else '']
                if show_stream_meta:
                    row.append(base_values['p_dets'] if first else '')
                row += [(base_values['gt'] if first else ''),
                        val['detected'], val['false'], val['recall'],
                        val['lag'], val['fp_burden'], val['fp_rate']]
                if show_stream_meta:
                    row.append(val['fp_per_det'])
                tbl.append(row)

        print("\n=== Threshold Comparison ===")
        print(f"Status:     pass [{len(passed_keys)}]   failed [{failed_count}]")
        alignments = ['<'] + ['^']*(len(headers) - 1)
        _print_table(headers, tbl, alignments, separators)

    reports = list(reports)
    table_mode = kwargs.get('results_table', False)
    if table_mode is True:
        table_mode = 'thrs_cmp'
    elif table_mode not in {False, None, 'standard', 'thrs_cmp'}:
        raise ValueError("results_table must be False, 'standard', or 'thrs_cmp'")
    if table_mode == 'standard':
        kwargs['multi_thresholds'] = True
        for index, report in enumerate(reports):
            if index:
                print()
            print_stream_metric(report, **kwargs)
    elif table_mode == 'thrs_cmp':
        print_thrs_cmp()

    show_meta = any(report.get('stream_meta', {}).get('person_dets') is not None
                    for report in reports)
    headers = ['Prediction', 'Recall', 'FP score', 'FP burden', 'Score', 'Detected', 'Missed', 'False', f'False/{fp_unit}']
    if show_meta:
        headers.append('False/P (1k)')

    table = []
    for report in reports:
        values = _metric_values(report, fp_unit)
        prediction = report.get('prediction', {})
        scores = report.get('scores', {})
        events = report.get('events', {})
        prediction_name = (str(prediction['column']) if prediction.get('column') is not None
                           else ('N/A' if prediction.get('threshold') is None
                                 else f"th={prediction['threshold']:g}"))
        detected = events.get('full', 0) + events.get('half', 0)
        missed = events.get('gt', 0) - detected
        row = [prediction_name,
               _fmt(scores.get('recall')),
               _fmt(scores.get('fp')),
               _fmt(scores.get('fp_burden')),
               _fmt(scores.get('total')),
               values['detected'], str(missed), values['false'], values['fp_rate']]
        if show_meta:
            detections = report.get('stream_meta', {}).get('person_dets')
            fp_per_dets = (None if not detections else
                           events.get('false', 0)*1000/detections)
            row.append(_fmt(fp_per_dets))
        table.append(row)

    print("\n=== Multi-Threshold Summary ===")
    alignments = ['<'] + ['>']*(len(headers) - 1)
    _print_table(headers, table, alignments)
    return reports

# endregion


#* region Other Helpers  ---------------------------------------------------
# -----------------------------------------------------------------------
_AUTO_META = object()


def _resolve_stream_meta_path(stream_path, timeline_files, meta_info):
    if meta_info is not _AUTO_META:
        return None if meta_info is None else Path(meta_info)

    source = Path(stream_path) if not isinstance(stream_path, (list, tuple, set, frozenset)) else None
    if source is not None and source.is_dir():
        path = source/SJ_META_INFO
        if path.is_file():
            return path
    parents = {Path(path).parent for path in timeline_files}
    if len(parents) == 1:
        path = next(iter(parents))/SJ_META_INFO
        if path.is_file():
            return path
    return DEFAULT_STREAM_META if DEFAULT_STREAM_META.is_file() else None


def _load_stream_meta(path: Path|None) -> list[dict]:
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
    def find_meta(stream: dict, records: list[dict]) -> dict | None:
        stem = stream_stem(stream.get('stream', ''))
        candidates = [record for record in records
                      if isinstance(record, dict) and record.get('stem') == stem]
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
        result['stream_meta'] = { 'path': str(meta_path),
                                  'matched': len(matched),
                                  'person_dets': sum(int(item.get('person_dets', 0)) for item in matched),
                                  'max_dets_frame': max(int(item.get('max_dets_frame', 0)) for item in matched),
                                  'consecutive_det_frames': max( int(item.get('consecutive_det_frames',
                                                                 item.get('max_consecutive_det_frames', 0)))
                                                                 for item in matched),
                                  }
    else:
        result['stream_meta'] = {'path': str(meta_path), 'matched': 0}
    return result

# endregion

#561(,14,1) ->530(,13,1)
#thcmp: 645(,27,13) -> cln-up:590(,13,13)
#csv-f: 694(,14,15)->700(,13,15)-707(,14,2)
#644(,14,2)->636(,9,2)

if __name__ == '__main__': pass
