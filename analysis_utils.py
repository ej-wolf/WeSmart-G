"""Timeline preparation, persistence, metadata, and presentation utilities."""
import re
import csv, json
import math
from statistics import median
from numbers import Integral, Real
from pathlib import Path
import numpy as np

#* project imports
from json_stream_utils import DEFAULT_STREAM_META, SJ_META_INFO, stream_stem
from json_utils import STREAM_FILE_TYPES
from common.my_local_utils import _fmt, as_collection, get_unique_name

TIMELINE_PROB_ATOL = 1e-6
TIMELINE_PROB_VARIANCE_K = 0.03
TIMELINE_TIME_ATOL = 1e-9
TIMELINE_MAX_ISSUES = 50


#* region Public API  ---------------------------------------------------
# -----------------------------------------------------------------------

def build_timelines(y_true, y_prob, streams, t_start, t_end, n_frames=None) -> list[dict]:
    """Build ordered stream timelines from normalized per-window arrays."""
    streams = np.asarray(streams)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    t_start = np.asarray(t_start, dtype=np.float64)
    t_end = np.asarray(t_end, dtype=np.float64)

    if n_frames is None:
        n_frames = np.full(len(y_true), -1, dtype=np.int64)
    else:
        n_frames = np.asarray(n_frames, dtype=np.int64)

    arrays = {'y_true': y_true, 'y_prob': y_prob, 'streams': streams,
              't_start': t_start, 't_end': t_end, 'n_frames': n_frames}
    invalid = {name: value.shape for name, value in arrays.items() if value.ndim != 1 or len(value) != len(y_true)}
    if invalid:
        raise ValueError(f"timeline arrays must be equal-length 1D arrays: {invalid}")
    if not len(y_true):
        return []
    if any(end <= start for start, end in zip(t_start, t_end)):
        raise ValueError("timeline windows must have t_end greater than t_start")

    grouped = {}
    for index, stream in enumerate(streams):
        grouped.setdefault(str(stream), []).append(index)

    timelines = []
    fields = ['win_idx', 't_frm', 't_start', 'n_frm', 'gt_label', 'y_prob']
    for stream, indices in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda index: (t_end[index], t_start[index], index))
        rows = [{'win_idx': win_idx,
                 't_frm': float(t_end[index]),
                 't_start': float(t_start[index]),
                 'n_frm': int(n_frames[index]),
                 'gt_label': int(y_true[index]),
                 'y_prob': float(y_prob[index])}
                 for win_idx, index in enumerate(ordered)]
        timelines.append({'metadata': {'timeline': stream, 'source': stream},
                          'fieldnames': list(fields), 'rows': rows})
    return timelines


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


def compare_timeline(test_timeline: dict, ref_timeline: dict, *,
                     prob_atol=TIMELINE_PROB_ATOL,
                     prob_variance_k=TIMELINE_PROB_VARIANCE_K,
                     time_atol=TIMELINE_TIME_ATOL,
                     max_issues=TIMELINE_MAX_ISSUES) -> dict:
    """Compare two normalized timeline dictionaries without file handling."""
    def add_structure_issue(issue_type, **details):
        structure['mismatches'] += 1
        if len(structure['issues']) < max_issues:
            structure['issues'].append({'type': issue_type, **details})

    def add_numeric_issue(field, row_idx, test_val, ref_val, abs_error):
        numeric['mismatches'] += 1
        if len(numeric['issues']) < max_issues:
            numeric['issues'].append({'field': field, 'row': row_idx,
                                      'test': test_val, 'ref': ref_val,
                                      'abs_error': abs_error})

    if not isinstance(test_timeline, dict) or not isinstance(ref_timeline, dict):
        raise TypeError('compare_timeline expects two timeline dictionaries')
    if prob_atol < 0 or prob_variance_k < 0 or time_atol < 0:
        raise ValueError('timeline tolerances must be non-negative')
    if max_issues < 0:
        raise ValueError('max_issues must be non-negative')

    test_rows = test_timeline.get('rows')
    ref_rows = ref_timeline.get('rows')
    test_fields = set(test_timeline.get('fieldnames') or ())
    ref_fields = set(ref_timeline.get('fieldnames') or ())
    if not isinstance(test_rows, list) or not isinstance(ref_rows, list):
        raise TypeError("timeline dictionaries must contain a 'rows' list")

    required = {'win_idx', 't_frm', 't_start', 'n_frm', 'gt_label', 'y_prob'}
    test_pred = {field for field in test_fields if field == 'y_pred' or field.startswith('y_prd-')}
    ref_pred = {field for field in ref_fields if field == 'y_pred' or field.startswith('y_prd-')}
    structure = {'ok': True,
                 'mismatches': 0,
                 'row_count': {'test': len(test_rows), 'ref': len(ref_rows)},
                 'missing_required_test': sorted(required - test_fields),
                 'missing_required_ref': sorted(required - ref_fields),
                 'missing_columns': sorted(ref_fields - test_fields),
                 'extra_columns': sorted(test_fields - ref_fields),
                 'issues': []}
    numeric = {'ok': True, 'count': 0, 'mismatches': 0,
               'max_abs': 0.0, 'max_path': None, 'issues': [],
               'time_atol': time_atol,
               'probability': {}}

    if len(test_rows) != len(ref_rows):
        add_structure_issue('row_count', test=len(test_rows), ref=len(ref_rows))
    if structure['missing_required_test'] or structure['missing_required_ref']:
        add_structure_issue('required_columns',
                            test=structure['missing_required_test'],
                            ref=structure['missing_required_ref'])
    if structure['missing_columns'] or structure['extra_columns']:
        add_structure_issue('columns', missing=structure['missing_columns'],
                            extra=structure['extra_columns'])
    if not test_pred or not ref_pred:
        add_structure_issue('prediction_columns', test=sorted(test_pred), ref=sorted(ref_pred))

    comparable = (not structure['missing_required_test'] and
                  not structure['missing_required_ref'] and
                  test_fields == ref_fields and test_pred == ref_pred)
    compared_rows = min(len(test_rows), len(ref_rows))
    flip_count = 0
    max_prob_delta = 0.0

    if comparable and compared_rows:
        ref_prob = np.asarray([row['y_prob'] for row in ref_rows[:compared_rows]], dtype=float)
        test_prob = np.asarray([row['y_prob'] for row in test_rows[:compared_rows]], dtype=float)
        finite = ref_prob[np.isfinite(ref_prob)]
        if finite.size >= 2:
            prob_median = float(np.median(finite))
            prob_spread = float(1.4826*np.median(np.abs(finite - prob_median)))
            prob_scale = float(np.sqrt(np.mean(np.abs(finite)**2)))
            denominator = max(prob_scale, prob_atol)
            prob_rtol = prob_variance_k*prob_spread/denominator if denominator else 0.0
        else:
            prob_median = float(finite[0]) if finite.size else None
            prob_spread = prob_scale = prob_rtol = 0.0

        prob_close = np.isclose(test_prob, ref_prob, atol=prob_atol,
                               rtol=prob_rtol, equal_nan=True)
        prob_delta = np.abs(test_prob - ref_prob)
        prob_delta = np.where(prob_close & ~np.isfinite(prob_delta), 0.0, prob_delta)
        prob_delta = np.where(~prob_close & np.isnan(prob_delta), np.inf, prob_delta)
        max_prob_delta = float(np.max(prob_delta))
        max_prob_idx = int(np.argmax(prob_delta))
        numeric['count'] += compared_rows
        numeric['max_abs'] = max_prob_delta
        numeric['max_path'] = f'row[{max_prob_idx}].y_prob'
        for row_idx in np.flatnonzero(~prob_close):
            idx = int(row_idx)
            add_numeric_issue('y_prob', idx, float(test_prob[idx]),
                              float(ref_prob[idx]), float(prob_delta[idx]))

        numeric['probability'] = {'atol': prob_atol,
                                  'variance_k': prob_variance_k,
                                  'rtol': prob_rtol,
                                  'median': prob_median,
                                  'spread': prob_spread,
                                  'scale': prob_scale,
                                  'mismatches': int(np.count_nonzero(~prob_close))}

        for row_idx, (test_row, ref_row) in enumerate(zip(test_rows, ref_rows)):
            for field in ('win_idx', 'n_frm', 'gt_label'):
                if test_row[field] != ref_row[field]:
                    add_structure_issue('value', field=field, row=row_idx,
                                        test=test_row[field], ref=ref_row[field])

            for field in ('t_frm', 't_start'):
                test_val, ref_val = float(test_row[field]), float(ref_row[field])
                delta = abs(test_val - ref_val)
                numeric['count'] += 1
                if not np.isclose(test_val, ref_val, atol=time_atol, rtol=0.0, equal_nan=True):
                    add_numeric_issue(field, row_idx, test_val, ref_val, delta)
                if delta > numeric['max_abs']:
                    numeric['max_abs'] = delta
                    numeric['max_path'] = f'row[{row_idx}].{field}'

            for field in sorted(test_pred):
                if test_row[field] != ref_row[field]:
                    flip_count += 1
                    add_structure_issue('prediction', field=field, row=row_idx,
                                        test=test_row[field], ref=ref_row[field])

    prediction_count = compared_rows*len(test_pred) if comparable else 0
    structure['ok'] = structure['mismatches'] == 0
    numeric['ok'] = numeric['mismatches'] == 0
    ok = structure['ok'] and numeric['ok'] and flip_count == 0
    return {'ok': ok, 'rows': compared_rows,
            'structure': structure, 'numeric': numeric,
            'flip_count': flip_count,
            'flip_rate': flip_count/prediction_count if prediction_count else 0.0,
            'max_prob_delta': max_prob_delta}


def load_timelines(timeline_input) -> tuple[list[dict], list[dict], list[Path]]:
    """ Load timeline files from one path, directory, or path collection."""

    inputs = as_collection(timeline_input)
    files = []
    for item in inputs:
        if isinstance(item, dict):
            files.append(item)
            continue
        path = Path(item)
        files.extend(sorted(path.glob('timeline_*.csv')) if path.is_dir() else [path])

    timelines, errors, paths = [], [], []
    seen = set()
    for item in files:
        if isinstance(item, dict):
            timelines.append(item)
            continue
        key = str(Path(item))
        if key in seen:
            continue
        seen.add(key)
        try:
            timelines.append(load_timeline_csv(item))
            paths.append(Path(item))
        except Exception as error:
            errors.append({'timeline': Path(item).stem, 'stream': None, 'status': 'fail',
                           'error': f"{type(error).__name__}: {error}"})
    return timelines, errors, paths


def convert_tcn_format(csv_path, out_dir=None) -> list[Path]:
    """Convert one TCN prediction CSV or a directory of CSVs into compatible timelines."""

    required = {'json_name', 'window_index', 'window_start_frame', 'window_end_frame',
                'window_start_time_sec', 'window_end_time_sec', 'target',
                'prob_raw', 'pred_raw'}

    def number(row, key, kind=float):
        return kind(float(row[key]))

    def json_stem(value):
        """Remove only a recognized JSON/archive suffix, preserving variant tags such as `.5fps`."""
        name = Path(value).name
        lower = name.lower()
        for suffix in STREAM_FILE_TYPES:
            if lower.endswith(suffix):
                return name[:-len(suffix)]
        return Path(name).stem

    csv_path = Path(csv_path)
    if csv_path.is_dir():
        csv_files = sorted(csv_path.glob('*.window_predictions.csv'))
        if not csv_files:
            raise FileNotFoundError(f'no *.window_predictions.csv files found: {csv_path}')
        target_dir = Path(out_dir) if out_dir is not None else csv_path
        out_files = []
        for path in csv_files:
            out_files.extend(convert_tcn_format(path, target_dir))
        return out_files

    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"TCN CSV is missing columns: {sorted(missing)}")
        groups = {}
        for row in reader:
            groups.setdefault(row['json_name'], []).append(row)

    out_files = []
    fields = ['win_idx', 't_frm', 't_start', 'n_frm', 'gt_label', 'y_prob', 'y_pred']
    for json_name, rows_in in sorted(groups.items()):
        rows_in = sorted(rows_in, key=lambda row: number(row, 'window_index', int))
        rows = []
        for row in rows_in:
            t_start = number(row, 'window_start_time_sec')
            t_end = number(row, 'window_end_time_sec')
            rows.append({'win_idx': number(row, 'window_index', int) - 1,
                         't_frm': t_end,
                         't_start': t_start,
                         'n_frm': number(row, 'window_end_frame', int) - number(row, 'window_start_frame', int),
                         'gt_label': number(row, 'target', int),
                         'y_prob': number(row, 'prob_raw'),
                         'y_pred': number(row, 'pred_raw', int)})

        spans = [row['t_frm'] - row['t_start'] for row in rows if row['t_frm'] > row['t_start']]
        times = [row['t_frm'] for row in rows]
        diffs = [right - left for left, right in zip(times, times[1:]) if right > left]
        frame_spans = [(row['n_frm'], row['t_frm'] - row['t_start']) for row in rows
                       if row['n_frm'] > 0 and row['t_frm'] > row['t_start']]
        source = json_stem(json_name)
        metadata = {'source': source,
                    'win_span': median(spans) if spans else None,
                    'fps': (sum(count for count, _ in frame_spans)/sum(span for _, span in frame_spans)
                            if frame_spans else None),
                    'infer_t': median(diffs) if diffs else None}
        if metadata['infer_t']:
            metadata['frq_i'] = 1.0/metadata['infer_t']
        if 'threshold_raw' in rows_in[0]:
            thresholds = {row.get('threshold_raw', '') for row in rows_in}
            if len(thresholds) == 1:
                metadata['threshold'] = number(rows_in[0], 'threshold_raw')
        metadata = {key: val for key, val in metadata.items() if val is not None}

        out_path = out_dir/f"timeline_{source}.csv"
        with out_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for key in ('source', 'threshold', 'win_span', 'fps', 'infer_t', 'frq_i'):
                if key in metadata:
                    val = (f"{float(metadata[key]):.2f}" if key in {'win_span', 'infer_t'} else
                           round(metadata[key], 9) if isinstance(metadata[key], Real) else metadata[key])
                    writer.writerow(['infer_frq' if key == 'frq_i' else key, val])
            writer.writerow(['other data', ''])
            table = csv.DictWriter(file, fieldnames=fields)
            table.writeheader()
            table.writerows(rows)
        out_files.append(out_path)
    return out_files


def save_timeline_csv(timeline: dict, output_path, pred_cols=None) -> Path:
    """ Save one loaded timeline dictionary and optional binary prediction columns."""

    def next_column_name(fields, name):
        if name not in fields:
            return name
        n = 2
        while f"{name}({n})" in fields:
            n += 1
        return f"{name}({n})"

    def base_rows_match(rows_a, rows_b, base_fields):
        if len(rows_a) != len(rows_b):
            return False
        return all(str(row_a.get(field, '')) == str(row_b.get(field, ''))
                   for row_a, row_b in zip(rows_a, rows_b)
                   for field in base_fields)

    output_path = Path(output_path)
    if output_path.is_dir():
        output_path = output_path/f"{timeline['metadata']['timeline']}.csv"
    elif not output_path.suffix:
        output_path = output_path.with_suffix('.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        saved = load_timeline_csv(output_path)
        rows = [dict(row) for row in saved['rows']]
        fields = list(saved['fieldnames'])
        base_fields = [field for field in timeline['fieldnames'] if field in fields and
                       field != 'y_pred' and not field.startswith('y_prd-')]
        if not base_rows_match(timeline['rows'], rows, base_fields):
            raise ValueError(f"existing timeline does not match generated timeline: {output_path}")
        metadata = saved.get('metadata', {})
    else:
        rows = [dict(row) for row in timeline['rows']]
        fields = list(timeline['fieldnames'])
        metadata = timeline.get('metadata', {})

    for name, values in (pred_cols or {}).items():
        if len(values) != len(rows):
            raise ValueError(f"prediction column {name} length mismatch")
        name = next_column_name(fields, name)
        fields.append(name)
        for row, value in zip(rows, values):
            row[name] = int(value)

    with output_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for key in ('source', 'threshold', 'win_span', 'fps', 'infer_t', 'frq_i'):
            if key in metadata:
                val = (f"{float(metadata[key]):.2f}" if key in {'win_span', 'infer_t'} else
                       round(metadata[key], 9) if isinstance(metadata[key], Real) else metadata[key])
                writer.writerow(['infer_frq' if key == 'frq_i' else key, val])
        writer.writerow(['other data', ''])
        table = csv.DictWriter(file, fieldnames=fields)
        table.writeheader()
        table.writerows(rows)
    return output_path


def load_report_file(report_path) -> dict | list:
    """Load a saved analysis JSON or tabular CSV report without evaluating it."""
    report_path = Path(report_path)
    if report_path.suffix.lower() == '.json':
        with report_path.open('r', encoding='utf-8') as file:
            return json.load(file)
    if report_path.suffix.lower() != '.csv':
        raise ValueError(f"unsupported analysis report format: {report_path.suffix}")

    metadata, rows, fields = {}, [], None
    with report_path.open('r', newline='', encoding='utf-8-sig') as file:
        for values in csv.reader(file):
            if not values or not any(value.strip() for value in values):
                continue
            if fields is None and values[:2] == ['parameter', 'value']:
                continue
            if fields is None and 'stream' in values:
                fields = values
                continue
            if fields is None:
                if len(values) >= 2:
                    metadata[values[0]] = values[1]
                continue
            rows.append(dict(zip(fields, values)))
    if fields is None:
        raise ValueError(f"analysis table header not found: {report_path}")
    return {'metadata': metadata, 'fieldnames': fields, 'rows': rows}


def save_metric_report(report: dict | list[dict], output_path) -> Path:
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
        batch_report = any(result.get('model') is not None for result in reports)
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
                row = {'stream': full_stream_name(stream),
                             'fps': timing.get('fps', ''),
                             'win_span': timing.get('window_span', ''),
                             'infer_frq': timing.get('frq_i', ''),
                             'threshold': '' if prediction.get('threshold') is None else prediction['threshold'],
                             'y_pred': prediction.get('column') or '',
                             'status': stream.get('status', 'N/A'),
                             'timeline_file': str(stream.get('timeline', '')),
                             'src_span': meta.get('duration', ''),
                             'frames_count': meta.get('frames', ''),
                             'yolo_threshold': meta.get('yolo_threshold', ''),
                             'yolo_dets': meta.get('person_dets', ''),
                             'max_dets_frm': meta.get('max_dets_frame', ''),
                             'consc_det_frms': meta.get('consc_det_frms', ''),
                             't_total': time_info.get('total', ''),
                             'gt_events': events.get('gt', ''),
                             'gt_duration': duration.get('total', ''),
                             'longest_gt': duration.get('longest', ''),
                             'pred_events': events.get('predicted', ''),
                             'det_full': events.get('full', ''),
                             'det_half': events.get('half', ''),
                             'false' : events.get('false', ''),
                             'recall': scores.get('recall', ''),
                             'avg_lag': events.get('avg_lag', ''),
                             't_fp': time_info.get('t_fp', ''),
                             't_tn': time_info.get('t_tn', ''),
                             'fp_per_h' : events.get('fp_per_h', ''),
                             'fp_burden': scores.get('fp_burden', ''),
                             'notes': stream.get('error', '')}
                if batch_report:
                    row['model'] = result.get('model', '')
                    row['source_dir'] = result.get('source_dir', '')
                rows.append(row)

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
        fields = ['model', 'source_dir'] if batch_report else []
        fields += ['stream', 'fps']
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
            return (str(row.get('model', '')).casefold(),
                    str(row.get('source_dir', '')).casefold(),
                    str(row['stream']).casefold(),
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
        return f'{value:.2f} s'

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
    meta = data.get('stream_meta', {})
    events = data.get('events', {})
    scores = data.get('scores', {})
    duration = events.get('duration', {})
    detected = events.get('full', 0) + events.get('half', 0) if events else 'N/A'
    fp_rate  = events.get('fp_per_h')
    if fp_rate is not None:
        fp_rate = fp_rate if fp_unit == 'h' else fp_rate/60.0
        fp_rate = f'{fp_rate:.{1 if fp_unit == "h" else 2}f}'
    yolo_dets = meta.get('person_dets')
    fp_per_det = ('N/A' if yolo_dets is None or yolo_dets <= 0 else
                  f'{events.get("false", 0)*1000/yolo_dets:.2f}')
    return {'stream' : _stream_name(data),
            'total'  : _fmt_duration(data.get('time', {}).get('total')),
            'gt_dur' : _fmt_duration(duration.get('total')),
            'longest': _fmt_duration(duration.get('longest')),
            'p_dets' : 'N/A' if yolo_dets is None else str(yolo_dets),
            'max_p'  : 'N/A' if meta.get('max_dets_frame') is None else str(meta['max_dets_frame']),
            'max_run': 'N/A' if meta.get('consc_det_frms') is None else str(meta['consc_det_frms']),
            'gt': str(events.get('gt', 'N/A')),
            'detected': str(detected),
            'false' : str(events.get('false', 'N/A')),
            'recall': _fmt(scores.get('recall'), d=2),
            'lag':   _fmt(events.get('avg_lag'), d=lag_digits),
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


def print_metric_report(report: dict, **kwargs):
    """ Print an aggregate stream-metric report and optional per-file details."""

    results_table = bool(kwargs.get('results_table', False))
    total_row = bool(kwargs.get('total_row', False))
    meta_info = kwargs.get('meta_info')
    meta_path = Path(meta_info) if meta_info is not None else None
    if meta_path is not None:
        attach_stream_meta(report, meta_path)
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
        row += [values['p_dets'], values['max_p'], values['max_run']] if show_meta else []
        row += [values['gt'], values['detected'], values['false'], values['recall'],
                values['lag'], values['fp_burden'], values['fp_rate']]
        row += [values['fp_per_det']] if show_meta else []
        table += [row]
        # table.append(tuple(row))

    if total_row:
        values = _metric_values(report, fp_unit)
        row = [f'Total [{len(table)}]', values['total'], values['gt_dur'], values['longest']]
        row += [values['p_dets'], values['max_p'], values['max_run']]  if show_meta else []
        row += [values['gt'], values['detected'], values['false'], values['recall'],
                values['lag'], values['fp_burden'], values['fp_rate']]
        row += [values['fp_per_det']]   if show_meta else []
        table += [row]
        # table.append(tuple(row))

    alignments = ['<', '>', '>', '>']
    if show_meta:
        alignments += ['^', '^', '^']
    alignments += ['^']*(7 + int(show_meta))
    _print_table(headers, table, alignments,
                 separators={len(table) - 1} if total_row else ())
    return report


def print_threshold_comparison(reports: list[dict], **kwargs):
    """ Print multi-threshold summaries using the selected table layout."""

    def print_thrs_cmp():
        def fmt_yolo_dets(value):
            if value is None:
                return 'N/A'
            value = int(value)
            return f'{value/1000:.0f}k' if value > 100_000 else f'{value:,}'

        selector_header = ('Prediction' if any(rep.get('prediction', {}).get('column') for rep in reports)
                                        else 'Threshold')
        grouped, stream_order = {}, []
        for rep_i, rep in enumerate(reports):
            for stm_i in rep.get('streams', []):
                key = str(stm_i.get('timeline', 'N/A'))
                if key not in grouped:
                    grouped[key] = {}
                    stream_order.append(key)
                grouped[key][rep_i] = stm_i

        passed_keys = [key for key in stream_order if len(grouped[key]) == len(reports)
                                                   and  all(stream.get('status') == 'pass' for stream in grouped[key].values())]
        failed_count = len(stream_order) - len(passed_keys)
        show_stream_meta = any( stream.get('stream_meta', {}).get('person_dets') is not None
                                for key in passed_keys for stream in grouped[key].values())

        headers = ['Stream', selector_header, 'Total(s)', 'GT dur']
        if show_stream_meta:
            headers.append('yolo-det')
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
                stm_i = by_report.get(rep_i)
                pred = rep.get('prediction', {})
                selector = (str(pred.get('column')) if selector_header == 'Prediction'
                                                    else ('N/A' if pred.get('threshold') is None
                                                                else f"{pred['threshold']:.2f}"))
                first = rep_i == 0
                val = _metric_values(stm_i, fp_unit, lag_digits=1)
                row = [base_values['stream'] if first else '', selector,
                       base_values['total'] if first else '',
                       base_values['gt_dur'] if first else '']
                if show_stream_meta:
                    yolo_dets = base_stream.get('stream_meta', {}).get('person_dets')
                    row.append(fmt_yolo_dets(yolo_dets) if first else '')
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

    fp_unit = 'min' if kwargs.get('fp_unit', 'h') in {'min', 'minute'} else 'h'

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
            print_metric_report(report, **kwargs)
    elif table_mode == 'thrs_cmp':
        print_thrs_cmp()

    show_meta = any(report.get('stream_meta', {}).get('person_dets') is not None
                    for report in reports)
    selector_type = ('Column' if any(r.get('prediction',{}).get('column') is not None for r in reports)
                              else 'Threshold')
    headers = ['Prediction', 'GT', 'Detected', 'Missed', 'False',
               'Recall', 'Lag(s)', 'FP score', 'FP burden', f'FP/{fp_unit}']
    if show_meta:
        headers.append('FP/kP')
    headers.append('Score')

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
        row = [prediction_name, str(events.get('gt', 'N/A')),
               values['detected'], str(missed), values['false'],
               _fmt(scores.get('recall')), values['lag'],
               _fmt(scores.get('fp')), _fmt(scores.get('fp_burden')),
               values['fp_rate']]
        if show_meta:
            dets = report.get('stream_meta', {}).get('person_dets')
            fp_per_dets = None if not dets else events.get('false', 0)*1000/dets
            row.append(_fmt(fp_per_dets))
        row.append(_fmt(scores.get('total')))
        table.append(row)

    def summary_tag():
        tags = []
        for r in reports:
            for s in r.get('streams', []):
                tl  = str(s.get('timeline', ''))
                src  = str(s.get('stream') or '')
                name = tl.removeprefix('timeline_')
                if src and name.endswith(f"_{src}"):
                    tags.append(name[:-(len(src) + 1)])
        tags = list(dict.fromkeys(t for t in tags if t))
        return tags[0] if len(tags) == 1 else None

    def print_grouped_summary(hdr, rows):
        def fmt_row(items):
            cells = [f"{str(item):{alignments[idx]}{widths[idx]}}" for idx, item in enumerate(items)]
            return (f" {cells[0]} ┃ "
                    f"{' | '.join(cells[1:5])} ┃ "
                    f"{' | '.join(cells[5:])}")

        alignments = ['<'] + ['^']*(len(hdr) - 1)
        widths = [max(5, len(header), *(len(str(row[idx])) for row in rows))
                  for idx, header in enumerate(hdr)]
        widths[0] = max(widths[0], len(selector_type))

        events_w = sum(widths[1:5]) + 3*(4 - 1)
        metrics_w = sum(widths[5:]) + 3*(len(widths[5:]) - 1)
        first_w = widths[0]
        group_row = (f" {'Prediction':<{first_w}} ┃ "
                     f"{'Events':^{events_w}} ┃ "
                     f"{'Metrics':^{metrics_w}}")
        header_row = fmt_row([selector_type] + hdr[1:])
        sep_cells = ['-'*width for width in widths]
        sep_row = (f" {sep_cells[0]} ┃ "
                   f"{'-+-'.join(sep_cells[1:5])} ┃ "
                   f"{'-+-'.join(sep_cells[5:])}")

        print(group_row)
        print(header_row)
        print(sep_row)
        for r in rows:
            print(fmt_row(r))

    tag = summary_tag()
    print(f"\n=== Threshold Summary{' for ' + tag if tag else ''} ===")
    print_grouped_summary(headers, table)
    return reports


def print_model_comparison(reports: list[dict], **kwargs):
    """Print aggregate metric rows ordered by model and operating point."""
    fp_unit = 'min' if kwargs.get('fp_unit', 'h') in {'min', 'minute'} else 'h'
    reports = list(reports)
    has_columns = any(report.get('prediction', {}).get('column') is not None
                      for report in reports)
    selector_header = 'Prediction' if has_columns else 'Threshold'
    headers = ['Model', selector_header, 'GT', 'Detected', 'False', 'Recall',
               'Lag(s)', 'FP burden', f'FP/{fp_unit}', 'Score']
    rows = []
    separators = set()
    last_model = None
    for report in reports:
        model = str(report.get('model', 'N/A'))
        if last_model is not None and model != last_model:
            separators.add(len(rows))
        last_model = model
        values = _metric_values(report, fp_unit, lag_digits=1)
        prediction = report.get('prediction', {})
        selector = (str(prediction['column']) if has_columns and prediction.get('column') is not None
                    else ('N/A' if prediction.get('threshold') is None
                          else f"th={prediction['threshold']:g}"))
        scores = report.get('scores', {})
        rows.append([model, selector, values['gt'], values['detected'], values['false'],
                     values['recall'], values['lag'], values['fp_burden'],
                     values['fp_rate'], _fmt(scores.get('total'))])

    print("\n=== Model Comparison ===")
    _print_table(headers, rows, ['<'] + ['^']*(len(headers) - 1), separators)
    return reports


def print_test_report(results, **kwargs):
    """Print one clip/video evaluation summary produced by evaluation_core."""
    from evaluation_core import support_pair

    if isinstance(results, (str, Path)):
        with Path(results).open('r', encoding='utf-8') as file:
            summary = json.load(file)
    elif isinstance(results, dict):
        summary = results
    else:
        raise TypeError("results must be dict or path to summary json")

    precision = kwargs.get('precision', 4)
    label_w = kwargs.get('label_width', 20)
    cm = summary.get('confusion_matrix')
    testing_set = summary.get('testing_set', {})
    support = summary.get('support_video', summary.get('support_clips'))
    if support is None:
        support = testing_set.get('videos_support', testing_set.get('clips_support'))
    support = support_pair(support)
    support_str = f"{support[0]}/ {support[1]}" if support is not None else 'N/A'
    num_samples = summary.get('num_videos', summary.get('num_clips'))
    if num_samples is None:
        num_samples = testing_set.get('videos_num', testing_set.get('clips_num'))

    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{precision}f}"
        if isinstance(value, np.integer):
            return str(int(value))
        return str(value)

    rows = [("Predictions file", Path(summary.get('raw_results_path', '')).name),
            ("Num_samples", num_samples),
            ("GT_counts 0/1", support_str),
            ("accuracy", summary.get('accuracy')),
            ("recall", summary.get('recall')),
            ("FPR", summary.get('FPR')),
            ("AUC", summary.get('roc_auc', summary.get('ROC AUC')))]
    print("\n===== Test Summary =====")
    for key, value in rows:
        if value is not None:
            print(f"{key:<{label_w}}: {fmt(value)}")
    if cm is not None and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        print(f"{'Confusion Matrix':<{label_w}}: pred-0  pred-1\n"
              f"{'True: 0':<{label_w}}[[{cm[0][0]:>5}, {cm[0][1]:>5}]\n"
              f"{'True: 1':<{label_w}} [{cm[1][0]:>5}, {cm[1][1]:>5}]]\n")
    return summary


def print_report_table(report: dict):
    """Print one previously saved tabular CSV report."""
    if not {'fieldnames', 'rows'}.issubset(report):
        raise ValueError("tabular report requires fieldnames and rows")
    metadata = report.get('metadata', {})
    if metadata:
        print("\n=== Parameters ===")
        _print_table(['Parameter', 'Value'], list(metadata.items()), ['<', '<'])
    print("\n=== Results ===")
    fields = report['fieldnames']
    rows = [[row.get(field, '') for field in fields] for row in report['rows']]
    _print_table(fields, rows, ['<']*len(fields))
    return report


# endregion


#* region Stream Metadata  ------------------------------------------------
# -----------------------------------------------------------------------
AUTO_META = object()


def resolve_stream_meta_path(stream_path, timeline_files, meta_info=AUTO_META):
    if meta_info is not AUTO_META:
        return None if meta_info is None else Path(meta_info)

    source = Path(stream_path) if isinstance(stream_path, (str, Path)) else None
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


def load_stream_meta(path: Path|None) -> list[dict]:
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


def attach_stream_meta(result: dict, meta_path: Path | None) -> dict:
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

    records = load_stream_meta(meta_path)
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
                                  'consc_det_frms': max( int(item.get('consc_det_frms',
                                                                 item.get('max_consc_det_frms', 0)))
                                                                 for item in matched),
                                  }
    else:
        result['stream_meta'] = {'path': str(meta_path), 'matched': 0}
    return result

# endregion

#sm-tools 636(,9,2) -> sm-tools 767(1,10,2)
#837(1,17,2) ; #912(2,15,2)

if __name__ == '__main__': pass
