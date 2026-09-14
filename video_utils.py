import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import cv2
import numpy as np

from common.my_local_utils import as_collection, get_unique_name, print_progress
from video_analytics import measure_video_fps, DEFAULT_MF_THRESHOLD, FPS_CONSISTENCY_RTOL

DECI_PRECISION = 5
MIN_ELIGIBLE_FPS = 0.5
VIDEO_SUFFIXES = {'.mp4', '.avi', '.wmv', '.flv', '.mkv', '.mov', '.m4v'}
DEFAULT_EFF_FPS_REPORT = 'effective_fps.json'
FPS_SUMMARY_COL_WIDTH = 16


# region API
def get_measured_fps(videos, mf_threshold=DEFAULT_MF_THRESHOLD, **kwargs) -> tuple[float, dict]:
    """ Measure one video or a collection/directory and aggregate its results.
    Directories are searched recursively by default. Invalid inputs are kept in
    `errors` and do not prevent the remaining videos from being analyzed.
    The returned FPS is the duration-weighted mean measured FPS.
    Keyword options are ``recursive`` (default ``True``), ``save_results``
    (a path, ``True``, or a false value), and ``print_res`` (default ``False``).
    With ``save_results=True``, the report is saved as ``effective_fps.json``
    in the supplied video directory, or beside a single video.
    """

    def report_path():
        """ Return the default report path for the supplied video inputs."""
        src = as_collection(videos)
        if len(src) == 1 and Path(src[0]).is_dir():
            report_dir = Path(src[0])
        elif video_paths:
            report_dir = video_paths[0].parent
        else:
            report_dir = Path.cwd()
        return report_dir/DEFAULT_EFF_FPS_REPORT

    recursive = kwargs.get('recursive', True)
    save_results = kwargs.get('save_to', None)
    print_res = kwargs.get('print_res', False)

    mf_threshold = _validate_threshold(mf_threshold)
    video_paths, input_errors = _resolve_video_inputs(videos, recursive)
    video_results, errors = [], list(input_errors)
    total_vids = len(video_paths)
    completed = 0
    for vp in video_paths:
        try:
            _, result = measure_video_fps(vp, mf_threshold)
            video_results.append(result)
        except (OSError, ValueError, TypeError, cv2.error) as err:
            errors.append({'path': str(vp), 'error': f'{type(err).__name__}: {err}'})
        finally:
            if total_vids > 1:
                completed += 1
                print_progress(total_vids, completed, mode='single_line', current=vp.name)
    if total_vids > 1:
        print()

    if not video_results:
        details = '; '.join(f"{item['path']}: {item['error']}" for item in errors)
        raise ValueError(f"no valid videos: {details or 'no video inputs'}")

    sources = [Path(src) for src in as_collection(videos)]
    source_dir = (sources[0] if len(sources) == 1 and sources[0].is_dir()
                  else Path(os.path.commonpath([str(v.parent) for v in video_paths] )))
    for result in video_results:
        result['path'] = os.path.relpath(result['path'], source_dir)

    fps_values = [result['fps_measured'] for result in video_results]
    durations = [result['duration'] for result in video_results]
    mean, std = _weighted_stats(fps_values, np.ones(len(fps_values)))
    weighted_mean, weighted_std = _weighted_stats(fps_values, durations)
    all_samples = (sample for result in video_results for sample in result['samples'])
    mean_diff, mean_mf_diff = _diff_means(all_samples, mf_threshold)
    inconsistent = [result for result in video_results
                    if not result['fps_consistent']]
    near_static = [result for result in video_results
                   if result['near_static']]
    report = _round_report({ 'video_dir': str(source_dir),
               'fps_measureds': {'mean': round(mean, DECI_PRECISION),
                                 'std': round(std, DECI_PRECISION),
                                 'weighted_mean': round(weighted_mean, DECI_PRECISION),
                                 'weighted_std': round(weighted_std, DECI_PRECISION), },
               'mf_threshold': round(mf_threshold, DECI_PRECISION),
               'min_eligible_fps': round(MIN_ELIGIBLE_FPS, DECI_PRECISION),
               'fps_consistency': {
                   'count': len(inconsistent),
                   'files': [result['path'] for result in inconsistent],
               },
               'near_static_files': {
                   'count': len(near_static),
                   'files': [result['path'] for result in near_static],
               },
               'mean_diff': round(mean_diff, DECI_PRECISION),
               'mean_mf_diff': (round(mean_mf_diff, DECI_PRECISION)
                                if mean_mf_diff is not None else None),
               'videos': video_results,
               'errors': errors,
               })
    if save_results:
        output_path = report_path()  if save_results is True else save_results
        _save_report(report, output_path)
    if print_res:
        print_eff_fps(report)
    return weighted_mean, report


get_effective_fps = get_measured_fps


def load_eff_fps_report(report_path) -> dict:
    """ Load a saved effective-FPS report from a JSON file."""
    report_path = Path(report_path)
    with report_path.open('r', encoding='utf-8') as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError('effective-FPS report must contain a JSON object')
    return report


def print_eff_fps(results:dict, **kwargs) -> None:
    """Print effective FPS results in standard or total-only table form."""

    legacy_report = 'fps_measureds' not in results
    legacy_inconsistency = results.get('inconsistent_fps_files')

    def measured(res):
        return res.get('fps_measured', res.get('fps_eff'))

    def measured_std(res):
        return res.get('fps_measured_std', res.get('fps_eff_std'))

    def near_static(res):
        return res.get('near_static', measured(res) < min_measured_fps)

    def consistent(res):
        if 'fps_consistent' in res:
            return res['fps_consistent']
        if isinstance(legacy_inconsistency, dict):
            return res['path'] not in set(legacy_inconsistency.get('files', []))
        fps_msr = measured(res)
        return (fps_msr > 0 and
                abs(fps_msr - res['fps_encoded'])/res['fps_encoded'] <= FPS_CONSISTENCY_RTOL)

    # def print_table(tbl_rows, hdrs, min_width=0, left_labels=False):
    #     widths = [max(min_width, len(str(header)), *(len(str(r[i])) for r in tbl_rows))
    #               for i, header in enumerate(hdrs)]
    #     separator = '-+-'.join('-' * width for width in widths)
    #     print(' | '.join(f'{header:^{width}}' for header, width in zip(hdrs, widths)))
    #     print(separator)
    #     for row in tbl_rows:
    #         cells = [f'{row[0]:<{widths[0]}}' if left_labels else f'{row[0]:^{widths[0]}}']
    #         cells += [f'{val:^{width}}' for val, width in zip(row[1:], widths[1:])]
    #         print(' | '.join(cells))

    def _frmt_val(val):
        return '--' if val is None else f'{val:.2f}'

    def ratio(res):
        fps_msr = measured(res)
        return res['fps_encoded'] / fps_msr if fps_msr > 0 else None

    def stats_ratio(res):
        return ratio(res) if not near_static(res) else None

    def _frmt_ratio(val):
        if val is None:
            return '--'
        return 'inf' if val > 500 else f'{val:.2f}'

    def sort_value(result):
        if sort == 'video':
            return result['titel'].lower()
        if sort == 'encoded':
            return result['fps_encoded']
        if sort == 'effective':
            return measured(result)
        value = ratio(result)
        return float('inf') if value is not None and value > 500 else value

    def average(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals)/len(vals) if vals else None

    def mean_std(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return _weighted_stats(vals, np.ones(len(vals)))

    def stats_cell(vals, ratio_stats=False):
        mean, std = mean_std(vals)
        if mean is None:
            return 'N/A'
        mean_text = ('inf' if ratio_stats and mean > 500 else
                     Decimal(str(mean)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        std_text = ('0' if std == 0 else
                    Decimal(str(std)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        return f'{mean_text} ({std_text})'

    sort_arg = kwargs.pop('sort', None)
    sort = 'video' if sort_arg is None else sort_arg
    show_num = sort_arg is not None
    total_only = kwargs.pop('total_only', False)
    rows = kwargs.pop('rows', None)
    order = kwargs.pop('order', 'ascending')
    if kwargs:
        unknown = ', '.join(sorted(kwargs))
        raise TypeError(f'unexpected keyword argument(s): {unknown}')

    sort = {'vid': 'video', 'enc': 'encoded', 'eff': 'effective'}.get( str(sort).lower(), str(sort).lower())
    order = {'asc': 'ascending', 'dsc': 'descending'}.get( str(order).lower(), str(order).lower())
    if sort not in {'video', 'encoded', 'effective', 'ratio'}:
        raise ValueError("sort must be 'video', 'encoded', 'effective', or 'ratio'")
    if order not in {'ascending', 'descending'}:
        raise ValueError("order must be 'ascending' or 'descending'")
    if rows is not None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError('rows must be positive')

    min_measured_fps = results.get( 'min_eligible_fps',
                       results.get( 'min_eff_fps_for_ratio', MIN_ELIGIBLE_FPS))
    fps_stats = results.get('fps_measureds', results.get('fps_effs'))
    fps_label = 'Effective FPS' if legacy_report else 'Measured FPS'
    print(f"Meaningful threshold: {results['mf_threshold']}")
    print(f"{fps_label}: mean={fps_stats['mean']:.3f}, std={fps_stats['std']:.3f}, "
          f"weighted mean={fps_stats['weighted_mean']:.3f}, "
          f"weighted std={fps_stats['weighted_std']:.3f}")
    print(f"Mean diff: {results['mean_diff']:.3f}; mean MF diff: {_format_optional(results['mean_mf_diff'])}")

    vid_results = results['videos']
    descending = order == 'descending'
    if sort == 'video':
        ordered = sorted(vid_results, key=sort_value, reverse=descending)
    else:
        sortable  =   [vr for vr in vid_results if sort_value(vr) is not None]
        un_sortable = [vr for vr in vid_results if sort_value(vr) is None]
        ordered = (sorted(sortable, key=sort_value, reverse=descending) + un_sortable)

    if total_only:
        print(f'\n{fps_label} Summary')
        summary_rows = []
        columns = { 'Encoded FPS': 'fps_encoded',
                    fps_label: None,
                    f'{fps_label} std': None,
                    'Dif ratio': None,
                    'Mean diff': 'mean_diff',
                    'Mean MF diff': 'mean_mf_diff',
                    'Duration': 'duration',
                    'Sampled duration': 'sampled_duration',
                    }
        for lbl, key in columns.items():
            if lbl == fps_label:
                values = [measured(res) for res in vid_results]
            elif lbl == f'{fps_label} std':
                values = [measured_std(res) for res in vid_results]
            else:
                values = [stats_ratio(res) if key is None else res[key]
                          for res in vid_results]
            values = [v for v in values if v is not None]
            formatter = _frmt_ratio if lbl == 'Dif ratio' else _frmt_val
            summary_rows.append((lbl, formatter(average(values)),
                                        formatter(max(values, default=None)),
                                        formatter(min(values, default=None)) ))
        print_table(summary_rows, ('Statistic', 'Mean', 'Max', 'Min'))
    else:
        displayed = ordered[:rows] if rows is not None else ordered
        table_rows = []
        for row_idx, result in enumerate(displayed, start=1):
            row = (result['titel'][:21], _frmt_val(result['duration']),
                               _frmt_val(result['sampled_duration']),
                               _frmt_val(result['fps_encoded']),
                               _frmt_val(measured(result)),
                               _frmt_val(measured_std(result)),
                               _frmt_ratio(ratio(result)),
                               _frmt_val(result['mean_diff']),
                               'X' if not consistent(result) else '',
                               'X' if near_static(result) else '')
            table_rows.append((str(row_idx), *row) if show_num else row)
        average_row = [ 'Average',
                        *(_frmt_val(average([result[key] for result in vid_results]))
                        for key in ('duration', 'sampled_duration', 'fps_encoded')),
                        _frmt_val  (average([measured(vr)     for vr in vid_results])),
                        _frmt_val  (average([measured_std(vr) for vr in vid_results])),
                        _frmt_ratio(average([stats_ratio(vr)  for vr in vid_results])),
                        _frmt_val  (average([vr['mean_diff']  for vr in vid_results])),
                        '', '', ]
        if show_num:
            average_row.insert(0, '')
        average_row = tuple(average_row)
        print()
        headers = ('Video', 'Duration', 'Sampled', 'FPS_enc', fps_label,
                   'Eff std', 'Dif ratio', 'Mean Diff', 'Not-Consis.', 'n.static')
        if show_num:
            headers = ('Num', *headers)
        print_table(table_rows + [average_row], headers)

    print(f"\nVideo directory: {results.get('video_dir', '--')}")
    print('FPS consistency')
    near_static_results = [vr for vr in vid_results if near_static(vr)]
    if legacy_report and not isinstance(legacy_inconsistency, dict):
        consistency_rows = [ ('Files', len(vid_results), 'N/A', 'N/A'),
                             ('Near static', len(near_static_results), 'N/A', 'N/A'),
                             ('Encoded FPS', stats_cell([vr['fps_encoded'] for vr in vid_results]),
                              'N/A', 'N/A'),
                             ('Effective FPS', stats_cell([vr['fps_eff'] for vr in vid_results]),
                              'N/A', 'N/A'),
                             ('Diff ratio', stats_cell([stats_ratio(vr) for vr in vid_results], ratio_stats=True),
                              'N/A', 'N/A'),
                            ]
    else:
        inconsistent_results = [vr for vr in vid_results if not consistent(vr)]
        consistent_results   = [vr for vr in vid_results if consistent(vr)]
        consistent_near_static = [vr for vr in consistent_results if near_static(vr)]
        inconsistent_near_static = [vr for vr in inconsistent_results if near_static(vr)]
        consistency_rows = [ ('Files', len(vid_results), len(consistent_results), len(inconsistent_results)),
                              ('Near static', len(near_static_results),
                             len(consistent_near_static),
                             len(inconsistent_near_static)),
                            ('Encoded FPS', stats_cell([result['fps_encoded'] for result in vid_results]),
                             stats_cell([r['fps_encoded'] for r in consistent_results]),
                             stats_cell([r['fps_encoded'] for r in inconsistent_results])),
                            (fps_label, stats_cell([measured(r) for r in vid_results]),
                             stats_cell([measured(r) for r in consistent_results]),
                             stats_cell([measured(r) for r in inconsistent_results])),
                            ('Diff ratio', stats_cell([stats_ratio(r) for r in vid_results], ratio_stats=True),
                             stats_cell([stats_ratio(r) for r in consistent_results], ratio_stats=True),
                             stats_cell([stats_ratio(r) for r in inconsistent_results], ratio_stats=True)),
                            ]
    print_table(consistency_rows, ('', 'Total', 'Consistent', 'Inconsistent'),
                min_width=FPS_SUMMARY_COL_WIDTH, left_labels=True)
    for err in results['errors']:
        print(f"[WARN] {err['path']}: {err['error']}")


def print_table(tbl_rows, headers, min_width=0, left_labels=False):
    widths = [max(min_width, len(str(header)), *(len(str(r[i])) for r in tbl_rows))
                                                for i, header in enumerate(headers)]
    separator = '-+-'.join('-' * width for width in widths)
    print(' | '.join(f'{header:^{width}}' for header, width in zip(headers, widths)))
    print(separator)
    for row in tbl_rows:
        cells = [f'{row[0]:<{widths[0]}}' if left_labels else f'{row[0]:^{widths[0]}}']
        cells += [f'{val:^{width}}' for val, width in zip(row[1:], widths[1:])]
        print(' | '.join(cells))

# endregion

# region Helpers
def _round_report(value):
    """ Round report numbers for the high-level utility API."""
    if isinstance(value, dict):
        return {key: _round_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_report(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), DECI_PRECISION)
    return value


def _fps_ratio(fps_encoded, fps_msr, min_msr_fps=MIN_ELIGIBLE_FPS):
    """Return encoded/measured FPS when measured FPS meets the ratio floor."""
    return fps_encoded/fps_msr if fps_msr >= min_msr_fps else None


def _validate_threshold(value):
    """Validate and normalize the meaningful-frame threshold."""
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError('mf_threshold must be finite and nonnegative')
    return value


def _resolve_video_inputs(videos, recursive):
    """ Resolve paths and directories into supported video files."""
    video_paths, errors, seen = [], [], set()
    for source in as_collection(videos):
        path = Path(source)
        if path.is_dir():
            candidates = path.rglob('*') if recursive else path.iterdir()
            for candidate in candidates:
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
                    if candidate not in seen:
                        seen.add(candidate)
                        video_paths.append(candidate)
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            if path not in seen:
                seen.add(path)
                video_paths.append(path)
        else:
            errors.append({'path': str(path), 'error': 'unsupported or missing video'})
    return sorted(video_paths), errors


def _weighted_stats(values, weights):
    """ Return weighted population mean and standard deviation."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean)**2, weights=weights))
    return float(mean), float(std)


def _diff_means(samples, mf_threshold):
    """ Return averages across all differences and meaningful differences."""
    total = mf_total = 0.0
    count = mf_count = 0
    for smp in samples:
        for diff in smp['diffs']:
            total += diff
            count += 1
            if diff >= mf_threshold:
                mf_total += diff
                mf_count += 1
    return total/count, mf_total/mf_count if mf_count else None


def _save_report(report, output_path):
    """ Save one complete effective-FPS report as JSON."""
    output_path = Path(output_path)
    if output_path.is_dir():
        output_path /= DEFAULT_EFF_FPS_REPORT
    elif output_path.suffix == '':
        output_path = output_path.with_suffix('.json')
    output_path = get_unique_name(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

def _load_report(report_path) -> dict:
    """ Load a saved effective-FPS report from a JSON file."""
    report_path = Path(report_path)
    with report_path.open('r', encoding='utf-8') as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError('effective-FPS report must contain a JSON object')
    return report


def _format_optional(value, width=0):
    """ Format an optional numeric value for the text report."""
    txt = '--' if value is None else f'{value:.3f}'
    return txt.rjust(width)

# endregion

#387(1,4,1) 400(1,2,)->379(1,1,1)
#505(1,2,5) #415(1,2,5)
#460(1,2,8)->#471(1,1,8)

if __name__ == '__main__': # pass

    tst_report = load_eff_fps_report("/mnt/local-data/Projects/Wesmart/Video-datasets/UBI_FIGHTS/eff_fps_ubi.json")
    print_eff_fps(tst_report, sort='ratio', rows=100)
