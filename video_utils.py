import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import cv2
import numpy as np

from common.my_local_utils import as_collection, get_unique_name, print_progress
from video_analytics import DEFAULT_MF_TH, measure_video_effective_fps


INCONSISTENT_FPS_RATIO_TH = 1.1
MIN_EFF_FPS_FOR_RATIO = 0.5
VIDEO_SUFFIXES = {'.mp4', '.avi', '.wmv', '.flv', '.mkv', '.mov', '.m4v'}
DEFAULT_EFF_FPS_REPORT = 'effective_fps.json'
FPS_SUMMARY_COL_WIDTH = 16


# region API
def get_effective_fps(videos, mf_threshold=DEFAULT_MF_TH, **kwargs) -> tuple[float, dict]:
    """ Measure one video or a collection/directory and aggregate its results.
    Directories are searched recursively by default. Invalid inputs are kept in
    `errors` and do not prevent the remaining videos from being analyzed.
    The returned FPS is the duration-weighted mean effective FPS.
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
    total_videos = len(video_paths)
    completed = 0
    for video in video_paths:
        try:
            _, result = measure_video_effective_fps(video, mf_threshold)
            video_results.append(result)
        except (OSError, ValueError, TypeError, cv2.error) as err:
            errors.append({'path': str(video), 'error': f'{type(err).__name__}: {err}'})
        finally:
            if total_videos > 1:
                completed += 1
                print_progress(total_videos, completed, mode='single_line',
                               current=video.name)
    if total_videos > 1:
        print()

    if not video_results:
        details = '; '.join(f"{item['path']}: {item['error']}" for item in errors)
        raise ValueError(f"no valid videos: {details or 'no video inputs'}")

    sources = [Path(source) for source in as_collection(videos)]
    source_dir = (sources[0] if len(sources) == 1 and sources[0].is_dir()
                  else Path(os.path.commonpath([str(video.parent) for video in video_paths])))
    for result in video_results:
        result['path'] = os.path.relpath(result['path'], source_dir)

    fps_values = [result['fps_eff'] for result in video_results]
    durations = [result['duration'] for result in video_results]
    mean, std = _weighted_stats(fps_values, np.ones(len(fps_values)))
    weighted_mean, weighted_std = _weighted_stats(fps_values, durations)
    all_samples = (sample for result in video_results for sample in result['samples'])
    mean_diff, mean_mf_diff = _diff_means(all_samples, mf_threshold)
    inconsistent_fps = []
    near_static_count = 0
    for res in video_results:
        fps_ratio = _fps_ratio(res['fps_encoded'], res['fps_eff'], MIN_EFF_FPS_FOR_RATIO)
        if fps_ratio is None:
            near_static_count += 1
        if fps_ratio is None or fps_ratio > INCONSISTENT_FPS_RATIO_TH:
            inconsistent_fps.append(res)
    report = { 'video_dir': str(source_dir),
               'fps_effs': {'mean': round(mean, 5),
                            'std': round(std, 5),
                            'weighted_mean': round(weighted_mean, 5),
                            'weighted_std': round(weighted_std, 5), },
               'mf_threshold': round(mf_threshold, 5),
               'min_eff_fps_for_ratio': round(MIN_EFF_FPS_FOR_RATIO, 5),
               'inconsistent_fps_files': {
                   'count': len(inconsistent_fps),
                   'near_static_count': near_static_count,
                   'files': [result['path'] for result in inconsistent_fps],
               },
               'mean_diff': round(mean_diff, 5),
               'mean_mf_diff': (round(mean_mf_diff, 5)
                                if mean_mf_diff is not None else None),
               'videos': video_results,
               'errors': errors,
               }
    if save_results:
        output_path = report_path()  if save_results is True else save_results
        _save_report(report, output_path)
    if print_res:
        print_eff_fps(report)
    return weighted_mean, report


def load_eff_fps_report(report_path) -> dict:
    """ Load a saved effective-FPS report from a JSON file."""
    report_path = Path(report_path)
    with report_path.open('r', encoding='utf-8') as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError('effective-FPS report must contain a JSON object')
    return report


def print_eff_fps(results: dict, **kwargs) -> None:
    """Print effective FPS results in standard or total-only table form."""

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

    def ratio(result):
        fps_eff = result['fps_eff']
        return result['fps_encoded'] / fps_eff if fps_eff > 0 else None

    def stats_ratio(result):
        return ratio(result) if result['fps_eff'] >= min_eff_fps else None

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
            return result['fps_eff']
        value = ratio(result)
        return float('inf') if value is not None and value > 500 else value

    def average(vals):
        vals = [value for value in vals if value is not None]
        return sum(vals) / len(vals) if vals else None

    def mean_std(vals):
        vals = [value for value in vals if value is not None]
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

    sort = {'vid': 'video', 'enc': 'encoded', 'eff': 'effective'}.get(
        str(sort).lower(), str(sort).lower())
    order = {'asc': 'ascending', 'dsc': 'descending'}.get(
        str(order).lower(), str(order).lower())
    if sort not in {'video', 'encoded', 'effective', 'ratio'}:
        raise ValueError("sort must be 'video', 'encoded', 'effective', or 'ratio'")
    if order not in {'ascending', 'descending'}:
        raise ValueError("order must be 'ascending' or 'descending'")
    if rows is not None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError('rows must be positive')

    min_eff_fps = results.get('min_eff_fps_for_ratio', MIN_EFF_FPS_FOR_RATIO)
    fps_effs = results['fps_effs']
    print(f"Meaningful threshold: {results['mf_threshold']}")
    print(f"Effective FPS: mean={fps_effs['mean']:.3f}, std={fps_effs['std']:.3f}, "
          f"weighted mean={fps_effs['weighted_mean']:.3f}, "
          f"weighted std={fps_effs['weighted_std']:.3f}")
    print(f"Mean diff: {results['mean_diff']:.3f}; "
          f"mean MF diff: {_format_optional(results['mean_mf_diff'])}")

    video_results = results['videos']
    inconsistency = results.get('inconsistent_fps_files')
    if isinstance(inconsistency, dict):
        inconsistent_paths = set(inconsistency.get('files', []))
    else:
        inconsistent_paths = {result['path'] for result in video_results
                              if (ratio(result) is None
                                  or ratio(result) > INCONSISTENT_FPS_RATIO_TH)}
    descending = order == 'descending'
    if sort == 'video':
        ordered = sorted(video_results, key=sort_value, reverse=descending)
    else:
        sortable = [result for result in video_results if sort_value(result) is not None]
        un_sortable = [result for result in video_results if sort_value(result) is None]
        ordered = (sorted(sortable, key=sort_value, reverse=descending) + un_sortable)

    if total_only:
        print('\nEffective FPS Summary')
        summary_rows = []
        columns = { 'Encoded FPS': 'fps_encoded',
                    'Effective FPS': 'fps_eff',
                    'Effective FPS std': 'fps_eff_std',
                    'Dif ratio': None,
                    'Mean diff': 'mean_diff',
                    'Mean MF diff': 'mean_mf_diff',
                    'Duration': 'duration',
                    'Sampled duration': 'sampled_duration',
                    }
        for label, key in columns.items():
            values = [stats_ratio(res) if key is None else res[key] for res in video_results]
            values = [v for v in values if v is not None]
            formatter = _frmt_ratio if key is None else _frmt_val
            summary_rows.append((label, formatter(average(values)),
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
                               _frmt_val(result['fps_eff']),
                               _frmt_val(result['fps_eff_std']),
                               _frmt_ratio(ratio(result)),
                               _frmt_val(result['mean_diff']),
                               'X' if result['path'] in inconsistent_paths else '',
                               'X' if result['fps_eff'] < min_eff_fps else '')
            table_rows.append((str(row_idx), *row) if show_num else row)
        average_row = (*(() if not show_num else ('',)),
                       'Average', *(_frmt_val(average(
            [result[key] for result in video_results]))
            for key in ('duration', 'sampled_duration', 'fps_encoded', 'fps_eff', 'fps_eff_std')),
            _frmt_ratio(average([stats_ratio(result) for result in video_results])),
            _frmt_val(average([result['mean_diff'] for result in video_results])),
            '', '')
        print()
        headers = ('Video', 'Duration', 'Sampled', 'FPS_enc', 'FPS_eff',
                   'Eff std', 'Dif ratio', 'Mean Diff', 'Not-Consis.', 'n.static')
        if show_num:
            headers = ('Num', *headers)
        print_table(table_rows + [average_row],
                    headers)

    print(f"\nVideo directory: {results.get('video_dir', '--')}")
    print('FPS consistency')
    near_static_results = [result for result in video_results
                           if result['fps_eff'] < min_eff_fps]
    if not isinstance(inconsistency, dict):
        consistency_rows = [
            ('Files', len(video_results), 'N/A', 'N/A'),
            ('Near static', len(near_static_results), 'N/A', 'N/A'),
            ('Encoded FPS', stats_cell([result['fps_encoded'] for result in video_results]),
             'N/A', 'N/A'),
            ('Effective FPS', stats_cell([result['fps_eff'] for result in video_results]),
             'N/A', 'N/A'),
            ('Diff ratio', stats_cell([stats_ratio(result) for result in video_results], ratio_stats=True),
             'N/A', 'N/A'),
        ]
    else:
        inconsistent_paths = set(inconsistency.get('files', []))
        inconsistent_results = [result for result in video_results
                                if result['path'] in inconsistent_paths]
        consistent_results = [result for result in video_results
                              if result['path'] not in inconsistent_paths]
        consistency_rows = [ ('Files', len(video_results), len(consistent_results), len(inconsistent_results)),
                            ('Near static', len(near_static_results), 0, len(near_static_results)),
                            ('Encoded FPS', stats_cell([result['fps_encoded'] for result in video_results]),
                             stats_cell([result['fps_encoded'] for result in consistent_results]),
                             stats_cell([result['fps_encoded'] for result in inconsistent_results])),
                            ('Effective FPS', stats_cell([result['fps_eff'] for result in video_results]),
                             stats_cell([result['fps_eff'] for result in consistent_results]),
                             stats_cell([result['fps_eff'] for result in inconsistent_results])),
                            ('Diff ratio', stats_cell([stats_ratio(result) for result in video_results], ratio_stats=True),
                             stats_cell([stats_ratio(result) for result in consistent_results], ratio_stats=True),
                             stats_cell([stats_ratio(result) for result in inconsistent_results], ratio_stats=True)),
                            ]
    print_table(consistency_rows, ('', 'Total', 'Consistent', 'Inconsistent'),
                min_width=FPS_SUMMARY_COL_WIDTH, left_labels=True)
    for error in results['errors']:
        print(f"[WARN] {error['path']}: {error['error']}")


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
def _fps_ratio(fps_encoded, fps_eff, min_eff_fps=MIN_EFF_FPS_FOR_RATIO):
    """Return encoded/effective FPS when effective FPS meets the ratio floor."""
    return fps_encoded/fps_eff if fps_eff >= min_eff_fps else None


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

if __name__ == '__main__': # pass

    tst_report = load_eff_fps_report("/mnt/local-data/Projects/Wesmart/Video-datasets/UBI_FIGHTS/eff_fps_ubi.json")
    print_eff_fps(tst_report, sort='ratio', rows=100)
