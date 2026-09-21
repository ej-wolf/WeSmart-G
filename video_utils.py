import os, json
from pathlib import Path
import cv2
import numpy as np

from common.my_local_utils import as_collection, get_unique_name, print_progress
from common.table_utils import print_table
from video_analytics import measure_video_fps, DEFAULT_MF_THRESHOLD, FPS_CONSISTENCY_RTOL

DECI_PRECISION = 5
INF_THRESHOLD = 500
MIN_ELIGIBLE_FPS = 0.5
VIDEO_SUFFIXES = {'.mp4', '.avi', '.wmv', '.flv', '.mkv', '.mov', '.m4v'}
DEFAULT_FPS_REPORT = 'measured_fps.json'
FPS_SUMMARY_COL_WIDTH = 16


# region API
def get_measured_fps(videos, mf_threshold=DEFAULT_MF_THRESHOLD, **kwargs) -> tuple[float, dict]:
    """ Measure visual FPS for one or more videos and aggregate the results.
    :param videos:  Inputs video/s, a file, dir/s, or collection of video paths.
    :param mf_threshold: Meaningful-Frame motion threshold for FPS measurement.
    optional
        recursive: Search directories recursively.
        save_results: path for saving the report.
        print_res: print to terminal the report after analysis.
    :return : measured FPS/ weighted mean fps(for multi-videos ) , full report (dict)
    """

    def resolve_video_inputs():
        """Resolve supplied files and directories into supported video paths."""
        paths, err, seen = [], [], set()
        for source in as_collection(videos):
            path = Path(source)
            if path.is_dir():
                candidates = path.rglob('*') if recursive else path.iterdir()
                for candidate in candidates:
                    if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
                        if candidate not in seen:
                            seen.add(candidate)
                            paths.append(candidate)
            elif path.suffix.lower() in VIDEO_SUFFIXES:
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
            else:
                err.append({'path': str(path),
                               'error': 'unsupported or missing video'})
        return sorted(paths), err

    def report_path():
        """ Return the default report path for the supplied video inputs."""
        src = as_collection(videos)
        if len(src) == 1 and Path(src[0]).is_dir():
            report_dir = Path(src[0])
        elif video_paths:
            report_dir = video_paths[0].parent
        else:
            report_dir = Path.cwd()
        return report_dir/DEFAULT_FPS_REPORT

    recursive = kwargs.get('recursive', True)
    save_results = kwargs.get('save_to', None)
    print_res = kwargs.get('print_res', False)

    mf_threshold = float(mf_threshold)
    if not np.isfinite(mf_threshold) or mf_threshold < 0:
        raise ValueError('mf_threshold must be finite and non-negative')
    video_paths, input_errors = resolve_video_inputs()
    video_results, errors = [], list(input_errors)
    total_vids = len(video_paths)
    completed = 0

    for vp in video_paths:
        try:
            _, result = measure_video_fps(vp, mf_threshold)
            video_results.append(result)
        except (OSError, ValueError, TypeError, cv2.error) as error:
            errors.append({'path': str(vp), 'error': f'{type(error).__name__}: {error}'})
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
    source_dir = (sources[0] if len(sources) == 1 and sources[0].is_dir() else
                  Path(os.path.commonpath([str(v.parent) for v in video_paths] )))
    for result in video_results:
        result['path'] = os.path.relpath(result['path'], source_dir)

    fps_values = [vr['fps_measured'] for vr in video_results]
    durations  = [vr['duration'] for vr in video_results]
    mean, std = _weighted_stats(fps_values, np.ones(len(fps_values)))
    weighted_mean, weighted_std = _weighted_stats(fps_values, durations)
    all_samples = (smp for vr in video_results for smp in vr['samples'])
    mean_diff, mean_mf_diff = _diff_means(all_samples, mf_threshold)
    inconsistent = [vr for vr in video_results if not vr['fps_consistent']]
    near_static  = [vr for vr in video_results if vr['near_static']]
    report = _round_report({ 'video_dir': str(source_dir),
                             'fps_stats': {'mean': mean, 'std': std,
                                               'weighted_mean': weighted_mean,
                                               'weighted_std': weighted_std, },
                             'mf_threshold': mf_threshold,
                             'min_eligible_fps': MIN_ELIGIBLE_FPS,
                             'fps_consistency' : {'count': len(inconsistent),
                                                  'files': [vr['path'] for vr in inconsistent],},
                             'near_static_files':{'count': len(near_static),
                                                  'files': [vr['path'] for vr in near_static], },
                             'mean_diff': mean_diff,
                             'mean_mf_diff': mean_mf_diff,
                             'videos': video_results,
                             'errors': errors,
                             })
    if save_results:
        output_path = report_path()  if save_results is True else save_results
        save_fps_report(report, output_path)
    if print_res:
        print_fps_report(report)
    return weighted_mean, report


def load_fps_report(report_path) -> dict:
    """Load a saved measured-FPS report from a JSON file."""
    report_path = Path(report_path)
    with report_path.open('r', encoding='utf-8') as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError('measured-FPS report must contain a JSON object')
    return report


def print_fps_report(report:dict, **kwargs) -> None:
    """Print measured FPS results in standard or total-only table form."""
    LEGACY_MAIN_LABEL = "fps_measureds"

    def _normalize_legacy_report(): #* 42
        """Normalize historical report schemas in place and return the report."""
        if 'fps_stats' in report:  #* exist only in new format or legacy fixed
            return report
        if LEGACY_MAIN_LABEL in report:
            report['fps_stats'] = report.pop(LEGACY_MAIN_LABEL)
        if 'min_measured_fps_for_ratio' in report:
            report['min_eligible_fps'] = report.pop('min_measured_fps_for_ratio')
        if 'fps_stats' in report:
            return report

        legacy_stats = report.pop('fps_effs')
        legacy_min_fps = report.pop('min_eff_fps_for_ratio', MIN_ELIGIBLE_FPS)
        legacy_consis = report.pop('inconsistent_fps_files', None)
        consis_meta = isinstance(legacy_consis, dict)
        legacy_paths = (set(legacy_consis.get('files', [])) if consis_meta else set())

        report['fps_stats'] = legacy_stats
        report['min_eligible_fps'] = legacy_min_fps
        for vid in report['videos']:
            vid['fps_measured'] = vid.pop('fps_eff')
            vid['fps_measured_std'] = vid.pop('fps_eff_std', None)
            vid['near_static'] = vid['fps_measured'] < legacy_min_fps
            vid['fps_consistent'] = ( vid['path'] not in legacy_paths if consis_meta else
                                      vid['fps_measured'] > 0 and  abs(vid['fps_measured'] - vid['fps_encoded'])/vid['fps_encoded']
                                      <= FPS_CONSISTENCY_RTOL)

        inconsistent = [vid for vid in report['videos'] if not vid['fps_consistent']]
        near_static  = [vid for vid in report['videos'] if vid['near_static']]
        report['fps_consistency'] = {'count': len(inconsistent),
                                     'files': [vid['path'] for vid in inconsistent], }
        report['near_static_files']= {'count': len(near_static),
                                      'files': [vid['path'] for vid in near_static], }
        report['_legacy_report'] = True
        report['_fps_label'] = 'Effective FPS'
        report['_consistency_known'] = consis_meta
        return report

    def _frmt_val(val):
        return '--' if val is None else f'{val:.2f}'

    def ratio(res):
        return res['fps_encoded']/res['fps_measured'] if res['fps_measured'] > 0 else None

    def stats_ratio(res):
        return ratio(res) if not res['near_static'] else None

    def sort_value(result):
        if sort == 'video':
            return result['titel'].lower()
        if sort == 'encoded':
            return result['fps_encoded']
        if sort == 'measured':
            return result['fps_measured']
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

    def _frmt_mean_std(mean, std, inf_th=None):
        if mean is None:
            return 'N/A'
        if inf_th is not None and mean > inf_th:
            return 'inf (N/A)'
        mean_text = f'{mean:5.2f}'
        std_text =  '(0)' if std == 0 else  f'({std:.2f})'
        return f"{mean_text} {std_text:6}"

    report = _normalize_legacy_report()
    consistency_known = report.get('_consistency_known', True)

    sort_arg = kwargs.pop('sort', None)
    sort = 'video' if sort_arg is None else sort_arg
    order = kwargs.pop('order', 'ascending')

    total_only = kwargs.pop('total_only', False)
    rows = kwargs.pop('rows', None)
    col_width = kwargs.pop('col_width', 'smart-opt')
    if kwargs:
        unknown = ', '.join(sorted(kwargs))
        raise TypeError(f'unexpected keyword argument(s): {unknown}')

    sort = {'vid': 'video', 'enc': 'encoded', 'msr': 'measured'}.get( str(sort).lower(), str(sort).lower())
    order = {'asc': 'ascending', 'dsc': 'descending'}.get( str(order).lower(), str(order).lower())
    if sort not in {'video', 'encoded', 'measured', 'ratio'}:
        raise ValueError("sort must be 'video', 'encoded', 'measured', or 'ratio'")
    if order not in {'ascending', 'descending'}:
        raise ValueError("order must be 'ascending' or 'descending'")
    if rows is not None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError('rows must be positive')

    fps_stats = report['fps_stats']
    fps_label = report.get('_fps_label', 'Measured FPS')
    print(f"Meaningful threshold: {report['mf_threshold']}"
          f"{fps_label}: mean = {fps_stats['mean']:.3f}, std = {fps_stats['std']:.3f},\n"
          f"{'Weighted':12}: mean = {fps_stats['weighted_mean']:.3f}, "
          f"std = {fps_stats['weighted_std']:.3f}\n"
          f"Mean diff: {report['mean_diff']:.3f}; "
          f"Mean MF diff: {_frmt_val(report['mean_mf_diff'])}\n")

    vid_results = report['videos']
    descending = order == 'descending'
    if sort == 'video':
        ordered = sorted(vid_results, key=sort_value, reverse=descending)
    else:
        sortable  =   [vr for vr in vid_results if sort_value(vr) is not None]
        un_sortable = [vr for vr in vid_results if sort_value(vr) is None]
        ordered = (sorted(sortable, key=sort_value, reverse=descending) + un_sortable)

    if total_only:
        print(f'\n{fps_label} Summary')
        table_rows = []
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
                values = [vr['fps_measured'] for vr in vid_results]
            elif lbl == f'{fps_label} std':
                values = [vr['fps_measured_std'] for vr in vid_results]
            else:
                values = [stats_ratio(vr) if key is None else vr[key] for vr in vid_results]
            values = [v for v in values if v is not None]
            table_rows.append((lbl, _frmt_val(average(values)),
                                      _frmt_val(max(values, default=None)),
                                      _frmt_val(min(values, default=None)) ))
        print_table(table_rows, ('Statistic', 'Mean', 'Max', 'Min'))
    else:
        displayed = ordered[:rows] if rows is not None else ordered
        table_rows = []
        for row_idx, result in enumerate(displayed, start=1):
            row = (result['titel'][:21], _frmt_val(result['duration']),
                               _frmt_val(result['sampled_duration']),
                               _frmt_val(result['fps_encoded']),
                               _frmt_val(result['fps_measured']),
                               _frmt_val(result['fps_measured_std']),
                               _frmt_val(ratio(result)),
                               _frmt_val(result['mean_diff']),
                               'X' if not result['fps_consistent'] else '',
                               'X' if result['near_static'] else '')
            table_rows.append((str(row_idx), *row) if sort_arg is not None else row)
        avg_row = ['Average',
                    *(_frmt_val(average([vr[key] for vr in vid_results]))
                    for key in ('duration', 'sampled_duration', 'fps_encoded')),
                    _frmt_val (average([vr['fps_measured']     for vr in vid_results])),
                    _frmt_val (average([vr['fps_measured_std'] for vr in vid_results])),
                    _frmt_val (average([stats_ratio(vr)  for vr in vid_results])),
                    _frmt_val (average([vr['mean_diff']  for vr in vid_results])),
                    '', '', ]

        headers = ('Video', 'Duration', 'Sampled', 'FPS_enc', fps_label,
                   'Measured std', 'Dif ratio', 'Mean Diff', 'Not-Consis.', 'n.static')
        if sort_arg is not None:
            headers = ('Num', *headers)
            avg_row.insert(0, '')
        avg_row = tuple(avg_row)
        fps_col = 5 if sort_arg is not None else 4
        print_table(table_rows + [avg_row], headers, col_width =col_width,
                    B={'cols': [fps_col]}, lines=[len(table_rows)], )

    print(f"\nVideo directory: {report.get('video_dir', '--')}")
    print('FPS consistency')
    static_vids = [vr for vr in vid_results if vr['near_static']]
    if not consistency_known:
        table_rows = [('Files', len(vid_results), 'N/A', 'N/A'),
                      ('Near static', len(static_vids), 'N/A', 'N/A'),
                      ('Encoded FPS', _frmt_mean_std(*mean_std([vr['fps_encoded'] for vr in vid_results])), 'N/A', 'N/A'),
                      (fps_label   , _frmt_mean_std(*mean_std([vr['fps_measured'] for vr in vid_results])), 'N/A', 'N/A'),
                      ('Diff ratio', _frmt_mean_std(*mean_std([stats_ratio(vr) for vr in vid_results]), inf_th=INF_THRESHOLD), 'N/A', 'N/A'),
                      ]
    else:
        normal_vids = [vr for vr in vid_results if vr['fps_consistent']]
        static_vids = [vr for vr in normal_vids if vr['near_static']]
        inconsis_vids = [vr for vr in vid_results if not vr['fps_consistent']]
        inconsis_n_static = [vr for vr in inconsis_vids if vr['near_static']]
        table_rows = [ ('Files', len(vid_results), len(normal_vids), len(inconsis_vids)),
                       ('Near static', len(static_vids), len(static_vids), len(inconsis_n_static)),
                       ('Encoded FPS', _frmt_mean_std(*mean_std([result['fps_encoded'] for result in vid_results])),
                       _frmt_mean_std(*mean_std([r['fps_encoded'] for r in normal_vids])),
                       _frmt_mean_std(*mean_std([r['fps_encoded'] for r in inconsis_vids]))),
                       (fps_label, _frmt_mean_std(*mean_std([r['fps_measured'] for r in vid_results])),
                       _frmt_mean_std(*mean_std([r['fps_measured'] for r in normal_vids])),
                       _frmt_mean_std(*mean_std([r['fps_measured'] for r in inconsis_vids]))),
                       ('Diff ratio', _frmt_mean_std(*mean_std([stats_ratio(r) for r in vid_results]), inf_th=INF_THRESHOLD),
                       _frmt_mean_std(*mean_std([stats_ratio(r) for r in normal_vids]),   inf_th=INF_THRESHOLD),
                       _frmt_mean_std(*mean_std([stats_ratio(r) for r in inconsis_vids]), inf_th=INF_THRESHOLD)),
                       ]
    print_table(table_rows, ('', 'Total', 'Consistent', 'Inconsistent'), align=['l', 'c', 'c', 'c'])
    for err in report['errors']:
        print(f"[WARN] {err['path']}: {err['error']}")

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


def save_fps_report(report, output_path):
    """Save one complete measured-FPS report as JSON."""
    output_path = Path(output_path)
    if output_path.is_dir():
        output_path /= DEFAULT_FPS_REPORT
    elif output_path.suffix == '':
        output_path = output_path.with_suffix('.json')
    output_path = get_unique_name(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

# endregion

#387(1,4,1) 400(1,2,)->379(1,1,1)
#505(1,2,5) #415(1,2,5)
#460(1,2,8)->#471(1,1,8) -> #427(1,2,2)->422(2,2,2)->407(1,1,1)
def test_printing(report_path, legacy_rep=None, r=100, **kwargs):
    report = load_fps_report(report_path)
    print(f"Print {r} rows:\n{72*'*'}  ")
    print_fps_report(report, rows=r, col_width='auto', **kwargs)
    print(f"\n\nPrint {r} rows, sorted by ratio:\n{72*'*'}")
    print_fps_report(report, rows=r, sort='ratio', **kwargs)

    if legacy_rep:
        report = load_fps_report(legacy_rep)
        print(f"\n\nPrint legacy {r} rows:\n{72*'*'}  ")
        print_fps_report(report, rows=r, sort='ratio', **kwargs)


if __name__ == '__main__': # pass

    test_report   = Path("/mnt/local-data/Python/Projects/weSmart/data/video/UBI_FIGHTS/videos/fps_msr_ubi.json")
    legacy_report = None # Path("/mnt/local-data/Projects/Wesmart/Video-datasets/UBI_FIGHTS/videos/eff_fps_ubi.json")
    # test_report   = Path("/mnt/local-data/Projects/Wesmart/Video-datasets/VioPeru/Copy (1) fps_msr_vp.json")
    # legacy_report = Path("/mnt/local-data/Projects/Wesmart/Video-datasets/VioPeru/Copy (1) eff_fps_vp_001.json")

    test_printing(test_report, legacy_report, r=50, order='dsc')
