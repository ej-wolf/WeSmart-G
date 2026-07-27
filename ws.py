"""Central command-line interface for weSmart project tools."""
import argparse
from pathlib import Path

from analysis_api import analyze_timelines
from analysis_utils import save_metric_report, print_metric_report, print_threshold_comparison
from json_stream_utils import DEFAULT_STREAM_META


def main():
    """ Parse project CLI commands and dispatch the selected operation."""
    def parse_meta_info(value):
        return None if value.lower() == 'none' else Path(value)

    parser = argparse.ArgumentParser(description='weSmart project commands')
    commands = parser.add_subparsers(dest='command', required=True)

    metric = commands.add_parser('metric', help='Evaluate timeline CSV with stream metrics', description='Evaluate timeline CSV with stream metrics')
    metric.add_argument('stream_path', type=Path, help='Timeline CSV, directory, or list is not supported by CLI')

    prediction = metric.add_mutually_exclusive_group()
    prediction.add_argument('-p', '--pred-cols', nargs='+', help='Binary prediction columns in timeline')
    prediction.add_argument('-th', '--threshold', nargs='+', type=float, help='One or more thresholds applied to y_prob col')

    metric.add_argument('-o', '--output', type=Path, help='CSV/JSON output file; default: stream_metric.csv')
    metric.add_argument('-c', '--config', type=Path, default=None, help='Metric YAML configuration file')
    metric.add_argument('-tbl', '--results-table', nargs='?', metavar='{standard, thrs_cmp}',
                                     const='auto', default=None, help='Print/ Select mode or results table')
    metric.add_argument('-tr', '--total-row', action='store_true', help='Print an aggregate row in the per-stream table')
    metric.add_argument('--fp-unit', choices=('h', 'min'), default='h', help='False-positive rate time unit')
    metric.add_argument('--meta-info', nargs='?', type=parse_meta_info, const=None, default=DEFAULT_STREAM_META,
                                     help='Streams meta info path; use without a value to disable metadata columns')
    args = parser.parse_args()

    selectors = args.threshold if args.threshold is not None else args.pred_cols
    table_mode = args.results_table
    if table_mode not in {None, 'auto', 'standard', 'thrs_cmp'}:
        metric.error("--results-table must be 'standard' or 'thrs_cmp'")
    if table_mode == 'auto' and selectors is not None:
        table_mode = 'thrs_cmp' if len(selectors) > 1 else 'standard'

    try:
        report = analyze_timelines(args.stream_path,
                                   thresholds=args.threshold,
                                   pred_cols=args.pred_cols,
                                   config_path=args.config,
                                   meta_info=args.meta_info)
    except ValueError as error:
        if selectors is None:
            metric.error(str(error))
        raise

    if table_mode == 'auto':
        table_mode = 'thrs_cmp' if isinstance(report, list) and len(report) > 1 else 'standard'

    print_kwargs = {'results_table': table_mode,
                    'total_row': args.total_row,
                    'fp_unit': args.fp_unit,
                    'meta_info': args.meta_info}
    if isinstance(report, list):
        print_threshold_comparison(report, **print_kwargs)
    else:
        print_metric_report(report, **print_kwargs)
    if args.output is not None:
        print(f"Saved: {save_metric_report(report, args.output)}")

if __name__ == '__main__':
    main()
#85()
