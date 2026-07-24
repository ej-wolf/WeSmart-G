"""     Central command-line interface for weSmart project tools.
    Parser layout:
    - metric: old-style shortcut for timeline metric evaluation.
    - analyze: grouped namespace for raw-result analysis, timeline analysis,
      report printing, and timeline plotting.
    - raw: analyze subcommand that starts from raw prediction NPZ files.
    - timelines: analyze subcommand that starts from existing timeline CSVs.
    - pred_grp/prd_g: mutually exclusive prediction selectors; use either binary prediction
     columns or y_prob thresholds for one metric run.
"""
import argparse
from pathlib import Path

from analysis_api import analyze_raw_results, analyze_timelines, plot_timeline, print_results
from json_stream_utils import DEFAULT_STREAM_META


def main():

    def parse_meta_info(value):
        return None if value.lower() == 'none' else Path(value)

    def add_metric_args(prs):
        """Add shared metric options to raw/timelines subparsers."""
        prd_g = prs.add_mutually_exclusive_group()
        prd_g.add_argument('-p' , '--pred-cols',  nargs='+', help='Binary prediction columns in timelines')
        prd_g.add_argument('-th', '--thresholds', dest='threshold', nargs='+', type=float, help='Thresholds applied to y_prob')

        prs.add_argument('-c', '--config', type=Path, default=None, help='Metric YAML configuration file')
        prs.add_argument('-o', '--output', type=Path, help='CSV/JSON metric output')
        prs.add_argument('-tbl', '--results-table', nargs='?', choices=('standard', 'thrs_cmp'), const='auto',
                                                    help='Print results using the default or selected table')
        prs.add_argument('-tr', '--total-row', action='store_true', help='Include the aggregate row')
        prs.add_argument('--fp-unit', choices=('h', 'min'), default='h', help='False-positive rate time unit')
        prs.add_argument('--meta-info', nargs='?', type=parse_meta_info, const=None, default=DEFAULT_STREAM_META,
                                        help="Stream metadata path; use 'none' to disable")

    parser = argparse.ArgumentParser(description='weSmart project commands')
    commands = parser.add_subparsers(dest='command', required=True)

    metric = commands.add_parser( 'metric', help='Evaluate timeline with stream metrics',
                                                  description='Evaluate timeline CSV with stream metrics')
    metric.add_argument('stream_path', type=Path, help='Timeline CSV, directory, or list is not supported by CLI')
    metric.add_argument('-o', '--output', type=Path, help='CSV/JSON output file; default: stream_metric.csv')
    metric.add_argument('-c', '--config', type=Path, default=None, help='Metric YAML configuration file')
    metric.add_argument('-tbl', '--results-table', nargs='?', metavar='{standard, thrs_cmp}', const='auto', default=None, help='Print/ Select mode or results table')
    metric.add_argument('-tr', '--total-row', action='store_true', help='Print an aggregate row in the per-stream table')
    metric.add_argument('--fp-unit', choices=('h', 'min'), default='h', help='False-positive rate time unit')
    metric.add_argument('--meta-info', nargs='?', type=parse_meta_info,  const=None, default=DEFAULT_STREAM_META, help='Path to Streams meta info; use without a value to disable metadata columns')

    pred_grp = metric.add_mutually_exclusive_group()
    pred_grp.add_argument('-p', '--pred-cols', nargs='+', help='Binary prediction columns in timeline')
    pred_grp.add_argument('-th', '--threshold', nargs='+', type=float, help='One or more thresholds applied to y_prob col')

    analyze = commands.add_parser('analyze', help='Analyze or present model results')
    actions = analyze.add_subparsers(dest='action', required=True)

    raw = actions.add_parser('raw', help='Analyze raw prediction NPZ results')
    raw.add_argument('test_results', type=Path)
    raw.add_argument('-m','--mode', choices=('clip', 'video', 'stream'), default='stream')
    raw.add_argument('-w','--win-metrics', action='store_true', help='Calculate win confusion and ROC diagnostics')
    raw.add_argument('-s', '--save-timelines', type=Path, help='Save generated timeline CSV files to this directory')
    add_metric_args(raw)

    timelines = actions.add_parser('timelines', help='Analyze timeline CSV files')
    timelines.add_argument('timeline_input', type=Path)
    add_metric_args(timelines)

    show = actions.add_parser('print', help='Print an existing CSV/JSON report')
    show.add_argument('report', type=Path)
    show.add_argument('-tbl', '--results-table', nargs='?', choices=('standard', 'thrs_cmp'), const='auto')
    show.add_argument('-tr', '--total-row', action='store_true')
    show.add_argument('--fp-unit', choices=('h', 'min'), default='h')

    plot = actions.add_parser('plot', help='Plot an existing timeline CSV')
    plot.add_argument('timeline', type=Path)
    plot.add_argument('-p', '--pred-col', default='y_pred')
    plot.add_argument('-th', '--threshold', type=float, default=0.5)
    plot.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()

    selectors = args.threshold if getattr(args, 'threshold', None) is not None \
                               else getattr(args, 'pred_cols', None)

    table_mode = getattr(args, 'results_table', None)
    if table_mode == 'auto':
        table_mode = 'thrs_cmp' if selectors is not None and len(selectors) > 1 \
                                else 'standard'

    print_kwargs = {'results_table': table_mode or False,
                    'total_row': getattr(args, 'total_row', False),
                    'fp_unit'  : getattr(args, 'fp_unit', 'h'),
                    'meta_info': getattr(args, 'meta_info', None)}

    if  args.command == 'metric':
        analyze_timelines( args.stream_path, thresholds=args.threshold, pred_cols=args.pred_cols,
                           config_path=args.config, meta_info=args.meta_info, output_path=args.output, print_results=True,
                           print_kwargs=print_kwargs)

    elif args.action == 'print':
        print_results(args.report, results_table=args.results_table or False,
                      total_row=args.total_row, fp_unit=args.fp_unit)

    elif args.action == 'plot':
        plot_timeline(args.timeline, pred_column=args.pred_col, threshold=args.threshold,
                      save_to=args.output, show=args.output is None)

    elif args.action in ('raw', 'timelines'):
        metric_args = {'thresholds' : args.threshold, 'pred_cols': args.pred_cols,
                       'config_path': args.config, 'meta_info': args.meta_info,
                       'output_path': args.output, 'print_results': True,
                       'print_kwargs': print_kwargs}
        if args.action == 'raw':
            analyze_raw_results(args.test_results, mode=args.mode, window_metrics=args.window_metrics,
                                timeline_dir=args.save_timelines, **metric_args)
        else: #* args.action == 'timelines'
            analyze_timelines(args.timeline_input, **metric_args)
    else:
        parser.error(f"Unsupported analyze action: {args.action}")
    return

#140(,1,)->116->122:113()

if __name__ == '__main__':
    main()
