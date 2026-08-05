""" Central command-line interface for weSmart project tools."""
import argparse
import csv
from pathlib import Path
#*project import
from json_stream_utils import DEFAULT_STREAM_META, save_pair_stream_json


def run_npz2stream(args):

    def _npz_json_pairs():
        if input_path.is_dir():
            jsons = {path.stem: path for path in sorted(input_path.glob('*.json'))}
            npzs  = {path.stem: path for path in sorted(input_path.glob('*.npz'))}
            return [(npzs[s], jsons[s], s) for s in sorted(set(jsons) & set(npzs))]

        if input_path.suffix.lower() == '.npz':
            npz_file  = input_path
            json_file = input_path.with_suffix('.json')
        elif input_path.suffix.lower() == '.json':
            json_file = input_path
            npz_file = input_path.with_suffix('.npz')
        else:
            raise ValueError(f"npz2stream input must be directory, .npz, or .json: {input_path}")
        if not npz_file.is_file() or not json_file.is_file():
            raise FileNotFoundError(f"Missing matched pair: {npz_file}, {json_file}")
        return [(npz_file, json_file, input_path.stem)]

    def _output_path_for_pair(stm, idx, count):
        default_suffix = '.json.zip' if not args.no_zip else '.json'

        def output_base(path):
            name = str(path)
            for ext in ('.json.zip',):
                if name.endswith(ext):
                    return Path(name[:-len(ext)])
            if path.suffix.lower() in {'.json', '.zip'}:
                return path.with_suffix('')
            return path

        def output_suffix(path):
            name = str(path)
            for ext in ('.json.zip',):
                if name.endswith(ext):
                    return ext
            if path.suffix.lower() in {'.zip', '.json'}:
                return path.suffix
            return default_suffix

        def output_file(path):
            return Path(f'{output_base(path)}{output_suffix(path)}')

        if args.output is None:
            return (input_path if input_path.is_dir() else input_path.parent)/f"{stm}{default_suffix}"

        out = Path(args.output)
        if str(out).endswith('.json.gz') or out.suffix.lower() == '.gz':
            raise ValueError("Saving gzip Stream JSON is disabled; use .json.zip or .json")
        is_dir_output = out.is_dir() or (not out.suffix and count == 1)
        if input_path.is_dir() and not out.suffix:
            is_dir_output = True

        if is_dir_output:
            return out/f"{stm}{default_suffix}"

        if count == 1:
            return output_file(out)
        #* count > 1
        suffix = output_suffix(out)
        out = output_base(out)
        return out.with_name(f"{out.name}_{idx:0{len(str(count))}d}{suffix}")

    input_path = Path(args.input_path)

    pairs = _npz_json_pairs()
    if not pairs:
        raise FileNotFoundError(f"No matched .json/.npz pairs found in {args.input_path}")
    for i, (npz_path, json_path, stem) in enumerate(pairs, start=1):
        out_path = _output_path_for_pair(stem, i, len(pairs))
        save_pair_stream_json(npz_path, json_path, out_path=out_path)
        print(f"Saved: {out_path}")

#* endregion

def main():
    """ Parse project CLI commands and dispatch the selected operation."""
    def parse_meta_info(value):
        return None if value.lower() == 'none' else Path(value)

    def is_tcn_format(path):
        required = {'json_name', 'window_index', 'window_start_frame', 'window_end_frame',
                    'window_start_time_sec', 'window_end_time_sec', 'target',
                    'prob_raw', 'pred_raw'}
        if not path.is_file() or path.suffix.lower() != '.csv':
            return False
        try:
            with path.open('r', encoding='utf-8-sig', newline='') as file:
                return required.issubset(csv.DictReader(file).fieldnames or [])
        except Exception:
            return False

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
    # metric.add_argument('-tr', '--total-row', action='store_true', help='Print an aggregate row in the per-stream table')
    metric.add_argument('--fp-unit', choices=('h', 'min'), default='h', help='False-positive rate time unit')
    metric.add_argument('--meta-info', nargs='?', type=parse_meta_info, const=None, default=DEFAULT_STREAM_META,
                                     help='Streams meta info path; use without a value to disable metadata columns')
    #* Conversions
    convert = commands.add_parser('convert', help='Convert project data formats')
    convert_commands = convert.add_subparsers(dest='convert_command', required=True)
    n2s = convert_commands.add_parser('npz2stream', aliases=['n2s'], help='Convert .npz/.json pairs to standard stream')
    n2s.add_argument('input_path', type=Path, help='Directory, .npz, or .json input')
    n2s.add_argument('-o', '--output', type=Path, default=None, help='Output file or directory')
    n2s.add_argument('-nz', '--no-zip', action='store_true', help='Save plain .json instead of default .json.zip')
    args = parser.parse_args()

    if args.command == 'convert':
        if args.convert_command in {'npz2stream', 'n2s'}:
            run_npz2stream(args)
        return

    from analysis_api import analyze_timelines
    from analysis_utils import convert_tcn_format, save_metric_report, print_metric_report, print_threshold_comparison

    stream_input = args.stream_path
    if is_tcn_format(stream_input):
        stream_input = convert_tcn_format(stream_input)
        if args.threshold is None and args.pred_cols is None:
            args.pred_cols = ['y_pred']

    selectors = args.threshold if args.threshold is not None else args.pred_cols
    table_mode = args.results_table
    if table_mode not in {None, 'auto', 'standard', 'thrs_cmp'}:
        metric.error("--results-table must be 'standard' or 'thrs_cmp'")
    if table_mode == 'auto' and selectors is not None:
        table_mode = 'thrs_cmp' if len(selectors) > 1 else 'standard'

    try:
        report = analyze_timelines(stream_input,
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
                    'total_row': True,
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
