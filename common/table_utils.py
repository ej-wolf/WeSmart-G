"""Plain-text tables with optional ANSI bold and italic styling."""

import math
import re, shutil, textwrap
from numbers import Integral, Real
from common.my_local_utils import cli_warning, collection


def print_table(tbl_data, headers=None, **kwargs):
    """ Print a plain-text table with optional ANSI bold and italic styling.
    * positional arguments:
      tbl_data          : iterable of rows; cells become strings, short rows are padded.
      headers=None      : use the first row as headers; '' hides headers; iterable supplies them.
    * keyword options:
      col_width='auto'  : width mode below, numeric width, or list of widths per column.
          auto          : fit headers and data without adding line breaks.
          data-opt      : fit data and wrap headers to match column widths.
          smart-opt     : try to fit terminal width, wrapping headers and optionally data.
          numeric       : Uniform column width (in characters)
          list          : list of numeric widths, each value corresponds to column .
      wrap=False        : False, no warping, always expand columns to fit content, headers or data (i.e. max(len(data)))
                          True: for numeric/list mode, allows warping to meet the requested widths
                                for 'smart-opt' allow data warping to fit the terminal
                          Other modes not effected by this argument
      lines=None        : separator every N data rows, or after listed 1-based row counts.
      align='c'         : 'l' (left), 'c' (center), 'r' (right), or a list per column.
      Bold/B='headers'  : Bold selection using the style formats below; B takes precedence.
      Italic/I=Non      : italic selection using the style formats below; 'I' takes precedence.
    * Styles options    :
          None/'all'/'headers' : no styling, entire table, or header cells only.
          [0, 2]        : select columns by zero-based index, including their headers.
          dict          : combine headers=True, cols=[...], rows=[...], cells=[(r, c)].
      Row and cell indices are zero-based and exclude the header row.
    Returns None. Invalid options warn and fall back to defaults; unknown options are ignored.
    """

    def option(name, default, validate):
        value = kwargs.pop(name, default)
        try:
            return validate(value)
        except (TypeError, ValueError, OverflowError):
            cli_warning(f'print_table: invalid {name}; using {default!r}')
            return validate(default)

    def boolean(value):
        if not isinstance(value, bool):
            raise ValueError
        return value

    def widths_option(value):
        if isinstance(value, str) and value in ('auto', 'data-opt', 'smart-opt'):
            return value
        values = value if isinstance(value, list) else [value] * count
        if len(values) != count:
            raise ValueError
        for width in values:
            if (isinstance(width, bool) or not isinstance(width, Real)
                    or not math.isfinite(width) or width < 0
                    or (wrap and width <= 0)):
                raise ValueError
        return [math.ceil(w) for w in values]

    def alignment(value):
        values = value if isinstance(value, list) else [value] * count
        if len(values) != count or any(v not in ('l', 'c', 'r') for v in values):
            raise ValueError
        return values

    def indices(values, limit):
        if not isinstance(values, list):
            raise ValueError
        if any(isinstance(v, bool) or not isinstance(v, Integral)
               or not 0 <= v < limit for v in values):
            raise ValueError
        return set(values)

    def style(value):
        if value is None:
            return {}
        if isinstance(value, str):
            if value not in ('all', 'headers'):
                raise ValueError
            return {value: True}
        style_spec = {'cols': value} if isinstance(value, list) else value
        if (not isinstance(style_spec, dict)
                or set(style_spec) - {'headers', 'cols', 'rows', 'cells'}):
            raise ValueError
        result = {'headers': boolean(style_spec.get('headers', False)),
                  'cols': indices(style_spec.get('cols', []), count),
                  'rows': indices(style_spec.get('rows', []), len(rows))}
        cells = style_spec.get('cells', [])
        if not isinstance(cells, list):
            raise ValueError
        result['cells'] = set()
        for cell in cells:
            if not isinstance(cell, (tuple, list)) or len(cell) != 2:
                raise ValueError
            indices([cell[0]], len(rows))
            indices([cell[1]], count)
            result['cells'].add(tuple(cell))
        return result

    def lines(value):
        if value is None:
            return set()
        if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
            return set(range(value, len(rows) + 1, value))
        if not isinstance(value, list) or any(
                isinstance(v, bool) or not isinstance(v, Integral)
                or not 1 <= v <= len(rows) for v in value):
            raise ValueError
        return set(value)

    def length(cell):
        return max(map(len, cell.split('\n')), default=0)

    def header_tokens(cell):
        tokens = []
        for part in cell.split('\n'):
            for word in part.split():
                tokens.extend(re.findall( r'[A-Za-z0-9]+(?:[-_.]+(?=[A-Za-z0-9]|$))?|[^A-Za-z0-9\s]+', word))
        return tokens

    def wrap_header(cell, width):
        wrapped = []
        for part in cell.split('\n'):
            current = ''
            for word in part.split():
                pieces = [word] if len(word) <= width else header_tokens(word)
                if not pieces:
                    pieces = [word]
                for piece_index, piece in enumerate(pieces):
                    if len(piece) > width:
                        if current:
                            wrapped.append(current)
                            current = ''
                        wrapped.extend(textwrap.wrap(piece, width, break_long_words=True, break_on_hyphens=False) or [''])
                        continue
                    spacer = ' ' if piece_index == 0 and current else ''
                    if current and len(current) + len(spacer) + len(piece) > width:
                        wrapped.append(current)
                        current, spacer = '', ''
                    current += spacer + piece
            if current:
                wrapped.append(current)
        return wrapped or ['']

    def wrap_cell(cell, width):
        wrapped = []
        for part in cell.split('\n'):
            wrapped.extend(textwrap.wrap(part, width, expand_tabs=False,
                                          break_long_words=True, break_on_hyphens=False) or [''])
        return wrapped

    def header_word_width(cell):
        tokens = header_tokens(cell)
        return max((len(token) for token in tokens), default=length(cell))

    def smart_layout_cost(width, data_wrap):
        header_cost = 0
        for cell, column_width in zip(heading or [], width):
            header_cost += len(wrap_header(cell, column_width))
            header_cost += 100 * sum(max(0, len(token) - column_width)
                                     for token in header_tokens(cell))
        data_cost = (sum(max((len(wrap_cell(r[c], width[c]))
                              for c in range(count)), default=1) for r in rows)
                     if data_wrap else len(rows))
        return header_cost + data_cost

    def smart_widths(data_sizes, header_sizes, data_wrap):
        def table_width(w):
            return sum(col_w + 2  for col_w in w) + 3*max(0, count - 1)

        preferred = [max(3, d_sz, h_sz)  for d_sz, h_sz in zip(data_sizes, header_sizes)]
        minimum = [3 if data_wrap else max(3, d_sz)  for d_sz in data_sizes]
        terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns

        if table_width(preferred) <= terminal_width:
            return preferred

        width = preferred[:]
        while (table_width(width) > terminal_width
               and any(column_width > lower for column_width, lower in zip(width, minimum))):
            current_cost = smart_layout_cost(width, data_wrap)
            candidates = []
            for col, column_width in enumerate(width):
                if column_width <= minimum[col]:
                    continue
                trial = width[:]
                trial[col] -= 1
                candidates.append((smart_layout_cost(trial, data_wrap) - current_cost,
                                   -column_width, col))
            _, _, col = min(candidates)
            width[col] -= 1

        if table_width(width) > terminal_width:
            cli_warning('print_table: smart-opt table exceeds terminal width')
        return width

    def selected(slct, r, c):
        return (slct.get('all', False)
                or (r is None and slct.get('headers', False))
                or c in slct.get('cols', ())
                or r in slct.get('rows', ())
                or (r, c) in slct.get('cells', ()))

    def print_separator():
        print(f'{all_codes}{separator}' + ('\033[0m' if all_codes else ''))

    def print_row(cells, row_index):
        parts = []
        for cell, width in zip(cells, column_widths):
            cell_lines = []
            for part in cell.split('\n'):
                if row_index is None and width_spec in ('data-opt', 'smart-opt'):
                    cell_lines.extend(wrap_header(part, width))
                elif wrap and (width_spec == 'smart-opt' or isinstance(width_spec, list)):
                    cell_lines.extend(wrap_cell(part, width))
                else:
                    cell_lines.append(part)
            parts.append(cell_lines)
        for n in range(max(map(len, parts))):
            rendered = []
            for col, (cell_lines, width, alignment_code) in enumerate(zip(parts, column_widths, align)):
                text = cell_lines[n] if n < len(cell_lines) else ''
                text = format(text, f'{dict(l="<", c="^", r=">")[alignment_code]}{width}')
                text = f' {text} '
                codes = ('\033[1m' if selected(bold, row_index, col) else '')
                codes += '\033[3m' if selected(italic, row_index, col) else ''
                rendered.append(codes + text + ('\033[0m' if codes else ''))
            print(' | '.join(rendered))
    try:
        if isinstance(tbl_data, (str, bytes)):
            raise TypeError
        rows = []
        for row in collection(tbl_data):
            if isinstance(row, (str, bytes)):
                raise TypeError
            rows.append([str(cell) for cell in row])
    except (TypeError, ValueError):
        cli_warning('print_table: tbl_data must contain iterable rows')
        return

    count = max((len(row) for row in rows), default=0)
    if any(len(row) != count for row in rows):
        cli_warning('print_table: ragged rows padded with empty cells')
        rows = [r + [''] * (count - len(r)) for r in rows]

    hidden = isinstance(headers, str) and headers == ''
    heading = None
    if headers is not None and not hidden:
        try:
            if isinstance(headers, (str, bytes)):
                raise ValueError
            heading = [str(cell) for cell in collection(headers)]
            if rows and len(heading) != count:
                raise ValueError
            count = len(heading)
        except (TypeError, ValueError):
            cli_warning('print_table: invalid headers; using the first r')
            headers, heading = None, None
    if headers is None and rows:
        heading = rows.pop(0)

    wrap = option('wrap', False, boolean)

    width_spec = option('col_width', 'auto', widths_option)

    align = option('align', 'c', alignment)

    for alias, option_name in (('B', 'Bold'), ('I', 'Italic')):
        if alias in kwargs:
            kwargs[option_name] = kwargs.pop(alias)
    bold   = option('Bold', 'headers', style)
    italic = option('Italic', None, style)

    separators = option('lines', None, lines)
    for name in kwargs:
        cli_warning(f'print_table: unknown option {name!r}; ignored')
    if not count or (not rows and heading is None):
        return

    data_widths = [max((length(row[c]) for row in rows), default=0)
                   for c in range(count)]
    header_widths = [length(cell) for cell in heading] if heading is not None else [0] * count
    fitted = [max(1, d, h) for d, h in zip(data_widths, header_widths)]
    if width_spec == 'auto':
        column_widths = fitted
    elif width_spec == 'data-opt':
        column_widths = [max(1, width) for width in (data_widths if rows else header_widths)]
    elif width_spec == 'smart-opt':
        column_widths = smart_widths(data_widths, [header_word_width(cell) for cell in heading]
                                                    if heading is not None else [0]*count, wrap)
    elif wrap:
        column_widths = width_spec
    else:
        column_widths = [max(width, fit) for width, fit in zip(width_spec, fitted)]

    separator = '-+-'.join('-' * (width + 2) for width in column_widths)
    all_codes = ''.join(code for selection, code in ((bold, '\033[1m'), (italic, '\033[3m'))
                        if selection.get('all'))

    if heading is not None:
        print_row(heading, None)
        print_separator()
    for index, row in enumerate(rows):
        print_row(row, index)
        if index + 1 in separators:
            print_separator()

#208(,2,)-< 306(,10,) -> 296(,,1)
