""" Annotation format loaders and converters.
    This module normalizes temporal event annotations into one event-list dict.
    Supported now:
        - default event annotations: start_time, end_time, flag
        - old HMC/Nana event annotations: metadata + event-name sections
        - UBI frame-label CSV annotations: one binary label per video frame
    Dataset annotation support is intentionally left as a TODO because it should
    replace `ann_file_utils.py` later.
"""

from __future__ import annotations
from pathlib import Path
import subprocess
from typing import Any, Iterable


FMT_DS = 'ds'
FMT_EVENT = 'event'
FMT_OLD_EVENT = 'old_event'
FMT_UBI = 'ubi'
FMT_UNKNOWN = 'unknown'

TIMELINE_BEGIN = 'TL_BEGIN'
TIMELINE_END = 'TL_END'
SPECIAL_EVENT_TAGS = {'s', 'x'}

OLD_EVENT_CODE_MAP = {'FALL': 2, 'TENSION': 3, 'FIGHT': 4,}
OLD_EVENT_NAME_MAP = {value: key for key, value in OLD_EVENT_CODE_MAP.items()}


# region Shared helpers

def parse_time_str(text:str|int|float|None) -> float|None:
    """Convert HH:MM:SS / MM:SS / SS into seconds, preserving empty values."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    text = str(text).strip().rstrip(',')
    if text == '':
        return None

    parts = [part.strip() for part in text.split(':')]
    if any(part == '' for part in parts):
        return None
    values = [float(part) for part in parts]

    if len(values) == 3:
        hrs, mins, secs = values
    elif len(values) == 2:
        hrs = 0.0
        mins, secs = values
    else:
        hrs, mins, secs = 0.0, 0.0, values[0]
    return hrs * 3600.0 + mins * 60.0 + secs


def parse_event_time_str(text:str|int|float|None) -> float|str|None:
    """Parse a default event time, preserving timeline boundary markers."""
    if _is_timeline_begin(text):
        return TIMELINE_BEGIN
    if _is_timeline_end(text):
        return TIMELINE_END
    return parse_time_str(text)


def resolve_event_time(value: str | int | float | None, *, timeline_end: float | None = None) -> float | None:
    """Resolve event time markers to concrete seconds when timeline bounds are known."""
    if _is_timeline_begin(value):
        return 0.0
    if _is_timeline_end(value):
        return None if timeline_end is None else float(timeline_end)
    return parse_time_str(value)


def format_time_str(seconds:float|int|None) -> str:
    """ Format seconds as HH:MM:SS for default event annotation files."""
    if seconds is None:
        return ''
    seconds = float(seconds)
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    rem = seconds - hrs * 3600
    mins = int(rem // 60)
    secs = rem - mins * 60
    if secs.is_integer():
        sec_txt = f'{int(secs):02d}'
    else:
        sec_txt = f'{secs:06.3f}'.rstrip('0').rstrip('.')
    return f'{hrs:02d}:{mins:02d}:{sec_txt}'


def format_event_time_str(value: float | int | str | None, *, default_marker: str | None = None) -> str:
    """Format default event times, including timeline boundary markers."""
    if _is_timeline_begin(value):
        return TIMELINE_BEGIN
    if _is_timeline_end(value):
        return TIMELINE_END
    if value is None and default_marker is not None:
        return default_marker
    return format_time_str(value)


def resolve_format(path: str | Path) -> str:
    """Infer annotation format from file content; extensions are ignored."""
    ann_path = Path(path)
    lines = _meaningful_lines(ann_path)
    if not lines:
        return FMT_EVENT

    if _looks_like_old_event(lines):
        return FMT_OLD_EVENT
    if _looks_like_default_event(lines):
        return FMT_EVENT
    if _looks_like_ubi_ann(lines):
        return FMT_UBI
    if _looks_like_ds_ann(lines):
        return FMT_DS
    return FMT_UNKNOWN


def resolve_formats(path: str | Path) -> str:
    """Compatibility alias for `resolve_format`."""
    return resolve_format(path)


def convert_to_event_ann(src: dict[str, Any] | str | Path, out_path: str | Path, **kwargs) -> Path:
    """Convert supported event annotations into the default event annotation file."""
    if isinstance(src, dict):
        data = src
    else:
        fmt = resolve_format(src)
        if fmt == FMT_EVENT:
            data = load_event_ann(src)
        elif fmt == FMT_OLD_EVENT:
            data = load_old_event_ann(src)
        elif fmt == FMT_UBI:
            data = load_ubi_ann(src, video_path=kwargs.get('video_path'), fps=kwargs.get('fps'))
        else:
            raise ValueError(f'Cannot convert unsupported annotation format: {fmt}')
    return save_event_ann(data, out_path)


def _ann_dict(source_format:str, source_path:Path, events:Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {'format': FMT_EVENT,
            'source_format': source_format,
            'source_path': str(source_path),
            'video': None,
            'fps': None,
            'step': None,
            'header': [],
            'events': list(events),
            }


def _event_dict(start:float|str|None, end:float|str|None, flag: int | str, *,
                label:str|None= None, seq:int=0)-> dict[str, Any]:
    return {'start': start, 'end': end, 'flag': _parse_event_flag(flag), 'label': label, 'seq': int(seq)}


def _events_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    events = data.get('events', [])
    if not isinstance(events, list):
        raise TypeError("annotation data must contain an 'events' list")
    return events


def _parse_event_flag(flag: Any) -> int | str:
    text = str(flag).strip().rstrip(',')
    if text.lower() in SPECIAL_EVENT_TAGS:
        return text
    return int(text)


def _format_event_flag(flag: Any) -> str:
    parsed = _parse_event_flag(flag)
    return str(parsed)


def _is_timeline_begin(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == TIMELINE_BEGIN


def _is_timeline_end(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == TIMELINE_END


def _meaningful_lines(path: Path, limit: int = 80) -> list[str]:
    lines: list[str] = []
    with path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
            if len(lines) >= limit:
                break
    return lines


def _leading_comment_header(path: Path) -> list[str]:
    header: list[str] = []
    with path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith('#'):
                break
            header.append(_comment_body(line))
    return header


# endregion


# region Default event annotations

def load_event_ann(path: str | Path) -> dict[str, Any]:
    """Load default event annotations into the normalized event dict."""
    ann_path = Path(path)
    header: list[str] = []
    events: list[dict[str, Any]] = []
    seen_data = False

    with ann_path.open('r', encoding='utf-8') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if not seen_data:
                    header.append(_comment_body(line))
                continue
            seen_data = True

            parts = _split_event_row(line)
            if len(parts) != 3:
                raise ValueError(f'Invalid event annotation row at {ann_path}:{line_no}: {raw_line.rstrip()}')

            start_txt, end_txt, flag_txt = parts
            flag = _parse_event_flag(flag_txt)
            events.append(_event_dict(parse_event_time_str(start_txt), parse_event_time_str(end_txt), flag, seq=len(events)))

    data = _ann_dict(FMT_EVENT, ann_path, _sort_events_by_time(events))
    data['header'] = header
    _apply_header_meta(data, header)
    return data


def save_event_ann(data: dict[str, Any], path: str | Path) -> Path:
    """ Save normalized event data in the default event annotation format."""
    ann_path = Path(path)
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    events = _sort_events_by_time(_events_from_data(data))
    with ann_path.open('w', encoding='utf-8') as handle:
        header_lines = _event_header_lines(data)
        for line in header_lines:
            handle.write(f'# {line}\n')
        if header_lines:
            handle.write('\n')
        for event in events:
            start_txt = format_event_time_str(event.get('start'), default_marker=TIMELINE_BEGIN)
            end_txt = format_event_time_str(event.get('end'), default_marker=TIMELINE_END)
            handle.write(f"{start_txt},\t{end_txt},\t{_format_event_flag(event['flag'])}\n")
    return ann_path


def _event_header_lines(data: dict[str, Any]) -> list[str]:
    lines = []
    source_path = data.get('source_path')
    if source_path:
        lines.append(f"source: {Path(source_path).name}")
    for key in ('video', 'fps', 'step'):
        value = data.get(key)
        if value is not None:
            lines.append(f'{key}: {value}')

    known_keys = {'source', 'video', 'fps', 'step'}
    for line in data.get('header', []):
        key = line.split(':', 1)[0].strip().lower() if ':' in line else ''
        if key not in known_keys:
            lines.append(str(line))
    return lines


def _comment_body(line: str) -> str:
    return line[1:].strip() if line.startswith('#') else line.strip()


def _apply_header_meta(data: dict[str, Any], header: Iterable[str]) -> None:
    for line in header:
        if ':' not in line:
            continue
        key, value = [part.strip() for part in line.split(':', 1)]
        key = key.lower()
        if key == 'video':
            data['video'] = value
        elif key == 'fps':
            data['fps'] = float(value)
        elif key == 'step':
            data['step'] = int(float(value))


# endregion


# region Old HMC/Nana event annotations

def load_old_event_ann(path: str | Path) -> dict[str, Any]:
    """Load old HMC/Nana block annotations into the normalized event dict."""
    ann_path = Path(path)
    meta: dict[str, Any] = {'video': None, 'fps': None, 'step': None}
    section = None
    header: list[str] = []
    events: list[dict[str, Any]] = []

    with ann_path.open('r', encoding='utf-8') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                if line.startswith('#') and section is None:
                    header.append(_comment_body(line))
                continue

            if ':' in line and section is None:
                key, value = [part.strip() for part in line.split(':', 1)]
                if not _set_old_meta(meta, key, value):
                    header.append(line)
                continue

            if line.isupper() and ':' not in line:
                section = line
                continue

            if line.startswith('sec:'):
                if section is None:
                    raise ValueError(f'Old event annotation has sec row before section at {ann_path}:{line_no}')
                intervals_txt = line.split(':', 1)[1].strip()
                for start_sec, end_sec in _parse_interval_list(intervals_txt):
                    flag = OLD_EVENT_CODE_MAP.get(section, 5)
                    events.append(_event_dict(start_sec, end_sec, flag, label=section, seq=len(events)))

    data = _ann_dict(FMT_OLD_EVENT, ann_path, events)
    data.update(meta)
    data['header'] = header
    return data


def save_old_event_ann(data: dict[str, Any], path: str | Path) -> Path:
    """Save normalized event data using the old HMC/Nana block layout."""
    ann_path = Path(path)
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    events = _events_from_data(data)
    grouped: dict[str, list[dict[str, Any]]] = {}

    for event in events:
        label = str(event.get('label') or OLD_EVENT_NAME_MAP.get(int(event['flag']), 'OTHER'))
        grouped.setdefault(label, []).append(event)

    with ann_path.open('w', encoding='utf-8') as handle:
        if data.get('video') is not None:
            handle.write(f"video: {data['video']}\n")
        if data.get('fps') is not None:
            handle.write(f"fps: {data['fps']}\n")
        if data.get('step') is not None:
            handle.write(f"step: {data['step']}\n")
        handle.write('\n')

        for label in _old_section_order(grouped):
            intervals = grouped.get(label, [])
            sec_text = _format_old_intervals(intervals)
            handle.write(f'{label}\n')
            handle.write(f'  raw: {sec_text}\n')
            handle.write(f'  sec: {sec_text}\n')
    return ann_path


def _set_old_meta(meta: dict[str, Any], key: str, value: str) -> bool:
    key = key.lower()
    if key == 'video':
        meta['video'] = value
        return True
    elif key == 'fps':
        meta['fps'] = float(value)
        return True
    elif key == 'step':
        meta['step'] = int(float(value))
        return True
    return False


def _parse_interval_list(text: str) -> list[tuple[float | None, float | None]]:
    if text.strip() == '-':
        return []

    intervals: list[tuple[float | None, float | None]] = []
    for item in text.split(','):
        item = item.strip()
        if not item or item == '-':
            continue
        intervals.append(_parse_interval(item))
    return intervals


def _parse_interval(text: str) -> tuple[float | None, float | None]:
    if '-' not in text:
        return parse_time_str(text), None

    start_txt, end_txt = text.split('-', 1)
    start = parse_time_str(start_txt.strip())
    end = parse_time_str(end_txt.strip())
    return start, end


def _old_section_order(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    ordered = [name for name in ('TENSION', 'FIGHT', 'FALL') if name in grouped]
    ordered.extend(sorted(name for name in grouped if name not in set(ordered)))
    return ordered


def _format_old_intervals(events: Iterable[dict[str, Any]]) -> str:
    parts = []
    for event in events:
        start = _fmt_old_time(event.get('start'))
        end = _fmt_old_time(event.get('end'))
        parts.append(f'{start}-{end}')
    return ', '.join(parts) if parts else '-'


def _fmt_old_time(seconds: float | int | None) -> str:
    if seconds is None:
        return ''
    seconds = float(seconds)
    return str(int(seconds)) if seconds.is_integer() else f'{seconds:.3f}'.rstrip('0').rstrip('.')


# endregion


# region UBI frame-label annotations


def load_ubi_ann(path: str | Path, video_path: str | Path | None = None,
                 fps: float | None = None) -> dict[str, Any]:
    """Load UBI frame-label CSV as fight events in the normalized event dict."""
    ann_path = Path(path)
    video_path = Path(video_path) if video_path is not None else _infer_ubi_video_path(ann_path)
    fps = float(fps or _probe_video_fps(video_path))
    if fps <= 0:
        raise ValueError(f'Invalid FPS for UBI annotation conversion: {fps}')

    header = _leading_comment_header(ann_path)
    labels = _load_ubi_labels(ann_path)
    events: list[dict[str, Any]] = []
    for start_idx, end_idx, label in _label_runs(labels):
        if label != 1:
            continue
        start_sec = start_idx / fps
        end_sec = (end_idx + 1) / fps
        events.append(_event_dict(start_sec, end_sec, 4, label='FIGHT', seq=len(events)))

    data = _ann_dict(FMT_UBI, ann_path, events)
    data['video'] = video_path.name
    data['fps'] = fps
    data['frame_count'] = len(labels)
    data['header'] = header
    return data


def save_ubi_ann(data: dict[str, Any], path: str | Path) -> Path:
    """Save normalized events as UBI one-label-per-frame CSV."""
    ann_path = Path(path)
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    fps = data.get('fps')
    frame_count = data.get('frame_count')
    if fps is None or frame_count is None:
        raise ValueError("UBI save requires 'fps' and 'frame_count' in annotation data")

    fps = float(fps)
    frame_count = int(frame_count)
    labels = [0] * frame_count
    for event in _events_from_data(data):
        if int(event.get('flag', 0)) == 0:
            continue
        start_idx = max(0, int(float(event.get('start') or 0.0) * fps))
        end_val = event.get('end')
        end_idx = frame_count if end_val is None else min(frame_count, int(float(end_val) * fps))
        for idx in range(start_idx, end_idx):
            labels[idx] = 1

    with ann_path.open('w', encoding='utf-8') as handle:
        for label in labels:
            handle.write(f'{label}\n')
    return ann_path


def _load_ubi_labels(path: Path) -> list[int]:
    labels: list[int] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                label = int(line)
            except ValueError as exc:
                raise ValueError(f'Invalid UBI label at {path}:{line_no}: {raw_line.rstrip()}') from exc
            if label not in {0, 1}:
                raise ValueError(f'Unsupported UBI label at {path}:{line_no}: {label}')
            labels.append(label)
    return labels


def _label_runs(labels: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if not labels:
        return runs

    start = 0
    current = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label == current:
            continue
        runs.append((start, idx - 1, current))
        start = idx
        current = label
    runs.append((start, len(labels) - 1, current))
    return runs


def _infer_ubi_video_path(ann_path: Path) -> Path:
    stem = ann_path.stem
    root = ann_path.parent.parent
    candidates = [
        root / 'videos' / 'fight' / f'{stem}.mp4',
        root / 'videos' / 'normal' / f'{stem}.mp4',
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Could not infer UBI video path for {ann_path}')


def _probe_video_fps(video_path: Path) -> float:
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,r_frame_rate',
        '-of', 'default=nw=1:nk=1',
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'Could not read video FPS for {video_path}') from exc

    for value in out:
        fps = _parse_fps_value(value)
        if fps > 0:
            return fps
    raise ValueError(f'Could not parse video FPS for {video_path}')


def _parse_fps_value(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    if '/' in value:
        num_txt, den_txt = value.split('/', 1)
        num = float(num_txt)
        den = float(den_txt)
        return num / den if den else 0.0
    return float(value)


# endregion


# region Dataset annotations

# TODO: replace ann_file_utils.py with dataset annotation loaders/savers here.
# Dataset annotation format is currently: <relative/video/path> <int_label>.

# endregion


# region Format detection helpers


def _sort_events_by_time(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort default event output by start time, then end time, then source order."""
    return sorted(events, key=lambda event: (
        _sort_time_value(event.get('start'), default=0.0),
        _sort_time_value(event.get('end'), default=float('inf')),
        int(event.get('seq', 0)),
        _sort_flag_value(event.get('flag', 0)),
    ))


def _sort_flag_value(flag: Any) -> tuple[int, str]:
    try:
        return 0, f'{int(flag):04d}'
    except (TypeError, ValueError):
        return 1, str(flag)


def _sort_time_value(value: Any, *, default: float) -> float:
    if _is_timeline_begin(value):
        return 0.0
    if _is_timeline_end(value):
        return float('inf')
    if value is None:
        return default
    return float(value)


def _looks_like_old_event(lines: list[str]) -> bool:
    has_section = any(line.isupper() and ':' not in line for line in lines)
    has_sec = any(line.startswith('sec:') for line in lines)
    return has_section and has_sec


def _looks_like_default_event(lines: list[str]) -> bool:
    first = lines[0]
    parts = _split_event_row(first)
    if len(parts) != 3:
        return False
    try:
        parse_event_time_str(parts[0])
        parse_event_time_str(parts[1])
        _parse_event_flag(parts[2])
    except ValueError:
        return False
    return True


def _looks_like_ubi_ann(lines: list[str]) -> bool:
    try:
        return all(int(line.strip()) in {0, 1} for line in lines)
    except ValueError:
        return False


def _looks_like_ds_ann(lines: list[str]) -> bool:
    try:
        for line in lines[:10]:
            path_str, label_str = line.rsplit(' ', 1)
            if not path_str:
                return False
            int(label_str)
    except ValueError:
        return False
    return True


def _split_event_row(line: str) -> list[str]:
    parts = [part.strip() for part in line.split(',\t')]
    if len(parts) == 3:
        return parts
    parts = [part.strip() for part in line.split('\t')]
    if len(parts) == 3:
        return parts
    return [part.strip() for part in line.split(',')]


# endregion
