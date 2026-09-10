""" Play videos with synced event-annotation overlays.
    Usage:
        python visual_analyzer.py <video.mp4> [annotation.ann|annotation.txt|annotation.csv]
        python visual_analyzer.py <video.mp4> <annotation> --start 12.5 --hold

    The player supports default event annotations files through annotations.py. It draws
    active event flags below the video, supports timeline mouse seeking, and exposes keyboard seek.
"""

from __future__ import annotations
import argparse
import re, shutil, subprocess
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from common.my_local_utils import as_collection, cli_warning, print_color
from annotations import (load_event_ann, load_ubi_ann, resolve_event_time, resolve_format,
                         FMT_EVENT, FMT_UBI, SPECIAL_EVENT_TAGS)

SEEK_JUMPS = {'COARSE'  : 5.0,  'MEDIUM'  : 1.0, 'FINE'    : 0.2, 'SUP_FINE': 0.02}
LEFT_ARROW_KEYS  = {81, 65361, 16777234, 2424832}
RIGHT_ARROW_KEYS = {83, 65363, 16777236, 2555904}
# TODO: add Shift+Arrow to SEEK_KEYS if OpenCV exposes reliable key codes on Pop!_OS.
SEEK_KEYS = {'COARSE'  : {'BCK': LEFT_ARROW_KEYS, 'FWD': RIGHT_ARROW_KEYS},
             'MEDIUM'  : {'BCK': 'z', 'FWD': 'x'},
             'FINE'    : {'BCK': 'a', 'FWD': 's'},
             'SUP_FINE': {'BCK': 'q', 'FWD': 'w'}}
ESC_KEY = 27
QUIT_KEY = ord('q')
MAIN_BAR_WIDTH_RATIO  = 0.25
SECONDARY_BAR_WIDTH_RATIO = 0.125
BAR_GAP_PX = 4
TOP_STRIP_HEIGHT = 28
BOTTOM_STRIP_HEIGHT = 74
FLAG_BAR_HEIGHT = 34
TIMELINE_HEIGHT = 8
TIMELINE_MARGIN_X = 10
TIMELINE_BOTTOM_MARGIN = 8

GREEN  = ( 70, 170,  70)
RED    = ( 40,  40, 220)
YELLOW = (  0, 215, 255)
ORANGE = (  0, 140, 255)
WHITE  = (245, 245, 245)
BLACK  = ( 15,  15,  15)
GRAY   = (195, 195, 195)
BLUE   = (210, 120,  40)
VIDEO_SUFFIXES = {'.mp4', '.avi', '.wmv', '.flv', '.mkv', '.mov', '.m4v'}
ANNOTATION_SUFFIXES = ('.ann', '.txt', '.csv')

@dataclass(frozen=True)
class NormalizedEvent:
    """ One display-ready event interval."""
    start_sec: float
    end_sec: float
    flag: int
    seq: int = 0


@dataclass(frozen=True)
class NormalizedAnnotation:
    """ Annotation payload after conversion to player-friendly event intervals."""
    source_path: Path
    events: tuple[NormalizedEvent, ...]


#* region *** Helpers *************************************************************#

def format_hhmmss(seconds: float)-> str:
    """Format one time value for overlays."""
    total = max(0.0, float(seconds))
    tenths = int(round((total - int(total)) * 10))
    whole = int(total)
    if tenths == 10:
        whole += 1
        tenths = 0
    h, r = divmod(whole, 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}.{tenths}'


def resolve_annotation_path(vid_path:str|Path, ann_path:str|Path|None = None)-> Path:
    """ Resolve one explicit or sibling annotation file."""
    if ann_path is not None:
        path = Path(ann_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    vid_path = Path(vid_path)
    candidates = (vid_path.with_suffix(suffix) for suffix in ANNOTATION_SUFFIXES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No sibling .txt/.ann/.csv annotation file for {vid_path}")


def load_player_annotations(path:str|Path, video_path:str|Path|None = None,
                            fps:float|None = None, total_sec:float|None= None)-> NormalizedAnnotation:
    """ Load one supported annotation file and normalize it for playback."""
    ann_path = Path(path)
    fmt = resolve_format(ann_path)
    if fmt == FMT_EVENT:
        data = load_event_ann(ann_path)
    elif fmt == FMT_UBI:
        data = load_ubi_ann(ann_path, video_path=video_path, fps=fps)
    else:
        raise ValueError(f'Unsupported annotation format for visual player: {fmt}')

    events: list[NormalizedEvent] = []
    for event in data['events']:
        if str(event.get('flag')).strip().lower() in SPECIAL_EVENT_TAGS:
            continue
        start_sec = resolve_event_time(event.get('start'), timeline_end=total_sec)
        end_sec = resolve_event_time(event.get('end'), timeline_end=total_sec)
        if start_sec is None or end_sec is None:
            print(f"[WARN] Open-ended intervals are unsupported in visual player: {ann_path}")
            continue
        if end_sec < start_sec:
            print(f"[WARN] Reversed interval in visual player: {ann_path}")
            continue
        events.append(NormalizedEvent(start_sec=start_sec, end_sec=end_sec,
                                      flag=int(event['flag']), seq=int(event.get('seq', len(events)))))

    events.sort(key=lambda evt: (evt.start_sec, evt.end_sec, evt.flag))
    return NormalizedAnnotation(source_path=ann_path, events=tuple(events))


def _flag_priority(flag:int) -> int:
    if flag == 4:
        return 0
    if flag == 3:
        return 1
    if flag in {1, 2}:
        return 2
    return 3


def _ordered_display_events(events: list[NormalizedEvent]) -> list[NormalizedEvent]:
    """Order active flags by priority and suppress duplicate flags."""
    unique_by_flag = {}
    for event in sorted(events, key=lambda evt: (_flag_priority(evt.flag), evt.seq)):
        unique_by_flag.setdefault(event.flag, event)
    return list(unique_by_flag.values())


def _flag_color(flag:int|None)-> tuple[int, int, int]:
    if flag is None:
        return GREEN
    if flag == 4:
        return RED
    if flag == 3:
        return YELLOW
    return ORANGE


def _window_is_open(title: str) -> bool:
    """Return False after the OS window close button is pressed."""
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def _maximize_window(title: str) -> None:
    """Maximize an OpenCV window when the desktop provides xdotool."""
    xdotool = shutil.which('xdotool')
    if xdotool is None:
        cli_warning("Cannot maximize window: 'xdotool' is not installed")
        return
    result = subprocess.run((xdotool, 'search', '--onlyvisible', '--name', f'^{re.escape(title)}$'),
                            capture_output=True, text=True, check=False)
    window_ids = result.stdout.split()
    if not window_ids:
        cli_warning(f'Cannot maximize window: window not found: {title}')
        return
    activate = subprocess.run((xdotool, 'windowactivate', '--sync', window_ids[0]),
                              capture_output=True, text=True, check=False)
    if activate.returncode != 0:
        detail = activate.stderr.strip() or 'xdotool could not activate the window'
        cli_warning(f'Cannot maximize window: {detail}')
        return
    maximize = subprocess.run((xdotool, 'key', '--clearmodifiers', 'alt+F10'),
                              capture_output=True, text=True, check=False)
    if maximize.returncode != 0:
        detail = maximize.stderr.strip() or 'xdotool maximize shortcut failed'
        cli_warning(f'Cannot maximize window: {detail}')


def _draw_timeline(image:np.ndarray, rect:tuple[int, int, int, int], t_sec:float, total_sec:float)-> None:
    """ Draw the clickable progress bar and its 10 percent tick marks."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), GRAY, thickness=-1)

    ratio = 0.0 if total_sec <= 0 else min(1.0, max(0.0, t_sec/total_sec))
    progress_x = int(round(x1 + ratio * (x2 - x1)))
    cv2.rectangle(image, (x1, y1), (progress_x, y2), BLUE, thickness=-1)
    cv2.rectangle(image, (x1, y1), (x2, y2), BLACK, thickness=1)

    for tick in range(1, 11):
        x = int(round(x1 + (tick/10) * (x2 - x1)))
        cv2.line(image, (x, y1 - 3), (x, y2 + 3), BLACK, 1)
    cv2.circle(image, (progress_x, (y1 + y2)//2), 6, BLACK, thickness=-1)


def _is_char_key(key: int, *chars: str)-> bool:
    return (key & 0xFF) in {ord(ch) for ch in chars}


def _is_seek_key(key: int, level: str, direction: str)-> bool:
    key_group = SEEK_KEYS[level][direction]
    if isinstance(key_group, str):
        return _is_char_key(key, key_group)
    return key in key_group


def _render_display(frame: np.ndarray, time_sec: float, events: list[NormalizedEvent], total_sec: float,
                    video_stem: str, show_navigation: bool = False
                    )-> tuple[np.ndarray, tuple[int, int, int, int], dict[str, tuple[int, int, int, int]]]:
    """Render one video frame plus time text, event bars, and timeline."""
    if frame.ndim == 2:
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    height, width = display.shape[:2]
    top_strip = np.full((TOP_STRIP_HEIGHT, width, 3), WHITE, dtype=np.uint8)
    bottom_strip = np.full((BOTTOM_STRIP_HEIGHT, width, 3), WHITE, dtype=np.uint8)
    merged = np.vstack((top_strip, display, bottom_strip))

    time_text = f'{format_hhmmss(time_sec)} / {format_hhmmss(total_sec)}'
    time_scale = 0.75
    time_thickness = 2

    display_events = events or [None]
    bar_height = FLAG_BAR_HEIGHT
    y1 = TOP_STRIP_HEIGHT + height + 7
    y2 = min(merged.shape[0] - 1, y1 + bar_height)
    x1 = 0
    for index, event in enumerate(display_events):
        ratio = MAIN_BAR_WIDTH_RATIO if index == 0 else SECONDARY_BAR_WIDTH_RATIO
        bar_width = max(100 if index == 0 else 50, int(round(width * ratio)))
        if x1 >= width:
            break
        x2 = min(width - 1, x1 + bar_width)
        cv2.rectangle(merged, (x1, y1), (x2, y2), _flag_color(event.flag if event else None), thickness=-1)

        flag_text = 'none' if event is None else str(event.flag)
        text_size, _ = cv2.getTextSize(flag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = x1 + max(8, (bar_width - text_size[0]) // 2)
        text_y = y1 + (bar_height + text_size[1]) // 2
        cv2.putText(merged, flag_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)
        x1 = x2 + BAR_GAP_PX + 1

    navigation_rects = {}
    time_x = 0
    if show_navigation:
        button_specs = (('< Prev', 'previous'), ('Next >', 'next'))
        button_x = 6
        for label, action in button_specs:
            button_width = 86
            x1 = button_x
            x2 = x1 + button_width
            y1 = 4
            y2 = TOP_STRIP_HEIGHT - 4
            cv2.rectangle(merged, (x1, y1), (x2, y2), GRAY, thickness=-1)
            cv2.rectangle(merged, (x1, y1), (x2, y2), BLACK, thickness=1)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_x = x1 + max(4, (button_width - text_size[0]) // 2)
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(merged, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)
            navigation_rects[action] = (x1, y1, x2, y2)
            button_x = x2 + 4
        time_x = button_x + 4

    time_size, _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX,
                                   time_scale, time_thickness)
    cv2.putText(merged, time_text, (time_x, TOP_STRIP_HEIGHT - 6),
                cv2.FONT_HERSHEY_SIMPLEX, time_scale, BLACK, time_thickness, cv2.LINE_AA)
    stem_scale = 0.65
    stem_thickness = 2
    stem_text = video_stem
    stem_left = time_x + time_size[0] + 12
    available_width = width - stem_left - 8
    stem_size, _ = cv2.getTextSize(stem_text, cv2.FONT_HERSHEY_SIMPLEX,
                                   stem_scale, stem_thickness)
    if stem_size[0] > available_width:
        while stem_text:
            candidate = f'{stem_text}...'
            stem_size, _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX,
                                           stem_scale, stem_thickness)
            if stem_size[0] <= available_width:
                stem_text = candidate
                break
            stem_text = stem_text[:-1]
    if stem_text and available_width > 0:
        stem_size, _ = cv2.getTextSize(stem_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       stem_scale, stem_thickness)
        stem_x = width - stem_size[0] - 8
        cv2.putText(merged, stem_text, (stem_x, TOP_STRIP_HEIGHT - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, stem_scale, BLACK, stem_thickness, cv2.LINE_AA)
    x1 = TIMELINE_MARGIN_X
    x2 = max(x1 + 1, width - TIMELINE_MARGIN_X)
    y2 = TOP_STRIP_HEIGHT + height + BOTTOM_STRIP_HEIGHT - TIMELINE_BOTTOM_MARGIN
    y1 = y2 - TIMELINE_HEIGHT
    timeline_rect = x1, y1, x2, y2
    _draw_timeline(merged, timeline_rect, time_sec, total_sec)

    return merged, timeline_rect, navigation_rects


# endregion

#* region *** API *****************************************************************#

def play_annotated_video(video_path:str|Path, ann_path:str|Path|None = None, **kwargs)-> str:
    """     Play one video with annotation overlays.
    :param   video_path: Video file to play.
    :param   ann_path: annotation file (Optional)
    :param   kwargs: Optional parameters  speed, start_sec, hold_on_end, fps_fallback,
                     window_title, navigation, size, _window_state.
    :return: Playback action: ``end``, ``previous``, ``next``, or ``close``.
    """
    def _tl_time_from_x(x: int, rect: tuple[int, int, int, int]) -> float:
        x1, _, x2, _ = rect
        return min(1.0, max(0.0, (x - x1)/max(1, x2 - x1))) * max(0.0, total_sec)

    def _in_timeline(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and (y1 - 8) <= y <= (y2 + 8)

    def _request_tl_seek(x: int) -> None:
        rect = timeline_state['rect']
        if rect is not None:
            timeline_state['target_sec'] = _tl_time_from_x(x, rect)

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if show_navigation and event == cv2.EVENT_LBUTTONDOWN:
            for action, rect in timeline_state['navigation_rects'].items():
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    timeline_state['action'] = action
                    return
        # rect = timeline_state['rect']
        if timeline_state['rect'] is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN and _in_timeline(x, y, timeline_state['rect']):
            timeline_state['dragging'] = True
            _request_tl_seek(x)
        elif event == cv2.EVENT_MOUSEMOVE and timeline_state['dragging']:
            _request_tl_seek(x)
        elif event == cv2.EVENT_LBUTTONUP:
            if timeline_state['dragging']:
                _request_tl_seek(x)
            timeline_state['dragging'] = False

    def _seek_to_sec(target_sec:float|None) -> None:
        nonlocal frame, frame_idx, ended
        seek_sec = min(max(0.0, target_sec), total_sec if total_sec > 0 else target_sec)
        if total_frames > 0:
            target_frame = min(total_frames - 1, max(0, int(round(seek_sec*fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            frame_idx = target_frame
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec*1000.0)
            frame_idx = int(round(seek_sec * fps))
        frame = None
        ended = False

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = float(kwargs.get('fps_fallback', 25.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = (total_frames/fps) if total_frames > 0 else 0.0
    try:
        ann_file = resolve_annotation_path(video_path, ann_path)
        annotation = load_player_annotations(ann_file, video_path=video_path, fps=fps, total_sec=total_sec)
    except Exception as exc:
        print_color(f'[WARN] Could not load annotation file: {exc}. Playing without annotations.', 'r')
        annotation = NormalizedAnnotation(source_path=Path(ann_path or video_path), events=())

    speed = max(0.05, float(kwargs.get('speed', 1.0)))
    start_sec = max(0.0, float(kwargs.get('start_sec', 0.0)))
    hold_on_end = bool(kwargs.get('hold_on_end', False))
    title = str(kwargs.get('window_title', 'visual_analyzer'))
    window_caption = video_path.stem
    size_mode = kwargs.get('size', 'org')
    if size_mode not in {'org', 'max'}:
        size_mode = float(size_mode)
        if size_mode <= 0:
            raise ValueError("size must be 'org', 'max', or a positive scale")
    window_state = kwargs.get('_window_state')
    keep_window = bool(kwargs.get('_keep_window', False))

    reused_window = (keep_window and window_state is not None
                     and window_state.get('created', False) and _window_is_open(title))
    if not reused_window:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if window_state is not None:
            window_state['created'] = True
    cv2.setWindowTitle(title, window_caption)
    show_navigation = bool(kwargs.get('navigation', False))
    timeline_state = {'target_sec': None, 'dragging': False, 'rect': None,
                      'navigation_rects': {}, 'action': None}
    cv2.setMouseCallback(title, _on_mouse)

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    frame = None
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    ended, paused = False, False
    size_initialized = reused_window and window_state is not None  and window_state.get('initialized', False)

    while True:
        if timeline_state['target_sec'] is not None:
            _seek_to_sec(timeline_state['target_sec'])
            timeline_state['target_sec'] = None

        if not ended and (not paused or frame is None):
            ok, next_frame = cap.read()
            if not ok:
                if hold_on_end and frame is not None:
                    ended = True
                    paused = True
                else:
                    break
            else:
                frame = next_frame
                frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        if frame is None:
            if hold_on_end:
                key = cv2.waitKeyEx(50)
                if key == ESC_KEY or not _window_is_open(title):
                    timeline_state['action'] = 'close'
                    break
                if (key & 0xFF) == QUIT_KEY:
                    break
                if timeline_state['action'] is not None:
                    break
                if show_navigation and _is_char_key(key, 'b', 'n'):
                    timeline_state['action'] = 'previous' if _is_char_key(key, 'b') else 'next'
                    break
                continue
            else:
                break

        time_sec = frame_idx/fps
        events = [event for event in annotation.events if event.start_sec <= time_sec <= event.end_sec]
        events = _ordered_display_events(events)
        display, tl_rect, navg_rects = _render_display(
            frame, time_sec, events, total_sec, video_path.stem, show_navigation)
        timeline_state['rect'] = tl_rect
        timeline_state['navigation_rects'] = navg_rects
        cv2.imshow(title, display)

        if not size_initialized:
            if size_mode == 'max':
                cv2.waitKey(1)
                _maximize_window(window_caption)
            else:
                scale = 1.0 if size_mode == 'org' else size_mode
                cv2.resizeWindow(title, max(1, int(display.shape[1] * scale)),
                                 max(1, int(display.shape[0] * scale)))
            if window_state is not None:
                window_state['initialized'] = True
            size_initialized = True

        delay_ms = max(1, int(1000.0 / fps / speed)) if not paused else 50
        key = cv2.waitKeyEx(delay_ms)

        if key == ESC_KEY or not _window_is_open(title):
            timeline_state['action'] = 'close'
            break
        if show_navigation and _is_char_key(key, 'b', 'n'):
            timeline_state['action'] = 'previous' if _is_char_key(key, 'b') else 'next'
            break
        if timeline_state['action'] is not None:
            break
        if (key & 0xFF) == QUIT_KEY and not paused:
            break
        if _is_char_key(key, ' '):
            if ended:
                _seek_to_sec(0.0)
                paused = False
                continue
            paused = not paused
            continue

        if _is_seek_key(key, 'COARSE', 'BCK'):
            step = SEEK_JUMPS['MEDIUM'] if paused else SEEK_JUMPS['COARSE']
            _seek_to_sec(time_sec - step)
            continue
        if _is_seek_key(key, 'COARSE', 'FWD'):
            step = SEEK_JUMPS['MEDIUM'] if paused else SEEK_JUMPS['COARSE']
            _seek_to_sec(time_sec + step)
            continue

        if _is_seek_key(key, 'MEDIUM', 'BCK'):
            _seek_to_sec(time_sec - SEEK_JUMPS['MEDIUM'])
            continue
        if _is_seek_key(key, 'MEDIUM', 'FWD'):
            _seek_to_sec(time_sec + SEEK_JUMPS['MEDIUM'])
            continue

        if paused and _is_seek_key(key, 'FINE', 'BCK'):
            _seek_to_sec(time_sec - SEEK_JUMPS['FINE'])
            continue
        if paused and _is_seek_key(key, 'FINE', 'FWD'):
            _seek_to_sec(time_sec + SEEK_JUMPS['FINE'])
            continue
        if paused and _is_seek_key(key, 'SUP_FINE', 'BCK'):
            _seek_to_sec(time_sec - SEEK_JUMPS['SUP_FINE'])
            continue
        if paused and _is_seek_key(key, 'SUP_FINE', 'FWD'):
            _seek_to_sec(time_sec + SEEK_JUMPS['SUP_FINE'])
            continue

    cap.release()
    if not keep_window:
        cv2.destroyWindow(title)
    return timeline_state['action'] or 'end'


def play_multi_vid(videos:str|Path|list|tuple|set, ann_path:str|Path|None = None, **kwargs) -> None:
    """Play one or more videos resolved from paths, directories, or playlists."""
    def is_video(p: Path)-> bool:
        return p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES

    def load_playlist(path: Path) -> list[Path]:
        paths = []
        for line in path.read_text(encoding='utf-8').splitlines():
            entry = line.strip()
            if not entry or entry.startswith('#'):
                continue
            item = Path(entry)
            paths.append(item if item.is_absolute() else Path.cwd() / item)
        return paths

    def resolve_source(src:str|Path) -> list[Path]:
        path = Path(src)
        if path.is_dir():
            return [item for item in path.rglob('*') if is_video(item)]
        if not path.exists():
            cli_warning(f'Skipping missing source: {path}')
            return []
        if path.is_file() and not is_video(path):
            try:
                return load_playlist(path)
            except OSError as err:
                cli_warning(f'Skipping unreadable playlist {path}: {err}')
                return []
        return [path]

    def resolve_annotation(vid_p:Path, ann_p: Path | None)-> Path|None:
        if ann_p is None:
            return None
        if ann_p.is_file():
            return ann_p
        if not ann_p.is_dir():
            cli_warning(f'Annotation path not found: {ann_p}')
            return ann_p
        for suffix in ANNOTATION_SUFFIXES:
            candidate = ann_p/f'{vid_p.stem}{suffix}'
            if candidate.is_file():
                return candidate
        cli_warning(f'No annotation found for {vid_p.name} in {ann_p}')
        return ann_p/f'{vid_p.stem}.ann'

    play_kwargs = dict(kwargs)
    order = play_kwargs.pop('order', None)
    reverse = bool(play_kwargs.pop('reverse', False))
    hold_on_end = bool(play_kwargs.get('hold_on_end', False))

    sources = []
    for vid in as_collection(videos):
        sources += resolve_source(vid)
    valid_sources = []
    for src in sources:
        src_path = Path(src)
        if is_video(src_path):
            valid_sources.append(src_path)
        else:
            cli_warning(f'Skipping unsupported video source: {src_path}')
    sources = valid_sources

    if order == 'name':
        sources.sort(key=lambda path: path.name.lower())
    elif order == 'date':
        sources.sort(key=lambda path: path.stat().st_mtime)
    elif order == 'size':
        sources.sort(key=lambda path: path.stat().st_size)
    elif order is not None:
        raise ValueError("order must be 'name', 'date', or 'size'")
    if reverse:
        sources.reverse()

    annotation = Path(ann_path) if ann_path is not None else None
    if annotation is not None:
        if annotation.is_file() and len(sources) > 1:
            raise ValueError('An annotation file can be used only with one video')

    if not sources:
        return

    video_count = len(sources)
    video_index = 0
    window_state = {}
    window_title = str(play_kwargs.setdefault('window_title', 'visual_analyzer'))
    try:
        while True:
            video = sources[video_index]
            current_ann = resolve_annotation(video, annotation)
            current_kwargs = dict(play_kwargs)
            if hold_on_end:
                current_kwargs['hold_on_end'] = video_index == len(sources) - 1
            current_kwargs['navigation'] = video_count > 1
            current_kwargs['_window_state'] = window_state
            current_kwargs['_keep_window'] = True
            try:
                action = play_annotated_video(video, current_ann, **current_kwargs)
            except Exception as error:
                cli_warning(f'Skipping {video}: {type(error).__name__}: {error}')
                action = 'end'

            if action == 'close':
                break
            if action == 'previous':
                video_index = (video_index - 1) % video_count
            elif action == 'next':
                video_index = (video_index + 1) % video_count
            elif video_index == video_count - 1:
                break
            else:
                video_index += 1
    finally:
        if window_state.get('created', False):
            try:
                cv2.destroyWindow(window_title)
            except cv2.error:
                pass


# endregion

#* region *** CLI *****************************************************************#

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Play video(s) with synced annotation event overlays.')
    parser.add_argument('video_path', type=Path, help='video, directory, or playlist file')
    parser.add_argument('annotation_path', type=Path, nargs='?', default=None, help='annotation file or dir; defaults to sibling annotations')
    parser.add_argument('-sd', '--speed', type=float, default=1.0, help='playback speed factor')
    parser.adargument('-s', '--start', type=float, default=0.0, help='start time in seconds')
    parser.add_argument('-o', '--order', choices=('name', 'date', 'size'), default=None, help='optional playback order')
    parser.add_argument('-r', '--reverse', action='store_true', help='reverse playback order')
    parser.add_argument('-z', '--size', default='org', help="window size: 'org', 'max', or a scale factor")
    parser.add_argument('--hold', action='store_true', help='keep the player open until Esc is pressed')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    play_multi_vid(args.video_path, args.annotation_path,
                   speed=args.speed, start_sec=args.start, hold_on_end=args.hold,
                   order=args.order, reverse=args.reverse, size=args.size)

# endregion
#442(2,4,1)-> 423->415 ->428(2,1,1)  #527(2,11,1)->640(2,11,2)->636(2,1,1)
#659(2,1,4)->652(1,1,1)

if __name__ == '__main__':
    main()
