"""Play videos with synced event-annotation overlays.

Usage:
    python visual_analyzer.py <video.mp4> [annotation.ann|annotation.txt|annotation.csv]
    python visual_analyzer.py <video.mp4> <annotation> --start 12.5 --hold

The player supports default event annotations and UBI CSV annotations through
annotations.py. It draws active event flags below the video, supports timeline
mouse seeking, and exposes keyboard seek controls through SEEK_JUMPS/SEEK_KEYS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from annotations import (load_event_ann, load_ubi_ann, resolve_event_time, resolve_format,
                         FMT_EVENT, FMT_UBI, SPECIAL_EVENT_TAGS)

from common.my_local_utils import print_color


SEEK_JUMPS = {'COARSE'  : 5.0,
              'MEDIUM'  : 1.0,
              'FINE'    : 0.2,
              'SUP_FINE': 0.02}
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


@dataclass(frozen=True)
class NormalizedEvent:
    """One display-ready event interval."""
    start_sec: float
    end_sec: float
    flag: int
    seq: int = 0


@dataclass(frozen=True)
class NormalizedAnnotation:
    """Annotation payload after conversion to player-friendly event intervals."""
    source_path: Path
    events: tuple[NormalizedEvent, ...]


def format_hhmmss(seconds: float) -> str:
    """Format one time value for overlays."""
    total = max(0.0, float(seconds))
    tenths = int(round((total - int(total)) * 10))
    whole = int(total)
    if tenths == 10:
        whole += 1
        tenths = 0
    hrs, rem = divmod(whole, 3600)
    mins, secs = divmod(rem, 60)
    return f'{hrs:02d}:{mins:02d}:{secs:02d}.{tenths}'


def resolve_annotation_path(video_path: str | Path, ann_path: str | Path | None = None) -> Path:
    """Resolve one explicit or sibling annotation file."""
    if ann_path is not None:
        path = Path(ann_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    video_path = Path(video_path)
    candidates = (video_path.with_suffix('.txt'), video_path.with_suffix('.ann'), video_path.with_suffix('.csv'))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'No sibling .txt/.ann/.csv annotation file for {video_path}')


def load_player_annotations(path:str|Path, video_path:str|Path|None = None,
                            fps:float|None = None, total_sec:float|None = None) -> NormalizedAnnotation:
    """Load one supported annotation file and normalize it for playback."""
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


def _flag_priority(flag: int) -> int:
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


def _flag_color(flag: int | None) -> tuple[int, int, int]:
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


def _draw_timeline(image:np.ndarray, rect:tuple[int, int, int, int],
                   time_sec: float, total_sec: float) -> None:
    """Draw the clickable progress bar and its 10 percent tick marks."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), GRAY, thickness=-1)

    ratio = 0.0 if total_sec <= 0 else min(1.0, max(0.0, time_sec / total_sec))
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


def _render_display(frame: np.ndarray, time_sec: float, events: list[NormalizedEvent], total_sec: float
                    )-> tuple[np.ndarray, tuple[int, int, int, int]]:
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

    cv2.putText(merged, time_text, (0, TOP_STRIP_HEIGHT - 6),
                cv2.FONT_HERSHEY_SIMPLEX, time_scale, BLACK, time_thickness, cv2.LINE_AA)
    x1 = TIMELINE_MARGIN_X
    x2 = max(x1 + 1, width - TIMELINE_MARGIN_X)
    y2 = TOP_STRIP_HEIGHT + height + BOTTOM_STRIP_HEIGHT - TIMELINE_BOTTOM_MARGIN
    y1 = y2 - TIMELINE_HEIGHT
    timeline_rect = x1, y1, x2, y2
    _draw_timeline(merged, timeline_rect, time_sec, total_sec)

    return merged, timeline_rect


def play_annotated_video(video_path:str|Path, ann_path:str|Path|None = None, **kwargs)-> None:
    """ Play one video with annotation overlays.
    :param  video_path: Video file to play.
    :param   ann_path: annotation file (Optional)
    :param   kwargs: Optional parameters  speed, start_sec, hold_on_end, fps_fallback,  window_title.
    """
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = float(kwargs.get('fps_fallback', 25.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = (total_frames / fps) if total_frames > 0 else 0.0
    try:
        ann_file = resolve_annotation_path(video_path, ann_path)
        annotation = load_player_annotations(ann_file, video_path=video_path, fps=fps, total_sec=total_sec)
    except Exception as exc:
        print_color(f'[WARN] Could not load annotation file: {exc}. Playing without annotations.', 'r')
        annotation = NormalizedAnnotation(source_path=Path(ann_path or video_path), events=())

    speed = max(0.05, float(kwargs.get('speed', 1.0)))
    start_sec = max(0.0, float(kwargs.get('start_sec', 0.0)))
    hold_on_end = bool(kwargs.get('hold_on_end', False))
    title = str(kwargs.get('window_title', f'visual_analyzer: {video_path.name}'))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    timeline_state = {'target_sec': None, 'dragging': False, 'rect': None}

    def _timeline_time_from_x(x: int, rect: tuple[int, int, int, int]) -> float:
        x1, _, x2, _ = rect
        ratio = min(1.0, max(0.0, (x - x1)/max(1, x2 - x1)))
        return ratio*max(0.0, total_sec)

    def _in_timeline(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and (y1 - 8) <= y <= (y2 + 8)

    def _request_timeline_seek(x: int) -> None:
        rect = timeline_state['rect']
        if rect is not None:
            timeline_state['target_sec'] = _timeline_time_from_x(x, rect)

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        rect = timeline_state['rect']
        if rect is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN and _in_timeline(x, y, rect):
            timeline_state['dragging'] = True
            _request_timeline_seek(x)
        elif event == cv2.EVENT_MOUSEMOVE and timeline_state['dragging']:
            _request_timeline_seek(x)
        elif event == cv2.EVENT_LBUTTONUP:
            if timeline_state['dragging']:
                _request_timeline_seek(x)
            timeline_state['dragging'] = False

    cv2.setMouseCallback(title, _on_mouse)

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    paused = False
    frame = None
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    ended = False

    def _seek_to_sec(target_sec: float) -> None:
        nonlocal frame, frame_idx, ended
        seek_sec = min(max(0.0, target_sec), total_sec if total_sec > 0 else target_sec)
        if total_frames > 0:
            target_frame = min(total_frames - 1, max(0, int(round(seek_sec * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            frame_idx = target_frame
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000.0)
            frame_idx = int(round(seek_sec * fps))
        frame = None
        ended = False

    while True:
        if timeline_state['target_sec'] is not None:
            target_sec = float(timeline_state['target_sec'])
            timeline_state['target_sec'] = None
            _seek_to_sec(target_sec)

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
                if key == ESC_KEY or (key & 0xFF) == QUIT_KEY or not _window_is_open(title):
                    break
                continue
            else:
                break

        time_sec = frame_idx/fps
        events = [event for event in annotation.events if event.start_sec <= time_sec <= event.end_sec]
        events = _ordered_display_events(events)
        display, timeline_rect = _render_display(frame, time_sec, events, total_sec)
        timeline_state['rect'] = timeline_rect
        cv2.imshow(title, display)

        delay_ms = max(1, int(1000.0 / fps / speed)) if not paused else 50
        key = cv2.waitKeyEx(delay_ms)

        if key == ESC_KEY or ((key & 0xFF) == QUIT_KEY and not paused) or not _window_is_open(title):
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
    cv2.destroyWindow(title)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Play one video with synced annotation event overlay (TXT/ANN/CSV...).')
    parser.add_argument('video_path', type=Path, help='path to the video file')
    parser.add_argument('annotation_path', type=Path, nargs='?', default=None, help='path to annotation file; defaults to sibling file')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed factor')
    parser.add_argument('--start', type=float, default=0.0, help='start time in seconds')
    parser.add_argument('--hold', action='store_true', help='keep the player open until Esc is pressed')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    play_annotated_video(args.video_path, args.annotation_path,
                         speed=args.speed, start_sec=args.start, hold_on_end=args.hold)

#442(2,4,1)- > 423->415
if __name__ == '__main__':
    main()
