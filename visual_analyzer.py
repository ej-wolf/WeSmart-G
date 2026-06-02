"""Small video player with synced annotation side bar.

First version supports interval annotations from `.txt` / `.ann` files.
Each non-comment row is expected to contain:
    start_time, end_time, event_flag

Example:
    00:00:49,\t00:01:06,\t4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SEEK_STEP_SEC = 5.0
MAIN_BAR_WIDTH_RATIO = 0.25
SECONDARY_BAR_WIDTH_RATIO = 0.125
BAR_GAP_PX = 4
TOP_STRIP_HEIGHT = 28
BOTTOM_STRIP_HEIGHT = 48
TEXT_SCALE = 0.9
TEXT_THICKNESS = 2

GREEN = (70, 170, 70)
RED = (40, 40, 220)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)
WHITE = (245, 245, 245)
BLACK = (15, 15, 15)


@dataclass(frozen=True)
class NormalizedEvent:
    start_sec: float
    end_sec: float
    flag: int
    seq: int = 0


@dataclass(frozen=True)
class NormalizedAnnotation:
    source_path: Path
    events: tuple[NormalizedEvent, ...]


def parse_time_str(text: str | None) -> float | None:
    """Convert HH:MM:SS / MM:SS / SS into seconds."""
    if text is None:
        return None
    text = str(text).strip().rstrip(',')
    if text == '':
        return None
    parts = [int(part) for part in text.split(':')]
    if len(parts) == 3:
        hrs, mins, secs = parts
    elif len(parts) == 2:
        hrs = 0
        mins, secs = parts
    else:
        hrs, mins, secs = 0, 0, parts[0]
    return float(hrs * 3600 + mins * 60 + secs)


def format_hhmmss(seconds: float) -> str:
    """Format one time value for overlays."""
    total = max(0, int(round(float(seconds))))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f'{hrs:02d}:{mins:02d}:{secs:02d}'


def resolve_annotation_path(video_path: str | Path, ann_path: str | Path | None = None) -> Path:
    """Resolve one explicit or sibling annotation file."""
    if ann_path is not None:
        path = Path(ann_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    video_path = Path(video_path)
    candidates = (video_path.with_suffix('.txt'), video_path.with_suffix('.ann'))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'No sibling .txt/.ann annotation file for {video_path}')


def load_txt_annotations(path: str | Path) -> NormalizedAnnotation:
    """Load one TXT/ANN interval file into normalized events."""
    ann_path = Path(path)
    events: list[NormalizedEvent] = []
    with ann_path.open('r', encoding='utf-8') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = [part.strip() for part in line.split(',\t')]
            if len(parts) != 3:
                parts = [part.strip() for part in line.split('\t')]
            if len(parts) != 3:
                print(f'[WARN] Invalid annotation row at {ann_path}:{line_no}: {raw_line.rstrip()}')
                continue

            start_txt, end_txt, flag_txt = parts
            try:
                start_sec = parse_time_str(start_txt)
                end_sec = parse_time_str(end_txt)
                flag = int(flag_txt.rstrip(','))
            except ValueError:
                print(f'[WARN] Invalid annotation row at {ann_path}:{line_no}: {raw_line.rstrip()}')
                continue

            if start_sec is None or end_sec is None:
                print(f'[WARN] Open-ended intervals are unsupported at {ann_path}:{line_no}')
                continue
            if end_sec < start_sec:
                print(f'[WARN] Reversed interval at {ann_path}:{line_no}: {raw_line.rstrip()}')
                continue

            events.append(NormalizedEvent(start_sec=start_sec, end_sec=end_sec, flag=flag, seq=len(events)))

    events.sort(key=lambda event: (event.start_sec, event.end_sec, event.flag))
    return NormalizedAnnotation(source_path=ann_path, events=tuple(events))


def _active_events(events: Iterable[NormalizedEvent], time_sec: float) -> list[NormalizedEvent]:
    return [event for event in events if event.start_sec <= time_sec <= event.end_sec]


def _flag_priority(flag: int) -> int:
    if flag == 4:
        return 0
    if flag == 3:
        return 1
    if flag in {1, 2}:
        return 2
    return 3


def _ordered_display_events(events: list[NormalizedEvent]) -> list[NormalizedEvent]:
    return sorted(events, key=lambda event: (_flag_priority(event.flag), event.seq))


def _flag_color(flag: int | None) -> tuple[int, int, int]:
    if flag is None:
        return GREEN
    if flag == 4:
        return RED
    if flag == 3:
        return YELLOW
    return ORANGE


def _render_display(frame: np.ndarray, time_sec: float, events: list[NormalizedEvent],
                    total_sec: float) -> np.ndarray:
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
    bar_height = max(24, BOTTOM_STRIP_HEIGHT - 14)
    y1 = TOP_STRIP_HEIGHT + height + (BOTTOM_STRIP_HEIGHT - bar_height) // 2
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

    return merged


def play_annotated_video(video_path: str | Path, ann_path: str | Path | None = None, **kwargs) -> None:
    """Play one video with a synced side bar derived from TXT/ANN annotations."""
    video_path = Path(video_path)
    ann_file = resolve_annotation_path(video_path, ann_path)
    annotation = load_txt_annotations(ann_file)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = float(kwargs.get('fps_fallback', 25.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = (total_frames / fps) if total_frames > 0 else 0.0

    seek_step = float(kwargs.get('seek_step_sec', SEEK_STEP_SEC))
    speed = max(0.05, float(kwargs.get('speed', 1.0)))
    start_sec = max(0.0, float(kwargs.get('start_sec', 0.0)))
    title = str(kwargs.get('window_title', f'visual_analyzer: {video_path.name}'))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    paused = False
    frame = None
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

    while True:
        if not paused or frame is None:
            ok, next_frame = cap.read()
            if not ok:
                break
            frame = next_frame
            frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        time_sec = frame_idx / fps
        events = _ordered_display_events(_active_events(annotation.events, time_sec))
        display = _render_display(frame, time_sec, events, total_sec)
        cv2.imshow(title, display)

        delay_ms = max(1, int(1000.0 / fps / speed)) if not paused else 50
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (27, ord('q')):
            break
        if key == ord(' '):
            paused = not paused
            continue
        if key in (81, ord('a'), ord('j')):
            target_sec = max(0.0, time_sec - seek_step)
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            frame = None
            paused = True
            continue
        if key in (83, ord('d'), ord('l')):
            target_sec = min(total_sec, time_sec + seek_step) if total_sec > 0 else (time_sec + seek_step)
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            frame = None
            paused = True
            continue

    cap.release()
    cv2.destroyWindow(title)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Play one video with synced TXT/ANN event overlay.')
    parser.add_argument('video_path', type=Path, help='path to the video file')
    parser.add_argument('annotation_path', type=Path, nargs='?', default=None,
                        help='path to .txt/.ann annotation file; defaults to sibling file')
    parser.add_argument('--seek-step', type=float, default=SEEK_STEP_SEC, help='seek step in seconds')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed factor')
    parser.add_argument('--start', type=float, default=0.0, help='start time in seconds')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    play_annotated_video(args.video_path, args.annotation_path,
                         seek_step_sec=args.seek_step,
                         speed=args.speed,
                         start_sec=args.start)


if __name__ == '__main__':
    main()
