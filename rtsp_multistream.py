#!/usr/bin/env python3
"""Run YOLO-pose on many live/video streams and keep rolling payload windows.

This is the stripped multistream runtime used as the base for later temporal
model integration:

RTSP/video readers -> latest frame per stream -> batched YOLO-pose ->
dataset-like detection payload windows -> dashboard / latest payload JSON.

Current scope:
- no dependency on infer_video_hysteresis
- no dependency on src.data.features
- no temporal model inference yet

Design notes for adding TMS later:
- Do not put model inference inside `stream_reader()`. Reader threads should
  only keep the newest decoded frame so one slow model cannot block input.
- The sampled raw image is `InferenceItem.frame`. It is an OpenCV BGR frame
  copied before drawing overlays. RGB/video models should take their frame
  window from this point, converting BGR -> RGB if needed.
- `payload` created by `build_payload()` matches the JSON-like dataset frame
  schema already used elsewhere: `f`, `t`, `group_events`, `detection_list`.
- `StreamState.sample_window` stores only these pose payloads, not raw images.
  If a TMS model needs RGB frame clips, add a separate per-stream deque for
  sampled raw frames alongside `sample_window`.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tms_runtime import (
    TemporalWindowSpec,
    collect_temporal_probe_window,
    resolve_temporal_probes,
    temporal_probe_status_line,
)

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    yaml = None


ROOT = Path(__file__).resolve().parent

COCO_SKELETON = ((5,  7),
                 (7,  9),
                 (6,  8),
                 (8, 10),
                 (5,  6),
                 (5, 11),
                 (6, 12),
                 (11, 12),
                 (11, 13),
                 (13, 15),
                 (12, 14),
                 (14, 16),
                 (0,   1),
                 (0,   2),
                 (1,   3),
                 (2,   4),)


@dataclass
class SamplingConfig:
    requested_mode: str
    resolved_mode: str
    step_frames: int
    sample_period_sec: float | None


@dataclass
class InferenceItem:
    # One sampled frame selected from a stream for expensive inference.
    # `frame` is raw BGR image data before keypoint/box drawing. Use this for
    # RGB/video models such as TMS, not `latest_display`, which has overlays.
    state: "StreamState"
    frame: np.ndarray
    frame_index: int
    time_sec: float


@dataclass
class StreamState:
    # All mutable state for one camera lives here. The reader thread writes the
    # latest raw frame; the main inference loop reads from it and writes model
    # outputs. Keep this split when adding another model.
    stream_id: str
    url: str
    display_name: str
    payload_history_sec: float
    payload_min_frames: int
    source_is_file: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: np.ndarray | None = None
    frame_index: int = -1
    frame_time_sec: float = 0.0
    frame_wall_time: float = 0.0
    first_frame_wall_time: float | None = None
    connected: bool = False
    last_error: str = ""
    reconnect_count: int = 0
    last_inferred_frame_index: int = -1
    next_sample_frame_index: int = 0
    next_sample_time_sec: float = 0.0
    inferred_count: int = 0
    latest_det_count: int = 0
    latest_infer_time: float = 0.0
    latest_display: np.ndarray | None = None
    latest_payload: dict[str, Any] | None = None
    dashboard_position: tuple[int, int] | None = None
    sample_count: int = 0
    # TMS online runtime state: shared buffer health + per-probe readiness.
    payload_window_span_sec: float = 0.0
    critical_low_buffer: bool = True
    temporal_probe_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    temporal_probe_last_trigger_t: dict[str, float] = field(default_factory=dict)
    # Rolling window of YOLO-pose payloads in the same JSON-like format used by
    # the training data. This is pose metadata, not image pixels.
    sample_window: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.sample_window = deque()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs. Install pyyaml or run in the project environment.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def resolve_sampling(config: dict[str, Any]) -> SamplingConfig:
    sampling_cfg = dict(config.get("sampling", {}) or {})
    requested_mode = str(sampling_cfg.get("mode", "auto")).strip().lower()
    if requested_mode not in {"auto", "frames", "time"}:
        raise ValueError("sampling.mode must be one of: auto, frames, time")

    step_frames = int(first_present(
        sampling_cfg.get("step_frames"),
        sampling_cfg.get("step"),
        default=5,
    ))
    step_frames = max(step_frames, 1)

    sample_period_sec = first_present(
        sampling_cfg.get("sample_period_sec"),
        sampling_cfg.get("period_sec"),
        default=None,
    )
    if sample_period_sec is not None:
        sample_period_sec = float(sample_period_sec)
        if sample_period_sec <= 0:
            raise ValueError("sampling.sample_period_sec must be positive")

    if requested_mode == "time":
        if sample_period_sec is None:
            raise ValueError("sampling.mode=time requires sampling.sample_period_sec")
        return SamplingConfig(requested_mode, "time", step_frames, sample_period_sec)

    if requested_mode == "frames":
        return SamplingConfig(requested_mode, "frames", step_frames, sample_period_sec)

    if sample_period_sec is not None:
        return SamplingConfig(requested_mode, "time", step_frames, sample_period_sec)
    return SamplingConfig(requested_mode, "frames", step_frames, None)


def resolve_payload_buffer_policy(config: dict[str, Any]) -> tuple[float, int]:
    sampling_cfg = dict(config.get("sampling", {}) or {})
    payload_history_sec = float(first_present(
        sampling_cfg.get("payload_history_sec"),
        default=8.0,
    ))
    payload_min_frames = int(first_present(
        sampling_cfg.get("payload_min_frames"),
        default=16,
    ))
    if payload_history_sec <= 0:
        raise ValueError("sampling.payload_history_sec must be positive")
    if payload_min_frames <= 0:
        raise ValueError("sampling.payload_min_frames must be positive")
    return payload_history_sec, payload_min_frames


# TMS online buffer support: shared payload-history pruning and health tracking.
def normalize_imgsz(value: Any) -> int | tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("model.imgsz list/tuple must contain [height, width]")
        return int(value[0]), int(value[1])
    return int(value)


def resolve_stream_source(url: str, config_dir: Path) -> str:
    value = str(url).strip()
    if "://" in value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def is_file_source(url: str) -> bool:
    value = str(url).strip()
    if "://" in value:
        return False
    return Path(value).exists()


def enabled_streams(
    config: dict[str, Any],
    config_dir: Path,
    payload_history_sec: float,
    payload_min_frames: int,
) -> list[StreamState]:
    stream_cfgs = config.get("streams", [])
    if not isinstance(stream_cfgs, list) or not stream_cfgs:
        raise ValueError("Config must contain a non-empty streams list")

    states: list[StreamState] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(stream_cfgs):
        if not isinstance(item, dict):
            raise ValueError(f"streams[{index}] must be a mapping")
        if item.get("enabled", True) is False:
            continue
        stream_id = str(item.get("id") or item.get("name") or f"stream_{index}")
        if stream_id in seen_ids:
            raise ValueError(f"Duplicate stream id: {stream_id}")
        url = item.get("url")
        if not url:
            raise ValueError(f"streams[{index}] is missing url")
        seen_ids.add(stream_id)
        resolved_url = resolve_stream_source(str(url), config_dir)
        states.append(
            StreamState(
                stream_id=stream_id,
                url=resolved_url,
                display_name=str(item.get("display_name") or stream_id),
                payload_history_sec=float(payload_history_sec),
                payload_min_frames=int(payload_min_frames),
                source_is_file=is_file_source(resolved_url),
            )
        )

    if not states:
        raise ValueError("No enabled streams found in config")
    return states


def stream_ids_by_enabled_state(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    enabled: list[str] = []
    disabled: list[str] = []
    for index, item in enumerate(config.get("streams", []) or []):
        if not isinstance(item, dict):
            continue
        stream_id = str(item.get("id") or item.get("name") or f"stream_{index}")
        if item.get("enabled", True) is False:
            disabled.append(stream_id)
        else:
            enabled.append(stream_id)
    return enabled, disabled


def opencv_backend(name: str) -> int:
    name = str(name or "ffmpeg").strip().lower()
    if name == "any":
        return cv2.CAP_ANY
    if name == "ffmpeg":
        return cv2.CAP_FFMPEG
    raise ValueError("capture.backend must be 'ffmpeg' or 'any'")


def stream_reader(
    state: StreamState,
    stop_event: threading.Event,
    backend: int,
    reconnect_delay_sec: float,
    read_sleep_sec: float,
    loop_files: bool,
    play_in_realtime: bool,
) -> None:
    # Reader threads are intentionally lightweight: decode RTSP/video and keep
    # only the newest frame. Temporal models should consume sampled copies in
    # the main loop below, not from here.
    frame_index = -1
    while not stop_event.is_set():
        cap = cv2.VideoCapture(state.url, backend)
        if not cap.isOpened():
            with state.lock:
                state.connected = False
                state.last_error = "cannot open stream"
                state.reconnect_count += 1
            time.sleep(reconnect_delay_sec)
            continue

        with state.lock:
            state.connected = True
            state.last_error = ""

        file_wall_t0: float | None = None
        file_pts_t0: float | None = None

        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    with state.lock:
                        state.connected = False
                        state.last_error = "eof" if state.source_is_file else "read failed"
                        state.reconnect_count += 1
                    break

                now = time.monotonic()
                frame_index += 1
                use_media_time = False
                frame_time_sec = now
                if state.source_is_file:
                    pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    if pos_msec > 0.0:
                        use_media_time = True
                        frame_time_sec = pos_msec
                        if play_in_realtime:
                            if file_pts_t0 is None:
                                file_pts_t0 = pos_msec
                                file_wall_t0 = now
                            else:
                                target_wall = float(file_wall_t0) + (pos_msec - float(file_pts_t0))
                                sleep_sec = target_wall - now
                                if sleep_sec > 0:
                                    time.sleep(sleep_sec)
                                    now = time.monotonic()

                with state.lock:
                    if state.first_frame_wall_time is None:
                        state.first_frame_wall_time = now
                    state.frame = frame
                    state.frame_index = frame_index
                    state.frame_wall_time = now
                    if use_media_time:
                        state.frame_time_sec = float(frame_time_sec)
                    else:
                        state.frame_time_sec = now - float(state.first_frame_wall_time)
                    state.connected = True
                    state.last_error = ""

                if read_sleep_sec > 0:
                    time.sleep(read_sleep_sec)
        finally:
            cap.release()

        if state.source_is_file and not loop_files:
            return

        if not stop_event.is_set():
            time.sleep(reconnect_delay_sec)


def should_sample_state(state: StreamState, sampling: SamplingConfig) -> InferenceItem | None:
    with state.lock:
        if state.frame is None:
            return None
        frame_index = int(state.frame_index)
        if frame_index == state.last_inferred_frame_index:
            return None

        if sampling.resolved_mode == "frames":
            if frame_index < state.next_sample_frame_index:
                return None
            state.next_sample_frame_index = frame_index + int(sampling.step_frames)
        else:
            sample_period_sec = float(sampling.sample_period_sec or 0.0)
            if state.frame_time_sec < state.next_sample_time_sec:
                return None
            state.next_sample_time_sec = float(state.frame_time_sec) + sample_period_sec

        state.last_inferred_frame_index = frame_index
        frame = state.frame.copy()
        return InferenceItem(
            state=state,
            frame=frame,
            frame_index=frame_index,
            time_sec=float(state.frame_time_sec),
        )


def tensor_item(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def result_to_detection_list(result: Any, conf_thresh: float, ignored_classes: set[int]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return detections

    kpts_xyn = result.keypoints.xyn
    kpts_conf = result.keypoints.conf

    for det_index, box in enumerate(result.boxes):
        cls_id = int(tensor_item(box.cls))
        conf = float(tensor_item(box.conf))
        if conf < conf_thresh or cls_id in ignored_classes:
            continue

        x1, y1, x2, y2 = [float(value) for value in box.xyxyn[0].detach().cpu().tolist()]
        keypoints_norm = kpts_xyn[det_index].detach().cpu()
        if kpts_conf is None:
            conf_values = torch.ones((keypoints_norm.shape[0], 1), dtype=keypoints_norm.dtype)
        else:
            conf_values = kpts_conf[det_index].detach().cpu().unsqueeze(-1)
        keypoints_xyc = torch.cat([keypoints_norm, conf_values], dim=-1)

        detections.append(
            {
                "class": cls_id,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "key_points": keypoints_xyc.flatten().tolist(),
            }
        )
    return detections


def color_for_class(cls_id: int) -> tuple[int, int, int]:
    palette = (
        (0, 220, 255),
        (0, 180, 0),
        (255, 120, 0),
        (255, 0, 180),
        (140, 120, 255),
        (255, 255, 0),
    )
    return palette[int(cls_id) % len(palette)]


def draw_pose_overlay(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    names: dict[int, str] | list[str] | None,
    show_boxes: bool,
    show_keypoints: bool,
    show_skeleton: bool,
    keypoint_conf: float,
) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]

    for det in detections:
        cls_id = int(det.get("class", 0))
        conf = float(det.get("conf", 0.0))
        color = color_for_class(cls_id)
        x1n, y1n, x2n, y2n = [float(value) for value in det["bbox"]]
        x1 = int(np.clip(round(x1n * width), 0, max(width - 1, 0)))
        y1 = int(np.clip(round(y1n * height), 0, max(height - 1, 0)))
        x2 = int(np.clip(round(x2n * width), 0, max(width - 1, 0)))
        y2 = int(np.clip(round(y2n * height), 0, max(height - 1, 0)))

        if show_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if isinstance(names, dict):
                cls_name = str(names.get(cls_id, cls_id))
            elif isinstance(names, list) and cls_id < len(names):
                cls_name = str(names[cls_id])
            else:
                cls_name = str(cls_id)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if show_keypoints:
            raw_kpts = list(det.get("key_points", []))
            points: list[tuple[int, int] | None] = []
            for offset in range(0, min(len(raw_kpts), 17 * 3), 3):
                x_norm = float(raw_kpts[offset])
                y_norm = float(raw_kpts[offset + 1])
                kp_conf = float(raw_kpts[offset + 2])
                if kp_conf < keypoint_conf or (x_norm == 0.0 and y_norm == 0.0):
                    points.append(None)
                    continue
                x = int(np.clip(round(x_norm * width), 0, max(width - 1, 0)))
                y = int(np.clip(round(y_norm * height), 0, max(height - 1, 0)))
                points.append((x, y))

            if show_skeleton:
                for a, b in COCO_SKELETON:
                    if a < len(points) and b < len(points) and points[a] is not None and points[b] is not None:
                        cv2.line(out, points[a], points[b], color, 2, cv2.LINE_AA)

            for point in points:
                if point is not None:
                    cv2.circle(out, point, 3, (0, 255, 0), -1, cv2.LINE_AA)

    return out


def build_payload(item: InferenceItem, detections: list[dict[str, Any]]) -> dict[str, Any]:
    # Dataset-compatible frame payload.
    # `bbox` and `key_points` inside `detection_list` are normalized values from
    # YOLO-pose. A colleague adding TMS can use this object directly only if
    # the model was trained on the same pose/JSON representation.
    return {
        "f": int(item.frame_index),
        "t": float(item.time_sec),
        "group_events": [],
        "detection_list": detections,
    }


def payload_window_span_sec(sample_window: deque[dict[str, Any]]) -> float:
    if len(sample_window) < 2:
        return 0.0
    return max(0.0, float(sample_window[-1]["t"]) - float(sample_window[0]["t"]))


def prune_payload_history(state: StreamState, latest_t: float) -> None:
    min_t = float(latest_t) - float(state.payload_history_sec)
    while state.sample_window and float(state.sample_window[0].get("t", 0.0)) < min_t:
        state.sample_window.popleft()
    state.payload_window_span_sec = payload_window_span_sec(state.sample_window)
    state.critical_low_buffer = len(state.sample_window) < int(state.payload_min_frames)


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "stream"


def write_latest_payload(output_dir: Path, state: StreamState, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_filename(state.stream_id)}.latest.json"
    document = {
        "stream_id": state.stream_id,
        "display_name": state.display_name,
        "payload_history_sec": float(state.payload_history_sec),
        "payload_min_frames": int(state.payload_min_frames),
        "payload_window_span_sec": float(state.payload_window_span_sec),
        "critical_low_buffer": bool(state.critical_low_buffer),
        "window_len": len(state.sample_window),
        "temporal_probes": state.temporal_probe_status,
        "latest_detection_count": int(state.latest_det_count),
        "latest_frame": payload,
        "window": list(state.sample_window),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def update_state_after_inference(
    item: InferenceItem,
    payload: dict[str, Any],
    display_frame: np.ndarray,
    output_dir: Path | None,
) -> None:
    # Called once per sampled stream frame after YOLO-pose finishes.
    # This is the central place where the latest pose payload becomes available
    # to temporal models. If adding another pose-based temporal model, append or
    # validate its input window here or immediately after this call in main().
    with item.state.lock:
        item.state.latest_payload = payload
        item.state.sample_window.append(payload)
        prune_payload_history(item.state, latest_t=float(payload["t"]))
        item.state.sample_count += 1
        item.state.latest_display = display_frame
        item.state.latest_det_count = len(payload["detection_list"])
        item.state.latest_infer_time = time.monotonic()
        item.state.inferred_count += 1

    if output_dir is not None:
        write_latest_payload(output_dir, item.state, payload)


# TMS online status integration: expose shared buffer health and per-probe readiness.
def status_lines(state: StreamState, sampling: SamplingConfig, temporal_probes: list[TemporalWindowSpec]) -> list[str]:
    now = time.monotonic()
    frame_age = now - state.frame_wall_time if state.frame_wall_time else 0.0
    infer_age = now - state.latest_infer_time if state.latest_infer_time else 0.0
    if sampling.resolved_mode == "time":
        sample_text = f"sample={float(sampling.sample_period_sec):.3f}s"
    else:
        sample_text = f"sample={sampling.step_frames}f"

    lines = [
        f"{state.display_name}",
        f"frame={state.frame_index} infer={state.inferred_count} det={state.latest_det_count}",
        (
            f"{sample_text} payload={len(state.sample_window)} "
            f"span={state.payload_window_span_sec:.1f}/{state.payload_history_sec:.1f}s"
        ),
        f"age={frame_age:.1f}s infer_age={infer_age:.1f}s",
        (
            f"CRITICAL low buffer <{state.payload_min_frames}/{state.payload_history_sec:.1f}s"
            if state.critical_low_buffer else
            f"buffer_ok min={state.payload_min_frames}"
        ),
        "connected" if state.connected else f"disconnected r={state.reconnect_count}",
        state.last_error[:80] if state.last_error else "",
    ]
    for spec in temporal_probes:
        lines.append(temporal_probe_status_line(state, spec))
    return lines


def draw_status(tile: np.ndarray, lines: list[str]) -> np.ndarray:
    out = tile.copy()
    clean_lines = [line for line in lines if line]
    if not clean_lines:
        return out
    line_height = 20
    pad = 8
    box_h = pad * 2 + line_height * len(clean_lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for idx, line in enumerate(clean_lines):
        y = pad + 15 + idx * line_height
        cv2.putText(out, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_dashboard(
    states: list[StreamState],
    sampling: SamplingConfig,
    temporal_probes: list[TemporalWindowSpec],
    display_cfg: dict[str, Any],
) -> np.ndarray:
    tile_w = int(display_cfg.get("tile_width", 480))
    tile_h = int(display_cfg.get("tile_height", 270))
    columns = max(int(display_cfg.get("columns", 2)), 1)
    show_status = bool(display_cfg.get("show_status", True))

    tiles: list[np.ndarray] = []
    for state_index, state in enumerate(states):
        row_idx = state_index // columns
        col_idx = state_index % columns
        with state.lock:
            state.dashboard_position = (row_idx, col_idx)
            frame = state.latest_display.copy() if state.latest_display is not None else None
            if frame is None and state.frame is not None:
                frame = state.frame.copy()
            lines = status_lines(state, sampling, temporal_probes)

        if frame is None:
            frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        if show_status:
            tile = draw_status(tile, lines)
        tiles.append(tile)

    while len(tiles) > columns and len(tiles) % columns != 0:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    rows = []
    for start in range(0, len(tiles), columns):
        rows.append(cv2.hconcat(tiles[start:start + columns]))
    dashboard = cv2.vconcat(rows)

    max_width = int(display_cfg.get("max_width", 0) or 0)
    max_height = int(display_cfg.get("max_height", 0) or 0)
    scale = 1.0
    if max_width > 0 or max_height > 0:
        height, width = dashboard.shape[:2]
        if max_width > 0:
            scale = min(scale, float(max_width) / max(float(width), 1.0))
        if max_height > 0:
            scale = min(scale, float(max_height) / max(float(height), 1.0))
        if scale < 1.0:
            dashboard = cv2.resize(
                dashboard,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
    return dashboard


def print_display_layout_once(
    states: list[StreamState],
    display_cfg: dict[str, Any],
    dashboard: np.ndarray,
    printed: bool,
) -> bool:
    if printed:
        return printed
    columns = max(int(display_cfg.get("columns", 2)), 1)
    tile_w = int(display_cfg.get("tile_width", 480))
    tile_h = int(display_cfg.get("tile_height", 270))
    max_width = int(display_cfg.get("max_width", 0) or 0)
    max_height = int(display_cfg.get("max_height", 0) or 0)
    full_width = tile_w * columns
    full_height = tile_h * max(1, int(np.ceil(len(states) / float(columns))))
    scale = 1.0
    if max_width > 0:
        scale = min(scale, float(max_width) / max(float(full_width), 1.0))
    if max_height > 0:
        scale = min(scale, float(max_height) / max(float(full_height), 1.0))
    effective_tile_w = max(1, int(round(tile_w * scale)))
    effective_tile_h = max(1, int(round(tile_h * scale)))
    positions = []
    for idx, state in enumerate(states):
        row_idx = idx // columns
        col_idx = idx % columns
        positions.append(f"{state.stream_id}=row{row_idx + 1}/col{col_idx + 1}")
    print(
        "[INFO] dashboard_shape="
        f"{dashboard.shape[1]}x{dashboard.shape[0]} configured_tile={tile_w}x{tile_h} "
        f"effective_tile={effective_tile_w}x{effective_tile_h} columns={columns} "
        + " ".join(positions),
        flush=True,
    )
    return True


def maybe_print_status(
    states: list[StreamState],
    sampling: SamplingConfig,
    temporal_probes: list[TemporalWindowSpec],
    interval_sec: float,
    last_print: float,
) -> float:
    if interval_sec <= 0:
        return last_print
    now = time.monotonic()
    if now - last_print < interval_sec:
        return last_print

    parts = []
    for state in states:
        with state.lock:
            error_part = f" err={state.last_error[:80]}" if state.last_error else ""
            critical_part = " CRIT_LOW_BUFFER" if state.critical_low_buffer else ""
            probe_part = " ".join(temporal_probe_status_line(state, spec) for spec in temporal_probes)
            parts.append(
                f"{state.stream_id}: frame={state.frame_index} infer={state.inferred_count} "
                f"det={state.latest_det_count} window={len(state.sample_window)} "
                f"span={state.payload_window_span_sec:.1f}/{state.payload_history_sec:.1f}s "
                f"connected={state.connected}{critical_part}{error_part} {probe_part}".strip()
            )
    print("[STATUS] " + " | ".join(parts), flush=True)
    return now


def close_display_windows(display_enabled: bool, window_name: str) -> None:
    if not display_enabled:
        return

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
    cv2.destroyAllWindows()

    for _ in range(5):
        cv2.waitKey(1)
        time.sleep(0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multistream RTSP/video YOLO-pose inference with rolling payload windows.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV dashboard even if config enables it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    sampling = resolve_sampling(config)
    payload_history_sec, payload_min_frames = resolve_payload_buffer_policy(config)
    temporal_probes = resolve_temporal_probes(config, first_present=first_present)
    states = enabled_streams(config,
                             config_path.parent,
                             payload_history_sec=payload_history_sec,
                             payload_min_frames=payload_min_frames)
    enabled_ids, disabled_ids = stream_ids_by_enabled_state(config)

    model_cfg = dict(config.get("model", {}) or {})
    yolo_path = model_cfg.get("yolo_pose")
    if not yolo_path:
        raise ValueError("model.yolo_pose is required")
    yolo_path = Path(yolo_path)
    if not yolo_path.is_absolute():
        yolo_path = (ROOT / yolo_path).resolve()

    runtime_cfg = dict(config.get("runtime", {}) or {})
    requested_device = str(runtime_cfg.get("device", "cuda:0"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU.", flush=True)
        requested_device = "cpu"
    device = torch.device(requested_device)
    half = bool(model_cfg.get("half", True)) and device.type == "cuda"
    conf = float(model_cfg.get("conf", 0.5))
    imgsz = normalize_imgsz(model_cfg.get("imgsz", 736))
    ignored_classes = {int(value) for value in model_cfg.get("ignored_classes", [3, 4])}
    max_batch_size = max(int(runtime_cfg.get("max_batch_size", len(states))), 1)
    poll_sleep_sec = float(runtime_cfg.get("poll_sleep_sec", 0.005))
    status_interval_sec = float(runtime_cfg.get("status_interval_sec", 5.0))

    output_cfg = dict(config.get("output", {}) or {})
    output_dir_value = output_cfg.get("latest_payload_dir")
    output_dir = Path(output_dir_value).resolve() if output_dir_value else None

    capture_cfg = dict(config.get("capture", {}) or {})
    backend = opencv_backend(str(capture_cfg.get("backend", "ffmpeg")))
    reconnect_delay_sec = float(capture_cfg.get("reconnect_delay_sec", 2.0))
    read_sleep_sec = float(capture_cfg.get("read_sleep_sec", 0.0))
    loop_files = bool(capture_cfg.get("loop_files", True))
    play_in_realtime = bool(capture_cfg.get("play_in_realtime", False))

    display_cfg = dict(config.get("display", {}) or {})
    display_enabled = bool(display_cfg.get("enabled", True)) and not bool(args.no_display)
    display_interval_sec = 1.0 / max(float(display_cfg.get("fps", 10.0)), 0.1)
    window_name = str(display_cfg.get("window_name", "rtsp_multistream"))
    show_boxes = bool(display_cfg.get("show_boxes", True))
    show_keypoints = bool(display_cfg.get("show_keypoints", True))
    show_skeleton = bool(display_cfg.get("show_skeleton", True))
    keypoint_conf = float(display_cfg.get("keypoint_conf", 0.25))

    print("[INFO] "
          f"streams={len(states)} device={device} half={half} imgsz={imgsz} "
          f"sampling={sampling.resolved_mode} step_frames={sampling.step_frames} "
          f"sample_period_sec={sampling.sample_period_sec} payload_history_sec={states[0].payload_history_sec} "
          f"payload_min_frames={states[0].payload_min_frames}",
          flush=True,)
    print("[INFO] temporal_probes="
          + ", ".join(
              f"{spec.name}(window={spec.window_sec:.2f}s every={spec.infer_every_sec:.2f}s "
              f"min_frames={spec.min_frames} tol={spec.tolerance_sec:.3f}s)"
              for spec in temporal_probes
          ),
          flush=True)
    print(f"[INFO] capture_backend={str(capture_cfg.get('backend', 'ffmpeg'))} "
          f"loop_files={loop_files} play_in_realtime={play_in_realtime}",
          flush=True)
    print(f"[INFO] enabled_streams={enabled_ids}", flush=True)
    if disabled_ids:
        print(f"[INFO] disabled_streams={disabled_ids}", flush=True)
    if display_enabled:
        print( "[INFO] display="
               f"columns={int(display_cfg.get('columns', 2))} "
               f"tile={int(display_cfg.get('tile_width', 480))}x{int(display_cfg.get('tile_height', 270))} "
               f"max={int(display_cfg.get('max_width', 0) or 0)}x{int(display_cfg.get('max_height', 0) or 0)}",
               flush=True,)

    yolo = YOLO(str(yolo_path))
    yolo.to(device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    stop_event = threading.Event()

    def request_shutdown(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        print(f"\n[INFO] received {signal_name}; shutting down cleanly.", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, request_shutdown)

    threads = [threading.Thread( target=stream_reader,
                                 args=(state, stop_event, backend, reconnect_delay_sec, read_sleep_sec,
                                       loop_files, play_in_realtime),
                                 name=f"reader:{state.stream_id}",
                                 daemon=True,)
               for state in states]
    for thread in threads:
        thread.start()

    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_display_time = 0.0
    last_status_print = 0.0
    display_layout_printed = False
    display_window_resized = False

    try:
        while not stop_event.is_set():
            batch: list[InferenceItem] = []
            for state in states:
                item = should_sample_state(state, sampling)
                if item is not None:
                    batch.append(item)
                    if len(batch) >= max_batch_size:
                        break

            if batch:
                # Batched YOLO-pose over the streams that are ready now.
                # This batch is intentionally variable-size; a missing/slow
                # stream does not block the other cameras.
                frames = [item.frame for item in batch]
                results = yolo(frames, conf=conf, verbose=False, imgsz=imgsz, half=half)
                if not isinstance(results, list):
                    results = [results]

                for item, result in zip(batch, results):
                    detections = result_to_detection_list(result, conf_thresh=conf, ignored_classes=ignored_classes)
                    payload = build_payload(item, detections)
                    # Per sampled stream we now have two useful data surfaces:
                    # - `item.frame`: raw sampled BGR image before overlays. Use
                    #   it for RGB clip models such as TMS.
                    # - `payload`: normalized boxes/keypoints in the same frame
                    #   schema used by our JSON training data. Use it for
                    #   pose/metadata models trained on the same representation.
                    display_frame = draw_pose_overlay(item.frame,
                                                      detections,
                                                      names=getattr(yolo.model, "names", None),
                                                      show_boxes=show_boxes,
                                                      show_keypoints=show_keypoints,
                                                      show_skeleton=show_skeleton,
                                                      keypoint_conf=keypoint_conf,
                                                      )
                    update_state_after_inference(item,
                                                 payload,
                                                 display_frame,
                                                 output_dir=output_dir,
                                                 )
                    # TMS online probe hook: evaluate time-window readiness from
                    # the shared payload history without running the model yet.
                    with item.state.lock:
                        for spec in temporal_probes:
                            collect_temporal_probe_window(item.state, spec, latest_t=float(payload["t"]))
                    # `sample_window` is the rolling pose-payload history. A TMS
                    # RGB model should not use this deque directly unless it was
                    # trained on pose payloads; add a separate raw-frame deque
                    # if it needs clips.
                    # TODO: Add TMS-ready temporal batching here, using the
                    # rolling payload window stored in item.state.sample_window.

            now = time.monotonic()
            if display_enabled and now - last_display_time >= display_interval_sec:
                dashboard = make_dashboard(states, sampling, temporal_probes, display_cfg)
                display_layout_printed = print_display_layout_once(states, display_cfg, dashboard, display_layout_printed)
                if not display_window_resized:
                    cv2.resizeWindow(window_name, dashboard.shape[1], dashboard.shape[0])
                    display_window_resized = True
                cv2.imshow(window_name, dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    stop_event.set()
                last_display_time = now

            last_status_print = maybe_print_status(states, sampling, temporal_probes, status_interval_sec, last_status_print)
            time.sleep(poll_sleep_sec)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        close_display_windows(display_enabled, window_name)
        for thread in threads:
            thread.join(timeout=2.0)
        close_display_windows(display_enabled, window_name)

# 1457(14!,1,,10) 1099
if __name__ == "__main__":
    main()
