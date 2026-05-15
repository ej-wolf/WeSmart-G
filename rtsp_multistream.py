#!/usr/bin/env python3
"""Run YOLO-pose and optional TCN inference on many live/video streams.

This is step 1 of the multistream pipeline:

RTSP/video readers -> latest frame per stream -> batched YOLO-pose ->
TCN-compatible detection payload windows -> optional batched TCN -> dashboard.

Notes for adding another temporal model, for example a TSM model:
- Do not put model inference inside `stream_reader()`. Reader threads should
  only keep the newest decoded frame so one slow model cannot block RTSP input.
- The sampled raw image is `InferenceItem.frame`. It is an OpenCV BGR frame
  copied before drawing overlays. RGB/video models should take their frame
  window from this point, converting BGR -> RGB if needed.
- The pose/detection data shared with our TCN is the `payload` created by
  `build_payload()`. It matches the JSON-like dataset frame schema used in
  training: `f`, `t`, `group_events`, and `detection_list`.
- `StreamState.sample_window` stores only these pose payloads, not raw images.
  If a TSM model needs RGB frame clips, add a separate per-stream deque for
  sampled raw frames alongside `sample_window`.
- `run_tcn_candidates()` is the reference pattern for batched temporal model
  inference across streams. A TSM implementation should follow the same shape:
  collect ready per-stream windows, stack them into one batch, run the model
  once, then write the result back to each `StreamState`.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_video_hysteresis import (  # noqa: E402
    HystCfg,
    Hysteresis,
    MOTION_ONLY_MODEL_TYPES,
    load_model_from_payload,
    resolve_model_artifacts,
)
from src.data.features import (  # noqa: E402
    EREZ_BASE_MOTION_DIM,
    extract_erez_motion_features,
    frame_to_vector,
    motion_extractor_kwargs,
    motion_feature_cfg,
    motion_feature_requires_extended,
    select_motion_features,
)


COCO_SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass
class SamplingConfig:
    requested_mode: str
    resolved_mode: str
    step_frames: int
    sample_period_sec: float | None
    training_sample_period_sec: float | None


@dataclass
class TcnRuntime:
    enabled: bool
    model: torch.nn.Module | None = None
    device: torch.device | None = None
    model_type: str = ""
    feature_cfg: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    run_dir: Path | None = None
    window_size: int = 0
    required_payload_window_size: int = 0
    stride: int = 1
    threshold: float = 0.5
    use_hysteresis: bool = True
    hyst_cfg: HystCfg | None = None
    smooth_mode: str = "off"
    ema_beta: float = 0.8
    mean_window: int = 5
    expected_input_dim: int | None = None


@dataclass
class InferenceItem:
    # One sampled frame selected from a stream for expensive inference.
    # `frame` is raw BGR image data before keypoint/box drawing. Use this for
    # RGB/video models such as TSM, not `latest_display`, which has overlays.
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
    sample_window_size: int
    tcn_window_size: int
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
    latest_vector_dim: int | None = None
    dashboard_position: tuple[int, int] | None = None
    sample_count: int = 0
    next_tcn_sample_count: int = field(init=False)
    latest_tcn_prob: float | None = None
    latest_tcn_smooth: float | None = None
    latest_tcn_state: bool = False
    latest_tcn_over_threshold: bool = False
    tcn_enabled: bool = False
    tcn_threshold: float = 0.5
    latest_tcn_infer_count: int = 0
    latest_tcn_time: float = 0.0
    latest_tcn_input_shape: tuple[int, ...] | None = None
    tcn_error: str = ""
    tcn_wait_reason: str = ""
    tcn_hyst: Hysteresis | None = field(init=False, default=None)
    tcn_smooth_buf: deque[float] = field(init=False)
    # Rolling window of YOLO-pose payloads in the same JSON-like format used by
    # the TCN training data. This is pose metadata, not image pixels.
    sample_window: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.sample_window = deque(maxlen=int(self.sample_window_size))
        self.next_tcn_sample_count = int(self.sample_window_size)
        self.tcn_smooth_buf = deque()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs. Install pyyaml or run in the project environment.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def deep_get(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def load_training_config(config: dict[str, Any]) -> dict[str, Any]:
    path_value = deep_get(config, ("tcn", "training_config"))
    if not path_value:
        run_dir_value = deep_get(config, ("tcn", "run_dir"))
        if not run_dir_value:
            return {}
        run_dir = Path(run_dir_value)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        candidate = run_dir / "config_resolved.yaml"
        if not candidate.exists():
            return {}
        path_value = str(candidate)
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"tcn.training_config does not exist: {path}")
    return load_yaml(path)


def resolve_tcn_window_size(config: dict[str, Any], training_cfg: dict[str, Any]) -> int:
    window_size = int(first_present(
        deep_get(config, ("tcn", "window_size")),
        deep_get(config, ("tcn", "window")),
        deep_get(training_cfg, ("data", "window_size")),
        default=8,
    ))
    return max(window_size, 1)


def resolve_tcn_model_type_hint(config: dict[str, Any], training_cfg: dict[str, Any]) -> str:
    return str(first_present(
        deep_get(config, ("tcn", "model_type")),
        deep_get(training_cfg, ("model", "type")),
        default="tcn",
    )).lower()


def required_payload_window_size(model_type: str, tcn_window_size: int) -> int:
    if str(model_type).lower() in MOTION_ONLY_MODEL_TYPES:
        return int(tcn_window_size) + 1
    return int(tcn_window_size)


def tcn_cfg_enabled(config: dict[str, Any]) -> bool:
    tcn_cfg = dict(config.get("tcn", {}) or {})
    default_enabled = bool(tcn_cfg.get("run_dir") or tcn_cfg.get("checkpoint") or tcn_cfg.get("ckpt"))
    return bool(tcn_cfg.get("enabled", default_enabled))


def tcn_uses_motion_features(model_type: str, feature_cfg: dict[str, Any]) -> bool:
    return str(model_type).lower() in MOTION_ONLY_MODEL_TYPES or bool(motion_feature_cfg(feature_cfg).get("enabled", False))


def required_payload_window_size_for_tcn(model_type: str, feature_cfg: dict[str, Any], tcn_window_size: int) -> int:
    if tcn_uses_motion_features(model_type, feature_cfg):
        return int(tcn_window_size) + 1
    return int(tcn_window_size)


def resolve_sampling(config: dict[str, Any], training_cfg: dict[str, Any]) -> SamplingConfig:
    sampling_cfg = dict(config.get("sampling", {}) or {})
    requested_mode = str(sampling_cfg.get("mode", "auto")).strip().lower()
    if requested_mode not in {"auto", "frames", "time"}:
        raise ValueError("sampling.mode must be one of: auto, frames, time")

    train_fps = deep_get(training_cfg, ("data", "fps"))
    train_step = deep_get(training_cfg, ("data", "sample_every_n_frames"))
    training_sample_period_sec = None
    if train_fps and train_step:
        training_sample_period_sec = float(train_step) / float(train_fps)

    step_frames = int(first_present(
        sampling_cfg.get("step_frames"),
        sampling_cfg.get("step"),
        train_step,
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
        resolved_period = sample_period_sec or training_sample_period_sec
        if resolved_period is None:
            raise ValueError("sampling.mode=time requires sampling.sample_period_sec or tcn.training_config data fps/step")
        return SamplingConfig(requested_mode, "time", step_frames, float(resolved_period), training_sample_period_sec)

    if requested_mode == "frames":
        return SamplingConfig(requested_mode, "frames", step_frames, sample_period_sec, training_sample_period_sec)

    if sample_period_sec is not None or training_sample_period_sec is not None:
        return SamplingConfig(
            requested_mode,
            "time",
            step_frames,
            float(sample_period_sec or training_sample_period_sec),
            training_sample_period_sec,
        )
    return SamplingConfig(requested_mode, "frames", step_frames, None, training_sample_period_sec)


def normalize_imgsz(value: Any) -> int | tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("model.imgsz list/tuple must contain [height, width]")
        return int(value[0]), int(value[1])
    return int(value)


def enabled_streams(
    config: dict[str, Any],
    sample_window_size: int,
    tcn_window_size: int,
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
        states.append(
            StreamState(
                stream_id=stream_id,
                url=str(url),
                display_name=str(item.get("display_name") or stream_id),
                sample_window_size=int(sample_window_size),
                tcn_window_size=int(tcn_window_size),
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

        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    with state.lock:
                        state.connected = False
                        state.last_error = "read failed"
                        state.reconnect_count += 1
                    break

                now = time.monotonic()
                frame_index += 1
                with state.lock:
                    if state.first_frame_wall_time is None:
                        state.first_frame_wall_time = now
                    state.frame = frame
                    state.frame_index = frame_index
                    state.frame_wall_time = now
                    state.frame_time_sec = now - float(state.first_frame_wall_time)
                    state.connected = True
                    state.last_error = ""

                if read_sleep_sec > 0:
                    time.sleep(read_sleep_sec)
        finally:
            cap.release()

        if not stop_event.is_set():
            time.sleep(reconnect_delay_sec)


def should_sample_state(state: StreamState, sampling: SamplingConfig) -> InferenceItem | None:
    # This is the gate between "camera FPS" and "model FPS".
    # If sampling.sample_period_sec=0.2, only one copied frame per stream every
    # ~0.2 seconds enters YOLO/TCN/possible TSM processing.
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
    # YOLO-pose. TCN uses this payload through `frame_to_vector()` and motion
    # feature extraction. A colleague adding TSM can also use this object if the
    # TSM model was trained on the same pose/JSON representation.
    return {
        "f": int(item.frame_index),
        "t": float(item.time_sec),
        "group_events": [],
        "detection_list": detections,
    }


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "stream"


def write_latest_payload(
    output_dir: Path,
    state: StreamState,
    payload: dict[str, Any],
    max_persons: int,
    feature_vector_dim: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_filename(state.stream_id)}.latest.json"
    document = {
        "stream_id": state.stream_id,
        "display_name": state.display_name,
        "max_persons": int(max_persons),
        "window_size": int(state.sample_window.maxlen or 0),
        "window_ready": len(state.sample_window) == (state.sample_window.maxlen or 0),
        "window_len": len(state.sample_window),
        "feature_vector_dim": feature_vector_dim,
        "tcn": {
            "prob": state.latest_tcn_prob,
            "prob_smooth": state.latest_tcn_smooth,
            "state": int(state.latest_tcn_state),
            "infer_count": int(state.latest_tcn_infer_count),
            "input_shape": list(state.latest_tcn_input_shape) if state.latest_tcn_input_shape is not None else None,
            "error": state.tcn_error,
        },
        "latest_frame": payload,
        "window": list(state.sample_window),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def update_state_after_inference(
    item: InferenceItem,
    payload: dict[str, Any],
    display_frame: np.ndarray,
    max_persons: int,
    feature_cfg: dict[str, Any],
    output_dir: Path | None,
) -> None:
    # Called once per sampled stream frame after YOLO-pose finishes.
    # This is the central place where the latest pose payload becomes available
    # to temporal models. If adding another pose-based temporal model, append or
    # validate its input window here or immediately after this call in main().
    vector_dim: int | None = None
    last_error = ""
    try:
        vector = frame_to_vector(payload, K=max_persons, cfg=feature_cfg)
        vector_dim = int(vector.shape[0])
    except Exception as exc:  # Keep running; expose the issue in the dashboard.
        last_error = f"TCN format error: {exc}"

    with item.state.lock:
        item.state.latest_payload = payload
        item.state.sample_window.append(payload)
        item.state.sample_count += 1
        item.state.latest_display = display_frame
        item.state.latest_det_count = len(payload["detection_list"])
        item.state.latest_infer_time = time.monotonic()
        item.state.inferred_count += 1
        item.state.latest_vector_dim = vector_dim
        if last_error:
            item.state.last_error = last_error

    if output_dir is not None:
        write_latest_payload(output_dir, item.state, payload, max_persons=max_persons, feature_vector_dim=vector_dim)


def resolve_tcn_checkpoint_path(tcn_cfg: dict[str, Any]) -> tuple[Path | None, Path | None]:
    run_dir_value = tcn_cfg.get("run_dir")
    run_dir = Path(run_dir_value).resolve() if run_dir_value else None

    checkpoint_value = first_present(tcn_cfg.get("checkpoint"), tcn_cfg.get("ckpt"), default=None)
    if checkpoint_value is None:
        return run_dir, None

    checkpoint = Path(checkpoint_value)
    if checkpoint.is_absolute():
        return run_dir, checkpoint

    candidates = []
    if run_dir is not None:
        candidates.append(run_dir / checkpoint)
    candidates.append(ROOT / checkpoint)
    candidates.append(checkpoint.resolve())

    for candidate in candidates:
        if candidate.exists():
            return run_dir, candidate.resolve()
    return run_dir, candidates[0].resolve()


def load_tcn_runtime(
    config: dict[str, Any],
    device: torch.device,
    max_persons: int,
    fallback_window_size: int,
) -> TcnRuntime:
    tcn_cfg = dict(config.get("tcn", {}) or {})
    if not tcn_cfg_enabled(config):
        return TcnRuntime(enabled=False)

    run_dir, checkpoint = resolve_tcn_checkpoint_path(tcn_cfg)
    args = SimpleNamespace(
        run_dir=str(run_dir) if run_dir is not None else None,
        ckpt=str(checkpoint) if checkpoint is not None else None,
        K=int(max_persons),
    )
    ckpt_path, payload, model_cfg = resolve_model_artifacts(args)
    model, model_type, feature_cfg = load_model_from_payload(payload, model_cfg, args, device=device)

    window_size = int(first_present(
        tcn_cfg.get("window_size"),
        tcn_cfg.get("window"),
        deep_get(model_cfg, ("data", "window_size")),
        fallback_window_size,
    ))
    window_size = max(window_size, 1)
    stride = max(int(tcn_cfg.get("stride", 1)), 1)
    required_payloads = required_payload_window_size_for_tcn(model_type, feature_cfg, window_size)

    threshold = float(first_present(tcn_cfg.get("threshold"), tcn_cfg.get("thr_on"), default=0.5))
    use_hysteresis = bool(tcn_cfg.get("use_hysteresis", tcn_cfg.get("hysteresis", False)))
    thr_on = float(tcn_cfg.get("thr_on", threshold))
    thr_off = float(tcn_cfg.get("thr_off", min(threshold, thr_on)))
    hyst_cfg = HystCfg(
        thr_on=thr_on,
        thr_off=thr_off,
        k_on=int(tcn_cfg.get("k_on", 1)),
        k_off=int(tcn_cfg.get("k_off", 1)),
        N=int(tcn_cfg.get("N", 1)),
    )
    smooth_mode = str(tcn_cfg.get("smooth_mode", "off")).lower()
    if smooth_mode not in {"off", "ema", "mean"}:
        raise ValueError("tcn.smooth_mode must be one of: off, ema, mean")

    return TcnRuntime(
        enabled=True,
        model=model,
        device=device,
        model_type=str(model_type),
        feature_cfg=feature_cfg,
        checkpoint_path=ckpt_path,
        run_dir=run_dir,
        window_size=window_size,
        required_payload_window_size=required_payloads,
        stride=stride,
        threshold=threshold,
        use_hysteresis=use_hysteresis,
        hyst_cfg=hyst_cfg,
        smooth_mode=smooth_mode,
        ema_beta=float(tcn_cfg.get("ema_beta", 0.8)),
        mean_window=max(int(tcn_cfg.get("mean_window", 5)), 1),
        expected_input_dim=int(getattr(model, "input_dim", 0)) or None,
    )


def configure_stream_tcn_state(states: list[StreamState], runtime: TcnRuntime) -> None:
    for state in states:
        with state.lock:
            state.tcn_enabled = bool(runtime.enabled)
            if not runtime.enabled:
                state.tcn_wait_reason = "disabled"
                continue
            old_items = list(state.sample_window)
            state.sample_window_size = int(runtime.required_payload_window_size)
            state.tcn_window_size = int(runtime.window_size)
            state.sample_window = deque(old_items[-state.sample_window_size:], maxlen=state.sample_window_size)
            state.next_tcn_sample_count = max(int(runtime.required_payload_window_size), 1)
            state.tcn_hyst = Hysteresis(runtime.hyst_cfg) if runtime.use_hysteresis and runtime.hyst_cfg is not None else None
            state.tcn_smooth_buf = deque(maxlen=max(int(runtime.mean_window), 1))
            state.tcn_threshold = float(runtime.threshold)
            state.tcn_wait_reason = f"waiting {len(state.sample_window)}/{runtime.required_payload_window_size}"


def fit_feature_dim(values: np.ndarray, target_dim: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if target_dim is None:
        return values
    target_dim = int(target_dim)
    if values.shape[-1] == target_dim:
        return values.astype(np.float32, copy=False)
    if values.shape[-1] > target_dim:
        return values[..., :target_dim].astype(np.float32, copy=False)
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, target_dim - values.shape[-1])
    return np.pad(values, pad_width, mode="constant").astype(np.float32, copy=False)


def build_motion_features_for_frames(
    frames: list[dict[str, Any]],
    runtime: TcnRuntime,
) -> np.ndarray:
    motion_cfg = motion_feature_cfg(runtime.feature_cfg)
    extended = (
        motion_feature_requires_extended(runtime.feature_cfg)
        if motion_cfg.get("enabled", False)
        else bool(runtime.expected_input_dim and runtime.expected_input_dim > EREZ_BASE_MOTION_DIM)
    )
    motion_seq = extract_erez_motion_features(
        frames,
        align=motion_cfg.get("align", "prev"),
        j_version=runtime.feature_cfg.get("erez_json_version", 2.0),
        extended=extended,
        **motion_extractor_kwargs(runtime.feature_cfg),
    )
    if motion_cfg.get("enabled", False):
        return select_motion_features(motion_seq, feature_cfg=runtime.feature_cfg)
    return fit_feature_dim(motion_seq, runtime.expected_input_dim)


def build_tcn_input_window(
    frames: list[dict[str, Any]],
    runtime: TcnRuntime,
    max_persons: int,
) -> np.ndarray:
    # Convert a list of dataset-compatible payloads into the exact tensor shape
    # expected by the loaded TCN checkpoint. This function is TCN-specific; a TSM
    # model should have its own equivalent builder, likely from raw RGB frame
    # clips or from the same payload list depending on how it was trained.
    if not runtime.enabled:
        raise RuntimeError("TCN runtime is disabled")

    window_size = int(runtime.window_size)
    if len(frames) < int(runtime.required_payload_window_size):
        raise ValueError(
            f"Need {runtime.required_payload_window_size} payloads for TCN, got {len(frames)}"
        )

    if runtime.model_type in MOTION_ONLY_MODEL_TYPES:
        motion_seq = build_motion_features_for_frames(frames[-(window_size + 1):], runtime)
        return motion_seq[-window_size:].astype(np.float32, copy=False)

    tcn_frames = frames[-window_size:]
    static_seq = np.stack(
        [frame_to_vector(frame, K=max_persons, cfg=runtime.feature_cfg) for frame in tcn_frames],
        axis=0,
    ).astype(np.float32, copy=False)

    if motion_feature_cfg(runtime.feature_cfg).get("enabled", False):
        motion_source_frames = frames[-(window_size + 1):]
        motion_seq = build_motion_features_for_frames(motion_source_frames, runtime)[-window_size:]
        static_seq = np.concatenate([static_seq, motion_seq], axis=-1).astype(np.float32, copy=False)

    return static_seq


def collect_tcn_window_if_ready(state: StreamState, runtime: TcnRuntime) -> tuple[list[dict[str, Any]], float | None] | None:
    if not runtime.enabled:
        return None
    with state.lock:
        if len(state.sample_window) < int(runtime.required_payload_window_size):
            state.tcn_wait_reason = f"waiting {len(state.sample_window)}/{runtime.required_payload_window_size}"
            return None
        if state.sample_count < int(state.next_tcn_sample_count):
            state.tcn_wait_reason = f"stride wait {state.sample_count}/{state.next_tcn_sample_count}"
            return None
        state.next_tcn_sample_count = int(state.sample_count) + int(runtime.stride)
        frames = list(state.sample_window)[-int(runtime.required_payload_window_size):]
        aspect_ratio = None
        if state.frame is not None:
            h, w = state.frame.shape[:2]
            if h > 0:
                aspect_ratio = float(w) / float(h)
        state.tcn_wait_reason = "ready"
        return frames, aspect_ratio


@torch.inference_mode()
def predict_tcn_probs_batch(runtime: TcnRuntime, windows: list[np.ndarray], aspect_ratios: list[float | None]) -> np.ndarray:
    if runtime.model is None or runtime.device is None:
        raise RuntimeError("TCN runtime is not loaded")
    batch_np = np.stack(windows, axis=0).astype(np.float32, copy=False)
    x = torch.from_numpy(batch_np).to(runtime.device)
    aspect_tensor = None
    if any(value is not None for value in aspect_ratios):
        aspect_tensor = torch.tensor(
            [1.0 if value is None else float(value) for value in aspect_ratios],
            dtype=x.dtype,
            device=x.device,
        )
    logits = runtime.model(x, frame_aspect_ratio=aspect_tensor)
    return torch.sigmoid(logits[:, -1]).detach().cpu().numpy().astype(np.float32)


def update_tcn_result(state: StreamState, raw_prob: float, runtime: TcnRuntime, input_shape: tuple[int, ...]) -> None:
    with state.lock:
        if runtime.smooth_mode == "off":
            smooth_prob = float(raw_prob)
        elif runtime.smooth_mode == "ema":
            if state.latest_tcn_smooth is None:
                smooth_prob = float(raw_prob)
            else:
                smooth_prob = float(runtime.ema_beta) * float(state.latest_tcn_smooth) + (1.0 - float(runtime.ema_beta)) * float(raw_prob)
        elif runtime.smooth_mode == "mean":
            state.tcn_smooth_buf.append(float(raw_prob))
            smooth_prob = float(np.mean(np.asarray(state.tcn_smooth_buf, dtype=np.float32)))
        else:
            raise ValueError(f"Unsupported smooth_mode: {runtime.smooth_mode}")

        if runtime.use_hysteresis:
            if state.tcn_hyst is None and runtime.hyst_cfg is not None:
                state.tcn_hyst = Hysteresis(runtime.hyst_cfg)
            tcn_state = bool(state.tcn_hyst.update(smooth_prob)) if state.tcn_hyst is not None else bool(smooth_prob >= runtime.threshold)
        else:
            tcn_state = bool(smooth_prob >= runtime.threshold)

        state.latest_tcn_prob = float(raw_prob)
        state.latest_tcn_smooth = float(smooth_prob)
        state.latest_tcn_state = tcn_state
        state.latest_tcn_over_threshold = bool(smooth_prob >= runtime.threshold)
        state.tcn_threshold = float(runtime.threshold)
        state.latest_tcn_infer_count += 1
        state.latest_tcn_time = time.monotonic()
        state.latest_tcn_input_shape = tuple(int(value) for value in input_shape)
        state.tcn_error = ""
        state.tcn_wait_reason = ""


def update_tcn_error(state: StreamState, message: str) -> None:
    with state.lock:
        state.tcn_error = message[:160]
        state.tcn_wait_reason = "error"


def run_tcn_candidates(
    candidates: list[tuple[StreamState, list[dict[str, Any]], float | None]],
    runtime: TcnRuntime,
    max_persons: int,
) -> None:
    # Reference pattern for any temporal model in this script:
    # 1. collect ready per-camera windows,
    # 2. convert each window to the model input format,
    # 3. stack windows into one batch,
    # 4. run one model call,
    # 5. write each prediction back to its owning StreamState.
    # A TSM model can follow this same function structure with a separate
    # runtime dataclass and input-window builder.
    if not candidates or not runtime.enabled:
        return

    windows: list[np.ndarray] = []
    states: list[StreamState] = []
    aspect_ratios: list[float | None] = []
    for state, frames, aspect_ratio in candidates:
        try:
            window = build_tcn_input_window(frames, runtime=runtime, max_persons=max_persons)
        except Exception as exc:
            update_tcn_error(state, f"TCN feature error: {exc}")
            continue
        windows.append(window)
        states.append(state)
        aspect_ratios.append(aspect_ratio)

    if not windows:
        return

    try:
        probs = predict_tcn_probs_batch(runtime, windows, aspect_ratios)
    except Exception as exc:
        for state in states:
            update_tcn_error(state, f"TCN inference error: {exc}")
        return

    for state, prob, window in zip(states, probs, windows):
        update_tcn_result(state, float(prob), runtime, tuple(window.shape))


def status_lines(state: StreamState, sampling: SamplingConfig) -> list[str]:
    now = time.monotonic()
    frame_age = now - state.frame_wall_time if state.frame_wall_time else 0.0
    infer_age = now - state.latest_infer_time if state.latest_infer_time else 0.0
    if sampling.resolved_mode == "time":
        sample_text = f"sample={float(sampling.sample_period_sec):.3f}s"
    else:
        sample_text = f"sample={sampling.step_frames}f"

    if not state.tcn_enabled:
        tcn_text = "tcn=disabled"
    elif state.latest_tcn_prob is None:
        reason = f" {state.tcn_wait_reason}" if state.tcn_wait_reason else ""
        tcn_text = f"tcn=p...{reason}"
    else:
        state_label = "ON" if state.latest_tcn_state else "OFF"
        if state.latest_tcn_smooth is not None and abs(state.latest_tcn_smooth - state.latest_tcn_prob) > 1e-6:
            tcn_text = f"tcn={state_label} p={state.latest_tcn_prob:.3f} ps={state.latest_tcn_smooth:.3f}"
        else:
            tcn_text = f"tcn={state_label} p={state.latest_tcn_prob:.3f}"

    tcn_error = state.tcn_error[:80] if state.tcn_error else ""
    return [
        f"{state.display_name}",
        f"frame={state.frame_index} infer={state.inferred_count} det={state.latest_det_count}",
        f"{sample_text} payload={len(state.sample_window)}/{state.sample_window.maxlen} tcn_win={state.tcn_window_size}",
        f"{tcn_text} n={state.latest_tcn_infer_count}",
        f"age={frame_age:.1f}s infer_age={infer_age:.1f}s",
        "connected" if state.connected else f"disconnected r={state.reconnect_count}",
        tcn_error,
        state.last_error[:80] if state.last_error else "",
    ]


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


def draw_tcn_indicator(
    tile: np.ndarray,
    prob: float | None,
    smooth_prob: float | None,
    over_threshold: bool,
    threshold: float,
) -> np.ndarray:
    if prob is None:
        return tile

    out = tile.copy()
    color = (0, 0, 255) if over_threshold else (0, 180, 0)
    label = "TCN ALERT" if over_threshold else "TCN OK"
    score = float(smooth_prob if smooth_prob is not None else prob)
    text = f"{label}  p={score:.3f}  thr={float(threshold):.2f}"

    thickness = max(3, min(out.shape[:2]) // 90)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(out.shape[1], out.shape[0]) / 620.0)
    text_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    pad_x = 10
    pad_y = 8
    x2 = out.shape[1] - 8
    y1 = out.shape[0] - text_h - baseline - pad_y * 2 - 8
    x1 = max(8, x2 - text_w - pad_x * 2)
    y2 = out.shape[0] - 8

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.putText(
        out,
        text,
        (x1 + pad_x, y2 - pad_y - baseline),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return out


def make_dashboard(states: list[StreamState], sampling: SamplingConfig, display_cfg: dict[str, Any]) -> np.ndarray:
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
            lines = status_lines(state, sampling)
            tcn_prob = state.latest_tcn_prob
            tcn_smooth = state.latest_tcn_smooth
            tcn_over_threshold = state.latest_tcn_over_threshold
            tcn_threshold = state.tcn_threshold

        if frame is None:
            frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        if show_status:
            tile = draw_status(tile, lines)
        tile = draw_tcn_indicator(
            tile,
            prob=tcn_prob,
            smooth_prob=tcn_smooth,
            over_threshold=tcn_over_threshold,
            threshold=tcn_threshold,
        )
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


def maybe_print_status(states: list[StreamState], sampling: SamplingConfig, interval_sec: float, last_print: float) -> float:
    if interval_sec <= 0:
        return last_print
    now = time.monotonic()
    if now - last_print < interval_sec:
        return last_print

    parts = []
    for state in states:
        with state.lock:
            if not state.tcn_enabled:
                tcn_part = "tcn=disabled"
            elif state.latest_tcn_prob is None:
                reason = f"({state.tcn_wait_reason})" if state.tcn_wait_reason else ""
                tcn_part = f"tcn=p...{reason}"
            else:
                tcn_part = f"tcn={state.latest_tcn_prob:.3f}/{'ON' if state.latest_tcn_state else 'OFF'}"
            error_part = f" err={state.tcn_error[:80]}" if state.tcn_error else ""
            parts.append(
                f"{state.stream_id}: frame={state.frame_index} infer={state.inferred_count} "
                f"det={state.latest_det_count} window={len(state.sample_window)}/{state.sample_window.maxlen} "
                f"{tcn_part} tcn_n={state.latest_tcn_infer_count}{error_part} connected={state.connected}"
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

    # OpenCV GUI backends usually process the close event only when waitKey
    # runs. A few short ticks make Ctrl+C cleanup deterministic.
    for _ in range(5):
        cv2.waitKey(1)
        time.sleep(0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multistream RTSP/video YOLO-pose inference with TCN-ready frame payload windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV dashboard even if config enables it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    training_cfg = load_training_config(config)
    tcn_cfg = dict(config.get("tcn", {}) or {})
    sampling = resolve_sampling(config, training_cfg)
    tcn_window_size = resolve_tcn_window_size(config, training_cfg)
    model_type_hint = resolve_tcn_model_type_hint(config, training_cfg)
    initial_required_payloads = required_payload_window_size(model_type_hint, tcn_window_size)
    if tcn_cfg_enabled(config) and motion_feature_cfg(deep_get(training_cfg, ("features",), default={}) or {}).get("enabled", False):
        initial_required_payloads = max(initial_required_payloads, tcn_window_size + 1)
    states = enabled_streams(
        config,
        sample_window_size=initial_required_payloads,
        tcn_window_size=tcn_window_size,
    )
    enabled_ids, disabled_ids = stream_ids_by_enabled_state(config)

    model_cfg = dict(config.get("model", {}) or {})
    yolo_path = model_cfg.get("yolo_pose")
    if not yolo_path:
        raise ValueError("model.yolo_pose is required")

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

    max_persons = int(first_present(
        tcn_cfg.get("max_persons"),
        deep_get(training_cfg, ("data", "max_persons")),
        default=25,
    ))
    fallback_feature_cfg = dict(first_present(tcn_cfg.get("feature_cfg"), deep_get(training_cfg, ("features",)), default={}) or {})
    tcn_runtime = load_tcn_runtime(
        config,
        device=device,
        max_persons=max_persons,
        fallback_window_size=tcn_window_size,
    )
    configure_stream_tcn_state(states, tcn_runtime)
    feature_cfg = tcn_runtime.feature_cfg if tcn_runtime.enabled else fallback_feature_cfg

    output_cfg = dict(config.get("output", {}) or {})
    output_dir_value = output_cfg.get("latest_payload_dir")
    output_dir = Path(output_dir_value).resolve() if output_dir_value else None

    capture_cfg = dict(config.get("capture", {}) or {})
    backend = opencv_backend(str(capture_cfg.get("backend", "ffmpeg")))
    reconnect_delay_sec = float(capture_cfg.get("reconnect_delay_sec", 2.0))
    read_sleep_sec = float(capture_cfg.get("read_sleep_sec", 0.0))

    display_cfg = dict(config.get("display", {}) or {})
    display_enabled = bool(display_cfg.get("enabled", True)) and not bool(args.no_display)
    display_interval_sec = 1.0 / max(float(display_cfg.get("fps", 10.0)), 0.1)
    window_name = str(display_cfg.get("window_name", "rtsp_yolo_pose_multistream"))
    show_boxes = bool(display_cfg.get("show_boxes", True))
    show_keypoints = bool(display_cfg.get("show_keypoints", True))
    show_skeleton = bool(display_cfg.get("show_skeleton", True))
    keypoint_conf = float(display_cfg.get("keypoint_conf", 0.25))

    print(
        "[INFO] "
        f"streams={len(states)} device={device} half={half} imgsz={imgsz} "
        f"sampling={sampling.resolved_mode} "
        f"step_frames={sampling.step_frames} sample_period_sec={sampling.sample_period_sec} "
        f"payload_window={states[0].sample_window.maxlen} tcn_window={states[0].tcn_window_size} max_persons={max_persons}",
        flush=True,
    )
    if tcn_runtime.enabled:
        print(
            "[INFO] tcn="
            f"model_type={tcn_runtime.model_type} checkpoint={tcn_runtime.checkpoint_path} "
            f"window={tcn_runtime.window_size} required_payloads={tcn_runtime.required_payload_window_size} "
            f"stride={tcn_runtime.stride} threshold={tcn_runtime.threshold} "
            f"hysteresis={tcn_runtime.use_hysteresis} smooth={tcn_runtime.smooth_mode}",
            flush=True,
        )
    else:
        print("[INFO] tcn=disabled", flush=True)
    print(f"[INFO] enabled_streams={enabled_ids}", flush=True)
    if disabled_ids:
        print(f"[INFO] disabled_streams={disabled_ids}", flush=True)
    if display_enabled:
        print(
            "[INFO] display="
            f"columns={int(display_cfg.get('columns', 2))} "
            f"tile={int(display_cfg.get('tile_width', 480))}x{int(display_cfg.get('tile_height', 270))} "
            f"max={int(display_cfg.get('max_width', 0) or 0)}x{int(display_cfg.get('max_height', 0) or 0)}",
            flush=True,
        )

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

    threads = [
        threading.Thread(
            target=stream_reader,
            args=(state, stop_event, backend, reconnect_delay_sec, read_sleep_sec),
            name=f"reader:{state.stream_id}",
            daemon=True,
        )
        for state in states
    ]
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

                # Per sampled stream we now have two useful data surfaces:
                # - `item.frame`: raw sampled BGR image before overlays. Use it
                #   for RGB clip models such as TSM.
                # - `payload`: normalized boxes/keypoints in the same frame
                #   schema used by our JSON training data. Use it for TCN or any
                #   other pose/metadata model trained on the same representation.
                tcn_candidates: list[tuple[StreamState, list[dict[str, Any]], float | None]] = []
                for item, result in zip(batch, results):
                    detections = result_to_detection_list(result, conf_thresh=conf, ignored_classes=ignored_classes)
                    payload = build_payload(item, detections)
                    # * payload is in the JSON/ dict "common format"
                    display_frame = draw_pose_overlay(
                        item.frame,
                        detections,
                        names=getattr(yolo.model, "names", None),
                        show_boxes=show_boxes,
                        show_keypoints=show_keypoints,
                        show_skeleton=show_skeleton,
                        keypoint_conf=keypoint_conf,
                    )
                    #* place here Erez Update function ....
                    #* store last 10 sec according  item.time_sec
                    update_state_after_inference(
                        item,
                        payload,
                        display_frame,
                        max_persons=max_persons,
                        feature_cfg=feature_cfg,
                        output_dir=output_dir,
                    )
                    # TCN consumes the rolling pose-payload window maintained in
                    # `StreamState.sample_window`. A TSM RGB model should not use
                    # this deque directly unless it was trained on pose payloads;
                    # add a separate raw-frame deque if it needs clips.
                    ready_window = collect_tcn_window_if_ready(item.state, tcn_runtime)
                    if ready_window is not None:
                        window_frames, aspect_ratio = ready_window
                        tcn_candidates.append((item.state, window_frames, aspect_ratio))

                # TCN is run once per loop as a batch across all streams with a
                # ready window. A TSM integration should run in the same place,
                # after YOLO has created the sampled data and before display.
                run_tcn_candidates(tcn_candidates, tcn_runtime, max_persons=max_persons)

            now = time.monotonic()
            if display_enabled and now - last_display_time >= display_interval_sec:
                dashboard = make_dashboard(states, sampling, display_cfg)
                display_layout_printed = print_display_layout_once(
                    states,
                    display_cfg,
                    dashboard,
                    display_layout_printed,
                )
                if not display_window_resized:
                    cv2.resizeWindow(window_name, dashboard.shape[1], dashboard.shape[0])
                    display_window_resized = True
                cv2.imshow(window_name, dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    stop_event.set()
                last_display_time = now

            last_status_print = maybe_print_status(states, sampling, status_interval_sec, last_status_print)
            time.sleep(poll_sleep_sec)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        close_display_windows(display_enabled, window_name)
        for thread in threads:
            thread.join(timeout=2.0)
        close_display_windows(display_enabled, window_name)


if __name__ == "__main__":
    main()
