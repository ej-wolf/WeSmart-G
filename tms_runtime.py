"""TMS online runtime helpers for temporal window readiness and probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TemporalWindowSpec:
    name: str
    window_sec: float
    infer_every_sec: float
    min_frames: int
    tolerance_sec: float


def default_temporal_probes() -> list[TemporalWindowSpec]:
    return [
        TemporalWindowSpec(name="fast", window_sec=1.2, infer_every_sec=0.2, min_frames=5, tolerance_sec=0.05),
        TemporalWindowSpec(name="slow", window_sec=3.0, infer_every_sec=0.6, min_frames=12, tolerance_sec=0.05),
    ]


def resolve_temporal_probes(config: dict[str, Any], first_present) -> list[TemporalWindowSpec]:
    temporal_cfg = dict(config.get("temporal", {}) or {})
    probe_cfgs = temporal_cfg.get("probes")
    if probe_cfgs is None:
        return default_temporal_probes()
    if not isinstance(probe_cfgs, list):
        raise ValueError("temporal.probes must be a list")

    specs: list[TemporalWindowSpec] = []
    seen_names: set[str] = set()
    for index, item in enumerate(probe_cfgs):
        if not isinstance(item, dict):
            raise ValueError(f"temporal.probes[{index}] must be a mapping")
        name = str(item.get("name") or f"probe_{index}").strip()
        if not name:
            raise ValueError(f"temporal.probes[{index}] name must be non-empty")
        if name in seen_names:
            raise ValueError(f"Duplicate temporal probe name: {name}")
        seen_names.add(name)
        window_sec = float(item.get("window_sec"))
        infer_every_sec = float(first_present(item.get("infer_every_sec"), item.get("trigger_every_sec"), default=0.0))
        min_frames = int(item.get("min_frames"))
        tolerance_sec = float(first_present(item.get("tolerance_sec"), default=0.05))
        if window_sec <= 0:
            raise ValueError(f"temporal.probes[{index}].window_sec must be positive")
        if infer_every_sec <= 0:
            raise ValueError(f"temporal.probes[{index}].infer_every_sec must be positive")
        if min_frames <= 0:
            raise ValueError(f"temporal.probes[{index}].min_frames must be positive")
        if tolerance_sec < 0:
            raise ValueError(f"temporal.probes[{index}].tolerance_sec must be non-negative")
        specs.append(
            TemporalWindowSpec(
                name=name,
                window_sec=window_sec,
                infer_every_sec=infer_every_sec,
                min_frames=min_frames,
                tolerance_sec=tolerance_sec,
            )
        )
    return specs


def frames_span_sec(frames: list[dict[str, Any]]) -> float:
    if len(frames) < 2:
        return 0.0
    return max(0.0, float(frames[-1]["t"]) - float(frames[0]["t"]))


def collect_temporal_probe_window(
    state: Any,
    spec: TemporalWindowSpec,
    latest_t: float,
) -> list[dict[str, Any]] | None:
    last_trigger_t = state.temporal_probe_last_trigger_t.get(spec.name)
    since_last = None if last_trigger_t is None else float(latest_t) - float(last_trigger_t)
    if since_last is not None and since_last + 1e-9 < float(spec.infer_every_sec):
        state.temporal_probe_status[spec.name] = {
            "ready": False,
            "reason": "cooldown",
            "frame_count": 0,
            "span_sec": 0.0,
            "latest_t": float(latest_t),
            "since_last_trigger_sec": float(since_last),
            "window_sec": float(spec.window_sec),
            "infer_every_sec": float(spec.infer_every_sec),
            "min_frames": int(spec.min_frames),
            "tolerance_sec": float(spec.tolerance_sec),
        }
        return None

    min_t = float(latest_t) - float(spec.window_sec) - float(spec.tolerance_sec)
    frames = [frame for frame in state.sample_window if float(frame.get("t", 0.0)) >= min_t]
    span_sec = frames_span_sec(frames)
    required_span_sec = max(float(spec.window_sec) - float(spec.tolerance_sec), 0.0)

    ready = len(frames) >= int(spec.min_frames) and span_sec + 1e-9 >= required_span_sec
    reason = "ready"
    if not ready:
        if len(frames) < int(spec.min_frames):
            reason = "min_frames"
        else:
            reason = "span"

    state.temporal_probe_status[spec.name] = {
        "ready": bool(ready),
        "reason": reason,
        "frame_count": len(frames),
        "span_sec": float(span_sec),
        "latest_t": float(latest_t),
        "since_last_trigger_sec": None if since_last is None else float(since_last),
        "window_sec": float(spec.window_sec),
        "infer_every_sec": float(spec.infer_every_sec),
        "min_frames": int(spec.min_frames),
        "tolerance_sec": float(spec.tolerance_sec),
    }
    if not ready:
        return None

    state.temporal_probe_last_trigger_t[spec.name] = float(latest_t)
    return frames


def temporal_probe_status_line(state: Any, spec: TemporalWindowSpec) -> str:
    status = state.temporal_probe_status.get(spec.name, {})
    frame_count = int(status.get("frame_count", 0))
    span_sec = float(status.get("span_sec", 0.0))
    reason = str(status.get("reason", "pending"))
    if bool(status.get("ready", False)):
        return f"{spec.name}: READY {frame_count}f {span_sec:.2f}/{spec.window_sec:.2f}s"
    if reason == "cooldown":
        since_last = status.get("since_last_trigger_sec")
        if since_last is None:
            return f"{spec.name}: cooldown"
        return f"{spec.name}: cool {float(since_last):.2f}/{spec.infer_every_sec:.2f}s"
    if reason == "span":
        return f"{spec.name}: span {frame_count}f {span_sec:.2f}/{spec.window_sec:.2f}s"
    if reason == "min_frames":
        return f"{spec.name}: cnt {frame_count}/{spec.min_frames} {span_sec:.2f}s"
    return f"{spec.name}: pending"
