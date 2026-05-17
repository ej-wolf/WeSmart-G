""" TMS runtime handling: accumulate and inspect online time windows.

    This module defines the TMS window specs used in live monitoring and checks
    whether the shared per-stream payload history has accumulated enough recent
    data for each window. It reads from `state.sample_window`, updates compact
    per-window readiness summaries in `state.temporal_probe_status`, and tracks
    the last trigger time per window in `state.temporal_probe_last_trigger_t`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class TemporalWindowSpec:
    """Online trigger policy for one TMS temporal window."""
    tag: str               #* runtime tag, e.g. 'fast' or 'slow'
    window_span: float     #* target time span for analysis
    t_infer: float         #* target time between two online trigger attempts
    min_frames: int        #* minimal number of payloads required for analysis
    tolerance: float       #* tolerance for near-boundary payload inclusion


def default_temporal_probes() -> list[TemporalWindowSpec]:
    """Return the default fast and slow TMS online probes."""
    return [TemporalWindowSpec(tag="fast", window_span=1.2, t_infer=0.2, min_frames=5, tolerance=0.05),
            TemporalWindowSpec(tag="slow", window_span=3.0, t_infer=0.6, min_frames=12, tolerance=0.05),]


def resolve_temporal_probes(config: dict[str, Any], first_present) -> list[TemporalWindowSpec]:
    """Load online TMS probe specs from config.

    If `temporal.probes` is not defined, return the built-in default probes.
    Validation here is intentionally strict so bad runtime settings fail early.
    """
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
        tag = str(first_present(item.get("tag"), item.get("name"), default=f"probe_{index}")).strip()
        if not tag:
            raise ValueError(f"temporal.probes[{index}] tag must be non-empty")
        if tag in seen_names:
            raise ValueError(f"Duplicate temporal probe tag: {tag}")
        seen_names.add(tag)
        window_span = float(first_present(item.get("window_span"), item.get("window_sec")))
        t_infer = float(first_present(item.get("t_infer"), item.get("infer_every_sec"), item.get("trigger_every_sec"), default=0.0))
        min_frames = int(item.get("min_frames"))
        tolerance = float(first_present(item.get("tolerance"), item.get("tolerance_sec"), default=0.05))
        if window_span <= 0:
            raise ValueError(f"temporal.probes[{index}].window_span must be positive")
        if t_infer <= 0:
            raise ValueError(f"temporal.probes[{index}].t_infer must be positive")
        if min_frames <= 0:
            raise ValueError(f"temporal.probes[{index}].min_frames must be positive")
        if tolerance < 0:
            raise ValueError(f"temporal.probes[{index}].tolerance must be non-negative")
        specs.append(TemporalWindowSpec(tag=tag,
                                        window_span=window_span,
                                        t_infer=t_infer,
                                        min_frames=min_frames,
                                        tolerance=tolerance,
                                        )
                     )
    return specs


def frames_span_sec(frames: list[dict[str, Any]]) -> float:
    """Return the covered time span of a payload frame list."""
    if len(frames) < 2:
        return 0.0
    return max(0.0, float(frames[-1]["t"]) - float(frames[0]["t"]))


def collect_probe_window(state: Any, spec:TemporalWindowSpec, latest_t:float, )-> list[dict[str, Any]]|None:
    """ Collect one probe window from shared payload history if it is ready.
    The check is time-based:
    - respect the probe cooldown (`t_infer`)
    - collect payloads from the last `window_span + tolerance`
    - require both minimum frame count and enough covered span

    The result is written back into `state.temporal_probe_status` even when the
    window is not ready, so the caller can inspect/debug readiness live.
    """
    last_trigger_t = state.temporal_probe_last_trigger_t.get(spec.tag)
    since_last = None if last_trigger_t is None else float(latest_t) - float(last_trigger_t)
    if since_last is not None and since_last + 1e-9 < float(spec.t_infer):
        state.temporal_probe_status[spec.tag] = {
                        "ready": False,
                        "reason": "cooldown",
                        "frame_count": 0,
                        "span_sec": 0.0,
                        "latest_t": float(latest_t),
                        "since_last_trigger_sec": float(since_last),
                        "window_span": float(spec.window_span),
                        "t_infer": float(spec.t_infer),
                        "min_frames": int(spec.min_frames),
                        "tolerance": float(spec.tolerance),
                         }
        return None

    min_t = float(latest_t) - float(spec.window_span) - float(spec.tolerance)
    frames = [frame for frame in state.sample_window if float(frame.get("t", 0.0)) >= min_t]
    span_sec = frames_span_sec(frames)
    required_span_sec = max(float(spec.window_span) - float(spec.tolerance), 0.0)

    ready = len(frames) >= int(spec.min_frames) and span_sec + 1e-9 >= required_span_sec
    reason = "ready"
    if not ready:
        if len(frames) < int(spec.min_frames):
            reason = "min_frames"
        else:
            reason = "span"

    state.temporal_probe_status[spec.tag] = {"ready": bool(ready),
                                             "reason": reason,
                                             "frame_count": len(frames),
                                             "span_sec": float(span_sec),
                                             "latest_t": float(latest_t),
                                             "since_last_trigger_sec": None if since_last is None else float(since_last),
                                             "window_span": float(spec.window_span),
                                             "t_infer": float(spec.t_infer),
                                             "min_frames": int(spec.min_frames),
                                             "tolerance": float(spec.tolerance),}
    if not ready:
        return None
    state.temporal_probe_last_trigger_t[spec.tag] = float(latest_t)
    return frames


def temporal_probe_status_line(state: Any, spec: TemporalWindowSpec) -> str:
    """Render one compact status line for a probe."""
    status = state.temporal_probe_status.get(spec.tag, {})
    frame_count = int(status.get("frame_count", 0))
    span_sec = float(status.get("span_sec", 0.0))
    reason = str(status.get("reason", "pending"))
    if bool(status.get("ready", False)):
        return f"{spec.tag}: READY {frame_count}f {span_sec:.2f}/{spec.window_span:.2f}s"
    if reason == "cooldown":
        since_last = status.get("since_last_trigger_sec")
        if since_last is None:
            return f"{spec.tag}: cooldown"
        return f"{spec.tag}: cool {float(since_last):.2f}/{spec.t_infer:.2f}s"
    if reason == "span":
        return f"{spec.tag}: span {frame_count}f {span_sec:.2f}/{spec.window_span:.2f}s"
    if reason == "min_frames":
        return f"{spec.tag}: cnt {frame_count}/{spec.min_frames} {span_sec:.2f}s"
    return f"{spec.tag}: pending"
#145
