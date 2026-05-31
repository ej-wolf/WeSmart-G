""" TMS runtime handling: accumulate and inspect online time windows.

    This module defines the TMS window specs used in live monitoring and checks
    whether the shared per-stream payload history has accumulated enough recent
    data for each window. It reads from `state.sample_window`, updates compact
    per-window readiness summaries in `state.temporal_probe_status`, and tracks
    the last trigger time per window in `state.temporal_probe_last_trigger_t`.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_feature_schema import assert_feature_schema_match, schema_has_na
from torch_clip_model import ClipMLP, LOCAL_CONFIG, _infer_hidden_dim_from_state


@dataclass
class TemporalWindowSpec:
    """Online trigger policy for one TMS temporal window."""
    tag: str               #* runtime tag, e.g. 'fast' or 'slow'
    window_span: float     #* target time span for analysis
    t_infer: float         #* target time between two online trigger attempts
    min_frames: int        #* minimal number of payloads required for analysis
    tolerance: float       #* tolerance for near-boundary payload inclusion


@dataclass
class TmsModelRuntime:
    """One loaded live TMS model bound to one online probe tag."""
    tag: str
    model_path: Path
    threshold: float
    hidden_dim: int
    input_dim: int
    device: torch.device
    model: torch.nn.Module
    feature_schema: dict[str, Any] | None = None
    temporal_profile: dict[str, Any] | None = None


@dataclass
class TmsCandidate:
    """One ready live TMS feature vector waiting for batched inference."""
    tag: str
    state: Any
    feature_vec: np.ndarray
    latest_t: float


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
                                        tolerance=tolerance,)
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
    feature_dim = status.get("feature_dim")
    feature_suffix = f" ->{int(feature_dim)}d" if feature_dim is not None else ""
    if status.get("feature_error"):
        return f"{spec.tag}: feat_err {status['feature_error']}"
    if "tms_prob" in status:
        return f"{spec.tag}: p={float(status['tms_prob']):.2f} y={int(status.get('tms_pred', 0))}{feature_suffix}"
    if bool(status.get("ready", False)):
        return f"{spec.tag}: READY {frame_count}f {span_sec:.2f}/{spec.window_span:.2f}s{feature_suffix}"
    if reason == "cooldown":
        since_last = status.get("since_last_trigger_sec")
        if since_last is None:
            return f"{spec.tag}: cooldown{feature_suffix}"
        return f"{spec.tag}: cool {float(since_last):.2f}/{spec.t_infer:.2f}s{feature_suffix}"
    if reason == "span":
        return f"{spec.tag}: span {frame_count}f {span_sec:.2f}/{spec.window_span:.2f}s{feature_suffix}"
    if reason == "min_frames":
        return f"{spec.tag}: cnt {frame_count}/{spec.min_frames} {span_sec:.2f}s{feature_suffix}"
    return f"{spec.tag}: pending"


def probe_matches_temporal_profile(spec: TemporalWindowSpec, temporal_profile: dict[str, Any]) -> bool:
    """Return whether one online probe is compatible with the model's canonical temporal profile."""
    target_window = float(temporal_profile.get("target_window"))
    window_tolerance = float(temporal_profile.get("window_tolerance", 0.0))
    return abs(float(spec.window_span) - target_window) <= window_tolerance


def validate_probe_temporal_profile(spec: TemporalWindowSpec, temporal_profile: dict[str, Any]) -> None:
    """Fail early if one online probe does not fit the model's temporal window family."""
    if not probe_matches_temporal_profile(spec, temporal_profile):
        raise ValueError(
            f"Probe '{spec.tag}' is incompatible with model temporal profile: "
            f"probe window={spec.window_span}, "
            f"model target_window={temporal_profile.get('target_window')} "
            f"(window_tolerance={temporal_profile.get('window_tolerance', 0.0)})"
        )


def validate_runtime_feature_schema(runtime_feature_schema: dict[str, Any], model_feature_schema: dict[str, Any]) -> None:
    """Fail early if the live feature builder does not match the model contract."""
    assert_feature_schema_match(model_feature_schema, runtime_feature_schema, context="runtime feature builder")


def _resolve_model_checkpoint(model_ref: str | Path, config_dir: Path) -> Path:
    """Resolve one TMS model ref to a checkpoint path."""
    path = Path(model_ref)
    if not path.is_absolute():
        repo_dir = Path(__file__).resolve().parent
        path = (repo_dir / path).resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)

    best_models = sorted(path.glob("best_model.*.pt"))
    if best_models:
        return best_models[-1]
    model_pt = path / "model.pt"
    if model_pt.is_file():
        return model_pt
    checkpoints = sorted(path.glob("checkpoint_ep-*.pt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No model checkpoint found in {path}")


def _load_model_contract(model_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    """Load saved runtime contract from one training config when available."""
    cfg_path = model_path.parent / LOCAL_CONFIG
    if not cfg_path.is_file():
        return None, None, None

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    return cfg.get("feature_schema", None), cfg.get("temporal_profile", None), cfg.get("hidden_dim", None)


def load_tms_runtimes(
    config: dict[str, Any],
    config_dir: Path,
    runtime_feature_schema: dict[str, Any],
    temporal_probes: list[TemporalWindowSpec],
    device: torch.device,
) -> dict[str, TmsModelRuntime]:
    """Load the configured live TMS models and validate them against the runtime contract."""
    tms_cfg = dict(config.get("tms", {}) or {})
    model_cfgs = tms_cfg.get("models", None)
    if model_cfgs is None:
        return {}
    if not isinstance(model_cfgs, list):
        raise ValueError("tms.models must be a list")

    probe_map = {spec.tag: spec for spec in temporal_probes}
    runtimes: dict[str, TmsModelRuntime] = {}
    for index, item in enumerate(model_cfgs):
        if not isinstance(item, dict):
            raise ValueError(f"tms.models[{index}] must be a mapping")
        if item.get("enabled", True) is False:
            continue
        tag = str(item.get("tag", "")).strip()
        if not tag:
            raise ValueError(f"tms.models[{index}] is missing tag")
        if tag in runtimes:
            raise ValueError(f"Duplicate tms model tag: {tag}")
        if tag not in probe_map:
            raise ValueError(f"tms.models[{index}] tag '{tag}' has no matching temporal probe")

        model_ref = item.get("model")
        if not model_ref:
            raise ValueError(f"tms.models[{index}] is missing model")
        model_path = _resolve_model_checkpoint(model_ref, config_dir)
        state = torch.load(model_path, map_location=device)
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported model state in {model_path}")

        input_dim = int(state["net.0.weight"].shape[1])
        feature_schema, temporal_profile, hidden_dim = _load_model_contract(model_path)
        if hidden_dim is None:
            hidden_dim = _infer_hidden_dim_from_state(state)
        hidden_dim = int(hidden_dim)

        if feature_schema and not schema_has_na(feature_schema):
            validate_runtime_feature_schema(runtime_feature_schema, feature_schema)
        elif input_dim != int(runtime_feature_schema["feature_dim"]):
            raise ValueError(
                f"Runtime feature_dim mismatch for {model_path}: runtime={runtime_feature_schema['feature_dim']} model={input_dim}"
            )
        if temporal_profile and not schema_has_na(temporal_profile):
            validate_probe_temporal_profile(probe_map[tag], temporal_profile)

        model = ClipMLP(input_dim, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()

        runtimes[tag] = TmsModelRuntime(
            tag=tag,
            model_path=model_path,
            threshold=float(item.get("threshold", 0.5)),
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            device=device,
            model=model,
            feature_schema=feature_schema,
            temporal_profile=temporal_profile,
        )
    return runtimes


@torch.inference_mode()
def predict_tms_probs_batch(runtime: TmsModelRuntime, feature_vecs: list[np.ndarray]) -> np.ndarray:
    """Run one loaded TMS model on a batch of pooled feature vectors."""
    batch = np.stack([np.asarray(vec, dtype=np.float32) for vec in feature_vecs], axis=0)
    x = torch.from_numpy(batch).to(runtime.device)
    logits = runtime.model(x)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return np.asarray(probs, dtype=np.float32)


def run_tms_candidates(candidates: list[TmsCandidate], runtimes: dict[str, TmsModelRuntime]) -> None:
    """ Batch ready feature vectors by tag/model and write probabilities back into stream state."""
    if not candidates or not runtimes:
        return

    by_tag: dict[str, list[TmsCandidate]] = {}
    for item in candidates:
        if item.tag not in runtimes:
            continue
        by_tag.setdefault(item.tag, []).append(item)

    for tag, tag_candidates in by_tag.items():
        runtime = runtimes[tag]
        probs = predict_tms_probs_batch(runtime, [item.feature_vec for item in tag_candidates])
        for item, prob in zip(tag_candidates, probs):
            y_pred = int(float(prob) >= float(runtime.threshold))
            with item.state.lock:
                item.state.temporal_probe_status.setdefault(tag, {})
                item.state.temporal_probe_status[tag]["tms_prob"] = float(prob)
                item.state.temporal_probe_status[tag]["tms_pred"] = y_pred
                item.state.temporal_probe_status[tag]["tms_threshold"] = float(runtime.threshold)
                item.state.temporal_probe_status[tag]["tms_model"] = runtime.model_path.name
                item.state.temporal_probe_status[tag]["tms_latest_t"] = float(item.latest_t)
                item.state.tms_results[tag] = {
                    "prob": float(prob),
                    "pred": y_pred,
                    "threshold": float(runtime.threshold),
                    "model": runtime.model_path.name,
                    "latest_t": float(item.latest_t),
                }

#370 (1,,)
