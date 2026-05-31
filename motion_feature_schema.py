""" Shared motion-feature contract for cache build, training, and live runtime."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
#local imports
from analyze_json_motion import DEFAULT_VERSION, _clip_pooling, _temporal_conv_1d, extract_motion_features


DEFAULT_FEATURE_EXTRACTOR = "extract_motion_features"
DEFAULT_POOL_MODE = "max"
DEFAULT_TEMP_SMOOTHING = True
DEFAULT_TEMP_KERNEL = 3
FEATURE_SCHEMA_KEY = "feature_schema"
TEMPORAL_SCHEMA_KEY = "temporal_schema"
SOURCE_CACHES_KEY = "source_caches"


def feature_dim_from_flags(*, pure_motion:bool = False, legacy: bool = False) -> int:
    if legacy:
        return 18
    if pure_motion:
        return 21
    return 25


def normalize_motion_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ Normalize one frame list to the training-side motion-extractor schema."""
    # TODO: move the function to json_utils (i want to do it only at the next commit)
    for frm in frames:
        if 'detections_list' not in frm and 'detection_list' not in frm:
            continue
        detections_list = frm.get('detections_list', frm.get('detection_list'))
        for det in detections_list:
            # key_pts = det.get("key_pts")
            det['key_pts'] = det.get('key_pts', det.get('key_points',[]))
            if 'key_points' in det:
                del det['key_points']
        frm['detections_list'] =  detections_list
        if 'detection_list' in frm:
            del frm['detection_list']
    return frames

def build_feature_schema(**kwargs) -> dict[str, Any]:
    """ Describe the exact pooled motion-feature contract."""
    pure_motion = bool(kwargs.get('pure_motion', False))
    legacy = bool(kwargs.get('legacy', False))
    return {'extractor': DEFAULT_FEATURE_EXTRACTOR,
            'extractor_version': kwargs.get('j_version', DEFAULT_VERSION),
            'pure_motion': kwargs.get('pure_motion', False),
            'legacy': kwargs.get('legacy', False),
            'temp_smooth': bool(kwargs.get('temp_smooth', DEFAULT_TEMP_SMOOTHING)),
            'temp_kernel': int(kwargs.get('temp_kernel', DEFAULT_TEMP_KERNEL)),
            'pool_mode'  : str(kwargs.get('pool_mode', DEFAULT_POOL_MODE)),
            'feature_dim': feature_dim_from_flags(pure_motion=pure_motion, legacy=legacy),
            }


def build_temporal_schema(window: float, stride: float) -> dict[str, float]:
    """ Describe the cache window/stride used to create one temporal dataset."""
    return {'window': float(window), 'stride': float(stride)}


def get_clip_features_vec(frames: list[dict[str, Any]], **kwargs) -> np.ndarray:
    """ Convert one clip/window frame list into the pooled feature
      vector saved in cache `X`."""
    norm_frames = normalize_motion_frames(frames)
    if len(norm_frames) < 2:
        raise ValueError("At least 2 frames are required to build one clip feature vector")

    feature_schema = build_feature_schema(**kwargs)
    motion_seq = extract_motion_features(norm_frames,
                                         j_version=float(feature_schema['extractor_version']),
                                         pure_motion=bool(feature_schema['pure_motion']),
                                         legacy=bool(feature_schema['legacy']), )
    if bool(feature_schema['temp_smooth']):
        motion_seq = _temporal_conv_1d(motion_seq, int(feature_schema['temp_kernel']))
    clip_feat = _clip_pooling(motion_seq, mode=str(feature_schema['pool_mode']))
    return np.asarray(clip_feat, dtype=np.float32)


def build_cache_record(path: str|Path, feature_schema:dict[str, Any], temporal_schema: dict[str,Any]) -> dict[str, Any]:
    """Build one provenance record for a source cache."""
    return {'path': str(path), 'feature_schema': dict(feature_schema), 'temporal_schema': dict(temporal_schema),}


def pack_json_value(value: Any) -> np.ndarray:
    """Pack a JSON-serializable value into an NPZ-friendly scalar string."""
    return np.asarray(json.dumps(value, sort_keys=True))


def unpack_json_value(value: Any) -> Any:
    """Unpack a JSON value previously saved with `pack_json_value(...)`."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_cache_contract(npz_path: str|Path) -> dict[str, Any]:
    """ Load canonical feature/temporal schema and provenance from one cache NPZ."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    missing = [key for key in (FEATURE_SCHEMA_KEY, TEMPORAL_SCHEMA_KEY) if key not in data.files]
    if missing:
        raise ValueError(f"{npz_path} is missing required cache metadata: {', '.join(missing)}")

    feature_schema = unpack_json_value(data[FEATURE_SCHEMA_KEY])
    temporal_schema = unpack_json_value(data[TEMPORAL_SCHEMA_KEY])
    source_caches = None
    if SOURCE_CACHES_KEY in data.files:
        source_caches = unpack_json_value(data[SOURCE_CACHES_KEY])
    if not source_caches:
        source_caches = [build_cache_record(npz_path, feature_schema, temporal_schema)]

    return {'path': str(npz_path), 'feature_schema': feature_schema,
            'temporal_schema': temporal_schema, 'source_caches': source_caches,}


def load_cache_contract_compact(npz_path:str|Path) -> tuple[dict[str, Any], bool]:
    """ Load one cache contract, falling back to `N/A` metadata for legacy caches."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    if FEATURE_SCHEMA_KEY in data.files and TEMPORAL_SCHEMA_KEY in data.files:
        return load_cache_contract(npz_path), False

    feature_dim = None
    if 'X' in data.files:
        X = data['X']
        if getattr(X, "ndim", None) == 2:
            feature_dim = int(X.shape[1])

    feature_schema = {'extractor': "N/A", 'extractor_version': "N/A",
                      'temp_smooth': "N/A",'temp_kernel': "N/A", 'pool_mode': "N/A",
                      'pure_motion': "N/A", 'legacy': "N/A",
                      'feature_dim': int(feature_dim) if feature_dim is not None else "N/A",
                      }
    temporal_schema = {'window': "N/A", 'stride': "N/A"}
    source_caches = [build_cache_record(npz_path, feature_schema, temporal_schema)]
    return {'path': str(npz_path), 'feature_schema': feature_schema,
            'temporal_schema': temporal_schema,'source_caches': source_caches,}, True


def schema_has_na(schema: dict[str, Any])->bool:
    """ Return whether a schema still contains legacy `N/A` placeholders."""
    return any(value == "N/A" for value in schema.values())


def assert_feature_schema_match(reference: dict[str, Any], candidate: dict[str, Any], *, context: str) -> None:
    """Fail if two caches/models do not share the exact same feature contract."""
    if dict(reference) != dict(candidate):
        raise ValueError(f"Feature schema mismatch for {context}")


def temporal_schema_compatible( reference: dict[str, Any], candidate: dict[str, Any], *,
                                window_tolerance: float, stride_tolerance: float,) -> bool:
    """ Return whether one temporal schema stays inside the allowed drift from a reference target."""
    return (abs(float(candidate['window']) - float(reference['window'])) <= float(window_tolerance) and
            abs(float(candidate['stride']) - float(reference['stride'])) <= float(stride_tolerance))

#222(,,1) -> 169()
