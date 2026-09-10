from pathlib import Path

import cv2
import numpy as np


DEFAULT_MF_TH = 1.0
SCOUT_INTERVAL = 1.0
SCOUT_MAX_INTERVAL = 8.0
SCOUT_SAMPLE_TARGET = 120
SAMPLE_DURATION = 5.0
MIN_CONVERGED_SAMPLES = 3
MAX_DENSE_SAMPLES = 8
FPS_CLUSTER_RTOL = 0.10
MIN_ACTIVITY_RUN = 2.0
LONG_VIDEO_DURATION = 60.0

ANALYSIS_WIDTH = 160
BLOCK_SIZE = 8
BLUR_KERNEL = 3
LOW_THRESHOLD_RATIO = 0.05
HIGH_THRESHOLD_RATIO = 0.25
LOW_NOISE_MAD = 3.0
HIGH_NOISE_MAD = 6.0
MIN_CHANGED_BLOCK_RATIO = 0.02
SCOUT_NOISE_MAD = 3.0


# region API
def legacy_measure_vid_eff_fps(video, mf_threshold=DEFAULT_MF_TH) -> tuple[float, dict]:
    """Run the former global-mean effective-FPS algorithm unchanged."""
    mf_threshold = float(mf_threshold)
    if not np.isfinite(mf_threshold) or mf_threshold < 0:
        raise ValueError('mf_threshold must be finite and nonnegative')
    video = Path(video)
    capture = cv2.VideoCapture(str(video))
    try:
        fps_encoded, frame_count, duration = _video_metadata(capture)
        samples = []
        for start, end in _legacy_sample_intervals(duration):
            samples.append(_read_legacy_sample(capture, start, end))

        spans = [sum(s['elapsed']) for s in samples]
        fps_vals = [sum(df >= mf_threshold for df in smp['diffs'])/spn for smp, spn in zip(samples, spans)]
        fps_eff, fps_eff_std = _weighted_stats(fps_vals, spans)
        mean_diff, mean_mf_diff = _diff_means(samples, mf_threshold)
        return fps_eff, _video_result(video, duration, fps_encoded, fps_eff, fps_eff_std if len(samples) > 1 else None,
                                      mean_diff, mean_mf_diff, samples)
    finally:
        capture.release()


def measure_video_effective_fps(video,
                                mf_threshold=DEFAULT_MF_TH,
                                scout_interval=SCOUT_INTERVAL,
                                sample_duration=SAMPLE_DURATION,
                                min_samples=MIN_CONVERGED_SAMPLES,
                                max_samples=MAX_DENSE_SAMPLES,
                                cluster_rtol=FPS_CLUSTER_RTOL,
                                min_activity_run=MIN_ACTIVITY_RUN) -> tuple[float, dict]:
    """Measure one video's effective FPS from dynamic frame-update cadence."""

    def _pos_n_finite(val, name):
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return val

    def  _trim_scout_interval(x):
        return round( min(SCOUT_MAX_INTERVAL, max(x, duration/SCOUT_SAMPLE_TARGET)))


    if not np.isfinite(mf_threshold) or mf_threshold < 0:
        raise ValueError('mf_threshold must be finite and nonnegative')
    scout_interval = _pos_n_finite(scout_interval, 'scout_interval')
    sample_duration = _pos_n_finite(sample_duration, 'sample_duration')
    min_activity_run = _pos_n_finite(min_activity_run, 'min_activity_run')
    cluster_rtol = _pos_n_finite(cluster_rtol, 'cluster_rtol')
    min_samples = _pos_n_finite(min_samples, 'min_samples')
    max_samples = _pos_n_finite(max_samples, 'max_samples')
    if min_samples > max_samples:
        raise ValueError('min_samples cannot exceed max_samples')

    video = Path(video)
    capture = cv2.VideoCapture(str(video))
    try:
        fps_encoded, frame_count, duration = _video_metadata(capture)
        scout_interval = _trim_scout_interval(scout_interval)
        scout = None
        near_static = False
        if duration > LONG_VIDEO_DURATION:
            scout = _scout_video(capture, duration, scout_interval, mf_threshold)
            intervals = _scout_intervals(scout, duration, sample_duration,
                                         min_activity_run, max_samples)
            near_static = not intervals
            if near_static:
                intervals = [_fallback_interval(scout, duration, sample_duration)]
        else:
            intervals = _short_video_intervals(duration, sample_duration, max_samples)

        converged = False
        cluster,samples = [], []
        for start, end in intervals:
            sample = _read_cadence_sample(capture, start, end, mf_threshold)
            samples.append(sample)
            cluster = _dominant_cluster(samples, cluster_rtol)
            converged = len(cluster) >= min_samples
            if converged:
                break

        if not near_static and not any(sample['informative'] for sample in samples):
            near_static = scout is None or not scout['dynamic_windows']

        cluster = [] if near_static else _dominant_cluster(samples, cluster_rtol)
        for sample in samples:
            sample['accepted'] = sample in cluster

        if cluster:
            estimates = [s['fps_estimate'] for s in cluster]
            confidences = [s['confidence'] for s in cluster]
            fps_eff     = _weighted_median(estimates, confidences)
            fps_eff_std = _weighted_spread(estimates, confidences, fps_eff)
            if len(cluster) == 1:
                fps_eff_std = None
        else:
            fps_eff, fps_eff_std = 0.0, None

        mean_diff, mean_mf_diff = _diff_means(samples, mf_threshold)
        result = _video_result(video, duration, fps_encoded, fps_eff, fps_eff_std, mean_diff, mean_mf_diff, samples)
        result['analysis'] = { 'method': 'adaptive_cadence',
                               'near_static': near_static,
                               'converged': converged,
                               'accepted_samples': len(cluster),
                               'sample_count': len(samples),
                               'scout_interval': scout_interval if scout is not None else None,
                               'min_activity_run': min_activity_run,
                               'dynamic_windows': scout['dynamic_windows'] if scout is not None else None,
                               }
        return fps_eff, result
    finally:
        capture.release()
# endregion

# region Helpers
def _video_metadata(capture):
    if not capture.isOpened():
        raise ValueError('cannot open video')
    fps_encoded = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not np.isfinite(fps_encoded) or fps_encoded <= 0:
        raise ValueError('invalid encoded FPS')
    if not np.isfinite(frame_count) or frame_count < 2:
        raise ValueError('at least two video frames are required')
    return float(fps_encoded), float(frame_count), float(frame_count / fps_encoded)


def _read_legacy_sample(capture, start, end):

    if start > 0 and not capture.set(cv2.CAP_PROP_POS_MSEC, start*1000):
        raise ValueError(f'cannot seek to {start:g}s')

    prev, t_prev = None, None
    first_t = None
    invalid_tail = False
    diffs, elapsed = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t_current = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
        valid_timestamp = np.isfinite(t_current) and t_current >= 0
        if prev is not None and (not valid_timestamp or t_current <= t_prev):
            invalid_tail = True
            continue
        if invalid_tail:
            raise ValueError('decoded timestamps must be strictly increasing')
        if not valid_timestamp:
            raise ValueError('invalid decoded timestamp')
        if t_current < start:
            continue
        if t_current >= end:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(cv2.absdiff(gray, prev).mean()))
            elapsed.append(float(t_current - t_prev))
        else:
            first_t = float(t_current)
        prev, t_prev = gray, t_current

    if not elapsed:
        raise ValueError(f'insufficient frames in sample {start:g}-{end:g}s')
    return {'start': first_t, 'end': float(t_prev), 'diffs': diffs, 'elapsed': elapsed}


def _read_cadence_sample(capture, start, end, mf_threshold):
    if not capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000):
        raise ValueError(f'cannot seek to {start:g}s')

    prev_gray,  prev_small, t_prev, first_t = None,  None,  None,  None
    diffs, elapsed, block_diffs = [], [], []
    invalid_tail = False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t_current = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        valid_timestamp = np.isfinite(t_current) and t_current >= 0
        if prev_gray is not None and (not valid_timestamp or t_current <= t_prev):
            invalid_tail = True
            continue
        if invalid_tail:
            raise ValueError('decoded timestamps must be strictly increasing')
        if not valid_timestamp:
            raise ValueError('invalid decoded timestamp')
        if t_current < start:
            continue
        if t_current >= end:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = _analysis_frame(gray)
        if prev_gray is not None:
            diffs.append(float(cv2.absdiff(gray, prev_gray).mean()))
            elapsed.append(float(t_current - t_prev))
            block_diffs.append(_block_differences(small, prev_small))
        else:
            first_t = float(t_current)
        prev_gray, prev_small, t_prev = gray, small, t_current

    if not elapsed:
        raise ValueError(f'insufficient frames in sample {start:g}-{end:g}s')

    elapsed_stats = {'mean': float(np.mean(elapsed)),
                     'std': float(np.std(elapsed))}
    low_th, high_th = _adaptive_thresholds(block_diffs, mf_threshold)
    updates = [bool(val.max() >= high_th  or np.mean(val >= low_th) >= MIN_CHANGED_BLOCK_RATIO)
                   for val in block_diffs]
    span = sum(elapsed)
    update_count = int(sum(updates))
    fps_estimate = min(update_count/span, 1 / np.median(elapsed))
    activity = float(np.mean([np.percentile(values, 90) for values in block_diffs]))
    informative = bool(update_count >= max(2, int(np.ceil(span * 0.5))))
    confidence = float(update_count*max(activity, np.finfo(float).eps))
    return {'start': first_t, 'end': float(t_prev),
            'diffs': diffs,
            'elapsed': elapsed_stats,
            'fps_estimate': float(fps_estimate),
            'activity': activity,
            'informative': informative,
            'accepted': False,
            'update_count': update_count,
            'confidence': confidence,
            'thresholds': {'low': low_th, 'high': high_th},
            }


def _analysis_frame(gray):
    scale = ANALYSIS_WIDTH / gray.shape[1]
    height = max(BLOCK_SIZE, int(round(gray.shape[0] * scale)))
    small = cv2.resize(gray, (ANALYSIS_WIDTH, height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (BLUR_KERNEL, BLUR_KERNEL), 0)


def _block_differences(current, previous):
    diff = cv2.absdiff(current, previous).astype(np.float32)
    height = diff.shape[0]// BLOCK_SIZE*BLOCK_SIZE
    width = diff.shape[1]// BLOCK_SIZE*BLOCK_SIZE
    if height == 0 or width == 0:
        return np.asarray([float(diff.mean())])
    blocks = diff[:height, :width].reshape(height// BLOCK_SIZE, BLOCK_SIZE, width// BLOCK_SIZE, BLOCK_SIZE)
    return blocks.mean(axis=(1, 3)).ravel()


def _adaptive_thresholds(block_diffs, mf_threshold):
    values = np.concatenate(block_diffs)
    lower = values[values <= np.percentile(values, 25)]
    noise = float(np.median(lower))
    noise_mad = float(np.median(np.abs(lower - noise)))
    low = max(mf_threshold * LOW_THRESHOLD_RATIO, noise + LOW_NOISE_MAD * noise_mad)
    high = max(mf_threshold * HIGH_THRESHOLD_RATIO,
               noise + HIGH_NOISE_MAD * noise_mad, low * 2)
    return float(low), float(high)


def _scout_video(capture, duration, interval, mf_threshold):
    observations = []
    prev = None
    for t_sec in np.arange(0.0, duration, interval):
        if t_sec > 0 and not capture.set(cv2.CAP_PROP_POS_MSEC, float(t_sec * 1000)):
            continue
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = _analysis_frame(gray)
        if prev is not None:
            values = _block_differences(small, prev)
            observations.append({'time': float(t_sec),
                                 'activity': float(np.percentile(values, 90))})
        prev = small

    if not observations:
        return {'threshold': None, 'observations': [], 'dynamic_windows': 0}
    activities = np.asarray([item['activity'] for item in observations])
    lower = activities[activities <= np.percentile(activities, 25)]
    noise = float(np.median(lower))
    noise_mad = float(np.median(np.abs(lower - noise)))
    threshold = max(mf_threshold, noise + SCOUT_NOISE_MAD * noise_mad)
    for item in observations:
        item['dynamic'] = item['activity'] >= threshold
    return {'threshold': float(threshold), 'observations': observations,
            'dynamic_windows': 0}


def _scout_intervals(scout, duration, sample_duration, min_activity_run, max_samples):
    observations = scout['observations']
    if not observations:
        return []
    step = (np.median(np.diff([item['time'] for item in observations]))
            if len(observations) > 1 else 1.0)
    runs = []
    run = []
    for obs in observations:
        if obs['dynamic']:
            run.append(obs)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)

    candidates = []
    for run in runs:
        run_duration = run[-1]['time'] - run[0]['time'] + step
        if run_duration >= min_activity_run:
            center = (run[0]['time'] + run[-1]['time']) / 2
            score = float(np.mean([obs['activity'] for obs in run]))
            candidates.append((score, center))
    scout['dynamic_windows'] = len(candidates)

    intervals = []
    min_distance = sample_duration/2
    for _, center in sorted(candidates, reverse=True):
        start = min(max(0.0, center - sample_duration/2),  max(0.0, duration - sample_duration))
        if all(abs(start - prior[0]) >= min_distance for prior in intervals):
            intervals.append((start, min(duration, start + sample_duration)))
        if len(intervals) >= max_samples:
            break
    return intervals


def _fallback_interval(scout, duration, sample_duration):
    observations = scout['observations']
    center = (max(observations, key=lambda item: item['activity'])['time']
              if observations else duration / 2)
    start = min(max(0.0, center - sample_duration / 2),
                max(0.0, duration - sample_duration))
    return start, min(duration, start + sample_duration)


def _short_video_intervals(duration, sample_duration, max_samples):
    window = min(duration, sample_duration)
    if duration <= window:
        return [(0.0, duration)]
    starts = [0.0, (duration - window) / 2, duration - window]
    starts += list(np.linspace(0.0, duration - window, max_samples))
    intervals = []
    for start in starts:
        interval = (float(start), float(min(duration, start + window)))
        if all(abs(interval[0] - prior[0]) > 1e-6 for prior in intervals):
            intervals.append(interval)
        if len(intervals) >= max_samples:
            break
    return intervals


def _dominant_cluster(samples, rtol):
    informative = [sample for sample in samples if sample['informative']]
    best = []
    best_key = (0, 0.0)
    for center_sample in informative:
        center = center_sample['fps_estimate']
        cluster = [sample for sample in informative
                   if abs(sample['fps_estimate'] - center)
                   <= rtol * max(sample['fps_estimate'], center)]
        key = (len(cluster), sum(sample['confidence'] for sample in cluster))
        if key > best_key:
            best, best_key = cluster, key
    return best


def _weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    midpoint = sum(weights) / 2
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= midpoint:
            return float(value)
    return float(pairs[-1][0])


def _weighted_spread(values, weights, center):
    return float(np.sqrt(np.average((np.asarray(values) - center) ** 2,
                                    weights=np.asarray(weights))))


def _video_result(video, duration, fps_encoded, fps_eff, fps_eff_std,
                  mean_diff, mean_mf_diff, samples):
    def round_report(value):
        if isinstance(value, dict):
            return {key: round_report(item) for key, item in value.items()}
        if isinstance(value, list):
            return [round_report(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return round(float(value), 5)
        return value

    sampled_duration = sum(
        sum(sample['elapsed']) if isinstance(sample['elapsed'], list)
        else sample['elapsed']['mean'] * len(sample['diffs'])
        for sample in samples)
    result = {'titel': video.stem,
              'path': str(video),
              'duration': float(duration),
              'sampled_duration': float(sampled_duration),
              'fps_encoded': float(fps_encoded),
              'fps_eff': float(fps_eff),
              'fps_eff_std': fps_eff_std,
              'mean_diff': mean_diff,
              'mean_mf_diff': mean_mf_diff,
              'samples': samples}
    return round_report(result)


def _legacy_sample_intervals(duration):
    if duration <= 5:
        return [(0.0, duration)]
    if duration <= 60:
        return [(0.0, 2.5), (duration - 2.5, duration)]
    intervals = [(float(start), min(float(start + 3), duration))
                 for start in range(0, int(np.ceil(duration)), 60)]
    intervals.append((max(0.0, duration - 3), duration))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _weighted_stats(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
    return float(mean), float(std)


def _diff_means(samples, mf_threshold):
    diffs = [diff for sample in samples for diff in sample['diffs']]
    if not diffs:
        return 0.0, None
    meaningful = [diff for diff in diffs if diff >= mf_threshold]
    return float(np.mean(diffs)), float(np.mean(meaningful)) if meaningful else None
# endregion
#460(1,1,) -448
