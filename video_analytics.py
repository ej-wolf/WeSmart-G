from pathlib import Path
import cv2
import numpy as np


DEFAULT_MF_THRESHOLD = 1.0       #* Meaningful-Frame (MF) motion threshold for FPS measurement.
MF_THRESHOLD_RATIO = {'low': 0.05, #* Low/high threshold floors expressed as ratios of the meaningful-frame threshold.
                      'high': 0.25}
MF_MAD_FACTOR = {'low': 3.0, 'high': 6.0} #* Low/high factors applied to background-noise MAD.
SAMPLE_DURATION = 5.0           #* sec, Duration of detailed measurement windows
MAX_DENSE_SAMPLES = 8           #* Maximum number of detailed sampeling
FRAME_ANALYSIS_WIDTH = 160      #* pixels, Width of resized frames for motion analysis.
BLOCK_SIZE = 8                  #* pixels, Side length of motion-analysis block.
BLUR_KERNEL = 3                 #* Length of the Gaussian blur kernel used for analysis.
MIN_CHANGED_BLOCK = 0.02        #* Minimum fraction of changed blocks required for a frame update.
#* Scouting
SCOUT_THRESHOLD_SEC = 60.0      #* Minimum video duration in seconds for scouting.
SCOUT_INTERVAL = {'min': 1.0,
                  'max': 8.0}   #* Minimum/maximum scout-frame gap in seconds.
SCOUT_NOISE_MAD = 3.0           #* Factor applied to the scout background-noise MAD.
MIN_CONVERGED_SAMPLES = 3       #* Minimum number of accepted samples required for convergence.
FPS_CLUSTER_RTOL = 0.10         #* Relative tolerance for grouping similar measured FPS values.
FPS_CONSISTENCY_RTOL = 0.10     #* Relative tolerance for measured versus encoded FPS.
MIN_DYNAMIC_RUN = 2.0           #* sec, Minimum duration for a dynamic run to be sampled.

# region API
def measure_video_fps(video, mf_threshold=DEFAULT_MF_THRESHOLD, **kwargs) -> tuple[float, dict]:
    """ Measure one video's visual FPS from dynamic frame-update cadence. """

    def _pos_n_finite(val, name):
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return val

    def get_safe_kwargs(arg, fall_back):
        return _pos_n_finite(kwargs.get(arg, fall_back), arg)

    def _trim_scout_interval(x):
        return round(min(SCOUT_INTERVAL['max'], max(SCOUT_INTERVAL['min'], x)))

    if not np.isfinite(mf_threshold) or mf_threshold < 0:
        raise ValueError('mf_threshold must be finite and non-negative')
    scout_interval  = get_safe_kwargs('scout_interval', SCOUT_INTERVAL['min'])
    sample_duration = get_safe_kwargs('sample_duration', SAMPLE_DURATION)
    min_samples  = get_safe_kwargs('min_samples', MIN_CONVERGED_SAMPLES)
    max_samples  = get_safe_kwargs('max_samples', MAX_DENSE_SAMPLES)
    cluster_rtol = get_safe_kwargs('cluster_rtol', FPS_CLUSTER_RTOL)
    min_active_run = get_safe_kwargs('min_active_run', MIN_DYNAMIC_RUN)
    if min_samples > max_samples:
        raise ValueError('min_samples cannot exceed max_samples')

    video = Path(video)
    capture = cv2.VideoCapture(str(video))
    try:
        fps_encoded, frame_count, duration = _video_metadata(capture)
        scout_interval = _trim_scout_interval(scout_interval)
        scout = None
        near_static = False
        if duration >= SCOUT_THRESHOLD_SEC:
            scout = _scout_video(capture, duration, scout_interval, mf_threshold)
            intervals = _scout_intervals(scout, duration, sample_duration, min_active_run, max_samples)
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

        cluster = _dominant_cluster(samples, cluster_rtol)
        for sample in samples:
            sample['accepted'] = sample in cluster

        if cluster:
            estimates = [s['fps_estimate'] for s in cluster]
            confidences = [s['confidence'] for s in cluster]
            fps_measured = _weighted_median(estimates, confidences)
            fps_msr_std  = _weighted_spread(estimates, confidences, fps_measured)
            if len(cluster) == 1:
                fps_msr_std = None
        else:
            fps_measured, fps_msr_std = 0.0, None

        fps_noise = fps_msr_std/fps_measured  if fps_msr_std is not None and fps_measured > 0 else float('inf')
        fps_consistent = (fps_measured > 0 and abs(fps_measured - fps_encoded)/fps_encoded <= FPS_CONSISTENCY_RTOL)

        mean_diff, mean_mf_diff = _diff_means(samples, mf_threshold)
        result = _video_result(video, duration, fps_encoded, fps_measured, fps_msr_std,
                               fps_noise,  fps_consistent, near_static, mean_diff, mean_mf_diff, samples)
        result['analysis'] = { 'method': 'adaptive_cadence',
                               'converged': converged,
                               'accepted_samples': len(cluster),
                               'sample_count': len(samples),
                               'scout_interval': scout_interval if scout is not None else None,
                               'min_active_run': min_active_run,
                               'dynamic_windows': scout['dynamic_windows'] if scout is not None else None,
                               }
        return fps_measured, result
    finally:
        capture.release()


def legacy_vid_eff_fps(video, mf_threshold=DEFAULT_MF_THRESHOLD) -> tuple[float, dict]:
    """Run the former global-mean effective-FPS algorithm unchanged."""

    mf_threshold = float(mf_threshold)
    if not np.isfinite(mf_threshold) or mf_threshold < 0:
        raise ValueError('mf_threshold must be finite and non-negative')
    video = Path(video)
    capture = cv2.VideoCapture(str(video))
    try:
        fps_enc, frm_count, duration = _video_metadata(capture)
        samples = []
        for start, end in _legacy_sample_intervals(duration):
            samples.append(_read_legacy_sample(capture, start, end))

        spans = [sum(s['elapsed']) for s in samples]
        fps_vals = [sum(df >= mf_threshold for df in smp['diffs'])/spn for smp, spn in zip(samples, spans)]
        fps_msr, fps_msr_std = _weighted_stats(fps_vals, spans)
        if len(samples) == 1:
            fps_msr_std = None
        fps_noise = (fps_msr_std/fps_msr if fps_msr_std is not None and fps_msr > 0 else float('inf'))
        fps_consistent = (abs(fps_msr - fps_enc)/fps_enc <= FPS_CONSISTENCY_RTOL)
        mean_diff, mean_mf_diff = _diff_means(samples, mf_threshold)

        return fps_msr, _video_result( video, duration, fps_enc, fps_msr, fps_msr_std, fps_noise,
                                       fps_consistent, fps_msr <= 0, mean_diff, mean_mf_diff, samples)
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
    updates = [bool(val.max() >= high_th  or np.mean(val >= low_th) >= MIN_CHANGED_BLOCK)
                   for val in block_diffs]
    span = sum(elapsed)
    update_count = int(sum(updates))
    fps_estimate = min(update_count/span, 1 / np.median(elapsed))
    activity = float(np.mean([np.percentile(values, 90) for values in block_diffs]))
    informative = bool(update_count >= max(2, int(np.ceil(span * 0.5))))
    confidence = float(update_count * max(activity, float(np.finfo(float).eps)))
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
    scale = FRAME_ANALYSIS_WIDTH / gray.shape[1]
    height = max(BLOCK_SIZE, int(round(gray.shape[0] * scale)))
    small = cv2.resize(gray, (FRAME_ANALYSIS_WIDTH, height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (BLUR_KERNEL, BLUR_KERNEL), 0)


def _block_differences(current, previous):
    diff = cv2.absdiff(current, previous).astype(np.float32)
    h = diff.shape[0]// BLOCK_SIZE*BLOCK_SIZE
    w = diff.shape[1]// BLOCK_SIZE*BLOCK_SIZE
    if h == 0 or w == 0:
        return np.asarray([float(diff.mean())])
    blocks = diff[:h, :w].reshape(h// BLOCK_SIZE, BLOCK_SIZE, w// BLOCK_SIZE, BLOCK_SIZE)
    return blocks.mean(axis=(1, 3)).ravel()


def _adaptive_thresholds(block_diffs, mf_threshold):
    values = np.concatenate(block_diffs)
    lower = values[values <= np.percentile(values, 25)]
    noise = float(np.median(lower))
    noise_mad = float(np.median(np.abs(lower - noise)))
    low =  max(mf_threshold*MF_THRESHOLD_RATIO['low'],  noise + MF_MAD_FACTOR['low'] *noise_mad)
    high = max(mf_threshold*MF_THRESHOLD_RATIO['high'], noise + MF_MAD_FACTOR['high']*noise_mad, 2*low)
    return float(low), float(high)


def _scout_video(capture, t_vid, interval, mf_threshold):
    samples = []
    prev = None
    for t_sec in np.arange(0.0, t_vid, interval):
        if t_sec > 0 and not capture.set(cv2.CAP_PROP_POS_MSEC, float(t_sec*1000)):
            continue
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = _analysis_frame(gray)
        if prev is not None:
            vals = _block_differences(small, prev)
            samples.append({'time': float(t_sec), 'activity': float(np.percentile(vals, 90))})
        prev = small

    if not samples:
        return {'threshold': None, 'samples': [], 'dynamic_windows': 0}
    activities = np.asarray([sample['activity'] for sample in samples])
    lower = activities[activities <= np.percentile(activities, 25)]
    noise = float(np.median(lower))
    noise_mad = float(np.median(np.abs(lower - noise)))
    threshold = max(mf_threshold, noise + SCOUT_NOISE_MAD * noise_mad)
    for sample in samples:
        sample['dynamic'] = sample['activity'] >= threshold
    return {'threshold': threshold, 'samples': samples, 'dynamic_windows': 0}


def _scout_intervals(scout_results, t_vid, t_smp, min_dynamic_seg, max_samples):
    scout_samples = scout_results['samples']
    if not scout_samples:
        return []
    step = (np.median(np.diff( [smp['time'] for smp in scout_samples] ))
                                if len(scout_samples) > 1 else 1.0)
    segment, dynamic_segments = [], []
    for sample in scout_samples:
        if sample['dynamic']:
            segment.append(sample)
        elif segment:
            dynamic_segments.append(segment)
            segment = []
    if segment:
        dynamic_segments.append(segment)

    candidates = []
    for seg in dynamic_segments:
        segment_duration = seg[-1]['time'] - seg[0]['time'] + step
        if segment_duration >= min_dynamic_seg:
            center = (seg[0]['time'] + seg[-1]['time']) / 2
            score = float(np.mean([sample['activity'] for sample in seg]))
            candidates.append((score, center))
    scout_results['dynamic_windows'] = len(candidates)

    intervals = []
    min_dist = t_smp/2
    for _, center in sorted(candidates, reverse=True):
        start = min(max(0.0, center - t_smp/2), max(0.0, t_vid - t_smp))
        if all(abs(start - prior[0]) >= min_dist for prior in intervals):
            intervals.append((start, min(t_vid, start + t_smp)))
        if len(intervals) >= max_samples:
            break
    return intervals


def _fallback_interval(scout_results, t_vid, t_smp):
    samples = scout_results['samples']
    t_mid = (max(samples, key=lambda sample:sample['activity'])['time']
              if samples else t_vid/2)
    t_start = min(max(0.0, t_mid - t_smp/2), max(0.0, t_vid - t_smp))
    return t_start, min(t_vid, t_start + t_smp)


def _short_video_intervals(t_vid, t_smp, max_samples):
    win = min(t_vid, t_smp)
    if t_vid <= win:
        return [(0.0, t_vid)]
    starts = [0.0, (t_vid - win)/2, t_vid - win]
    starts += list(np.linspace(0.0, t_vid - win, max_samples))
    intervals = []
    for start in starts:
        interval = (float(start), float(min(t_vid, start + win)))
        if all(abs(interval[0] - prior[0]) > 1e-6 for prior in intervals):
            intervals.append(interval)
        if len(intervals) >= max_samples:
            break
    return intervals


def _dominant_cluster(samples, rtol):
    informative = [smp for smp in samples if smp['informative']]
    best = []
    best_key = (0, 0.0)
    for ref_sample in informative:
        ref = ref_sample['fps_estimate']
        cluster = [smp for smp in informative if abs(smp['fps_estimate'] - ref)
                                              <= rtol*max(smp['fps_estimate'], ref)]
        key = (len(cluster), sum(smp['confidence'] for smp in cluster))
        if key > best_key:
            best, best_key = cluster, key
    return best


def _weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    mid_pt = sum(weights)/2
    w_sum = 0.0
    for v, w in pairs:
        w_sum += w
        if w_sum >= mid_pt:
            return float(v)
    return float(pairs[-1][0])


def _weighted_spread(v, w, ref):
    return float(np.sqrt(np.average((np.asarray(v) - ref)* 2, weights=np.asarray(w))))


def _video_result(video, t_vid, fps_enc, fps_msr, fps_msr_std, fps_noise, fps_cns,
                  near_static, mean_diff, mean_mf_diff, samples):
    t_sampled = sum( sum(smp['elapsed']) if isinstance(smp['elapsed'], list) else
                     smp['elapsed']['mean']*len(smp['diffs'])  for smp in samples)

    return   {'titel': video.stem,
              'path': str(video),
              'duration': float(t_vid),
              'sampled_duration': float(t_sampled),
              'fps_encoded': float(fps_enc),
              'fps_measured': float(fps_msr),
              'fps_measured_std': fps_msr_std,
              'fps_noise': fps_noise,
              'fps_consistent': bool(fps_cns),
              'near_static': bool(near_static),
              'mean_diff': mean_diff,
              'mean_mf_diff': mean_mf_diff,
              'samples': samples}


def _legacy_sample_intervals(t_vid):
    if t_vid <= 5:
        return [(0.0, t_vid)]
    if t_vid <= 60:
        return [(0.0, 2.5), (t_vid - 2.5, t_vid)]
    intervals = [(float(start), min(float(start + 3), t_vid))
                 for start in range(0, int(np.ceil(t_vid)), 60)]
    intervals.append((max(0.0, t_vid - 3), t_vid))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _weighted_stats(values, weights):
    values  = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = np.average(values, weights=weights)
    std  = np.sqrt(np.average((values - mean)**2, weights=weights))
    return float(mean), float(std)


def _diff_means(samples, mf_threshold):
    diffs = [diff for sample in samples for diff in sample['diffs']]
    if not diffs:
        return 0.0, None
    meaningful = [diff for diff in diffs if diff >= mf_threshold]
    return float(np.mean(diffs)), float(np.mean(meaningful)) if meaningful else None

# endregion
#460(1,1,) -448
#492(1,,2)-> 455()
