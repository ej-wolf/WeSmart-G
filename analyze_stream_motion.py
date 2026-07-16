"""     Extract order-free dt-normalized motion features from stream JSON frame sequences.
    extract_motion_features(...): main API func.
    converts per-frame BB/keypoints into a clip-level motion representation.
    The motion is represented by a 25-dim feature vector for each pair of consecutive frames.
    The features vectors are normalized by frame timestamp deltas.
    Static overlap features are copied from frame t+1 and remain unscaled.
    Feature order:
    1.  d_mean_center_x      : change in mean bbox center (x,y); global  crowd motion
    2.  d_mean_center_y      :
    3.  d_var_center_x       : change in bbox-center variance;  spread/compression
    4.  d_var_center_y       :
    5.  d_mean_size_w        : change in mean bbox size (width/height); average scale/depth change
    6.  d_mean_size_h        :
    7.  d_max_size_w         : change in largest bbox size; strongest scale change
    8.  d_max_size_h         :
    9.  d_mean_pairwise      : change in mean pairwise bbox-center distance; crowd density cue
    10. d_var_pairwise       : change in variance of pairwise distances; spacing heterogeneity
    11. d_kp_mean_x          : change in mean keypoint(x, y); global articulated  motion
    12. d_kp_mean_y          :
    13. d_kp_var_x           : change in keypoint variance; pose spread
    14. d_kp_var_y           :
    15. bb_nn_mean           : mean nearest-neighbor bbox-center motion; average object motion
    16. bb_nn_max            : max nearest-neighbor bbox-center motion; strongest mover
    17. kp_nn_mean           : mean nearest-neighbor keypoint motion; average articulated motion
    18. kp_nn_max            : max nearest-neighbor keypoint motion; strongest articulated motion
    * Added later (14/03/26)
    19. d_union_coverage     : change in union area covered by all bboxes
    20. d_mean_pairwise_iou  : change in mean pairwise bbox IoU
    21. d_max_pairwise_iou   : change in max pairwise bbox IoU
    * Static features
    22. union_coverage       : union area covered by all bboxes in frame t+1
    23. mean_pairwise_iou    : mean pairwise bbox IoU in frame t+1
    24. max_pairwise_iou     : max pairwise bbox IoU in frame t+1
    25. overlap_ratio        : redundant bbox overlap ratio in frame t+1
"""

from __future__ import annotations
import numpy as np

N_KEYPOINTS = 17
DEFAULT_VERSION = 3.0
MOTION_FPS_REF = 5.0
MOTION_FPS_MIN = 1.0
MOTION_FPS_MAX = 20.0
DEFAULT_TOP_K_RATIO = 0.2
DEFAULT_TOP_K_MIN = 2
POOL_MODE_ALIASES = {'mm': 'mean_max', 'msm': 'mean_std_max'}


# region API

def extract_motion_features(frames, j_version: float = DEFAULT_VERSION, **kwargs):
    """ Convert frames into a (T-1) x C dt-normalized motion sequence."""
    def dt_scale(frm_0, frm_1):
        dt = frm_1.get('t', 0.0) - frm_0.get('t', 0.0)
        dt = min(max(dt, dt_min), dt_max)
        return 1.0/dt if fps_ref_scale is None else fps_ref_scale/dt

    if len(frames) < 2:
        raise ValueError("At least 2 frames are required to extract motion features")

    frame_ftrs, raw_pts = [], []
    kp_conf = float(j_version) >= 2.0
    fps_min = float(kwargs.get('motion_fps_min', MOTION_FPS_MIN))
    fps_max = float(kwargs.get('motion_fps_max', MOTION_FPS_MAX))
    dt_min, dt_max = 1.0/fps_max, 1.0/fps_min
    fps_ref = kwargs.get('motion_fps_ref', MOTION_FPS_REF)
    fps_ref_scale = None if fps_ref is None or float(fps_ref) == 1.0 else 1.0/float(fps_ref)

    for frame in frames:
        bb_centers, bb_sizes, keypoints, bboxes = extract_frame_geometry(frame, kp_conf=kp_conf)
        frame_ftrs.append(frame_aggregates(bb_centers, bb_sizes, keypoints, bboxes))
        raw_pts.append((bb_centers, keypoints))

    frame_ftrs = np.stack(frame_ftrs)
    motion_feats = []
    for idx in range(len(frames) - 1):
        scale = dt_scale(frames[idx], frames[idx + 1])
        delta_agg = (frame_ftrs[idx + 1] - frame_ftrs[idx]) * scale
        curr_agg = frame_ftrs[idx + 1]

        bb_mean, bb_max = nearest_neighbor_motion(raw_pts[idx][0], raw_pts[idx + 1][0])
        kp_mean, kp_max = nearest_neighbor_motion(raw_pts[idx][1], raw_pts[idx + 1][1])
        nn_motion = np.asarray([bb_mean, bb_max, kp_mean, kp_max], dtype=np.float32) * scale

        motion_vec = np.concatenate([delta_agg[:14], nn_motion,
                                     delta_agg[14:17],
                                     curr_agg[14:17],
                                     curr_agg[17:18],])
        motion_feats.append(motion_vec)

    motion_feats = np.stack(motion_feats).astype(np.float32)
    #* ToDo: Future time-shift features should be inserted here, before clip pooling.
    if kwargs.get('legacy', False):
        return motion_feats[:, :18]
    if kwargs.get('pure_motion', False):
        return motion_feats[:, :21]
    return motion_feats


def _clip_pooling(motion_seq, mode='max', **kwargs):
    """Reduce a variable-length motion sequence to one fixed feature vector."""
    if len(motion_seq) == 0:
        return None
    mode = canonical_pool_mode(mode)
    if mode == 'max':
        return motion_seq.max(axis=0)
    if mode == 'mean':
        return motion_seq.mean(axis=0)
    if mode == 'lse':
        alpha = kwargs.get('alpha', 5.0)
        return (1.0/alpha) * np.log(np.exp(alpha * motion_seq).sum(axis=0))
    if mode == 'top_k':
        k_ratio = float(kwargs.get('top_k_ratio', kwargs.get('pool_top_k_ratio', DEFAULT_TOP_K_RATIO)))
        k_min = int(kwargs.get('top_k_min', kwargs.get('pool_top_k_min', DEFAULT_TOP_K_MIN)))
        k = min(len(motion_seq), max(k_min, int(np.ceil(len(motion_seq)*k_ratio))))
        return np.sort(motion_seq, axis=0)[-k:].mean(axis=0)
    if mode == 'mean_max':
        return np.concatenate([motion_seq.mean(axis=0), motion_seq.max(axis=0)])
    if mode == 'mean_std_max':
        return np.concatenate([motion_seq.mean(axis=0), motion_seq.std(axis=0), motion_seq.max(axis=0)])
    raise ValueError(f"Unknown pooling mode: {mode}")


def canonical_pool_mode(mode):
    """Return the stored name for a pooling mode or alias."""
    return POOL_MODE_ALIASES.get(str(mode), str(mode))


def _temporal_conv_1d(motion_seq, kernel_size=3):
    """Apply a light temporal mean filter over a motion sequence."""
    if len(motion_seq) < kernel_size:
        return motion_seq

    pad = kernel_size // 2
    padded = np.pad(motion_seq, ((pad, pad), (0, 0)), mode='edge')
    return np.stack([padded[idx:idx + kernel_size].mean(axis=0) for idx in range(len(motion_seq))])

# endregion

# region helpers

def extract_frame_geometry(frame, **kwargs):
    """Extract bbox centers/sizes, keypoint coordinates, and raw boxes from one frame."""
    vec_kp = kwargs.get('vec_kp', 3 if kwargs.get('kp_conf', True) else 2)
    bb_centers, bb_sizes, keypoints, bboxes = [], [], [], []

    for det in frame.get('detections_list', []):
        x1, y1, x2, y2 = det['bbox']
        w, h = x2 - x1, y2 - y1
        bb_centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
        bb_sizes.append([w, h])
        bboxes.append([x1, y1, x2, y2])

        kps = det.get('key_pts', [])
        for idx in range(0, len(kps), vec_kp):
            x, y = kps[idx], kps[idx + 1]
            if x == 0 and y == 0:
                continue
            keypoints.append([x, y])

    return (np.asarray(bb_centers, dtype=np.float32),
            np.asarray(bb_sizes  , dtype=np.float32),
            np.asarray(keypoints , dtype=np.float32),
            np.asarray(bboxes    , dtype=np.float32),)


def frame_aggregates(bb_centers, bb_sizes, keypoints, bboxes):
    """ Compute order-free, tracking-free frame geometry descriptors."""
    feats = []
    if len(bb_centers) > 0:
        mean_center = bb_centers.mean(axis=0)
        var_center = bb_centers.var(axis=0)
        mean_size = bb_sizes.mean(axis=0)
        max_size = bb_sizes.max(axis=0)
        if len(bb_centers) > 1:
            diffs = bb_centers[:, None, :] - bb_centers[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            pairwise = dists[np.triu_indices(len(bb_centers), k=1)]
            mean_pairwise = pairwise.mean()
            var_pairwise = pairwise.var()
        else:
            mean_pairwise = var_pairwise = 0.0
        union_coverage = _bbox_union_area(bboxes)
        mean_pairwise_iou, max_pairwise_iou = _pairwise_iou_stats(bboxes)
        overlap_ratio = _overlap_ratio(bboxes)
    else:
        mean_center = var_center = mean_size = max_size = np.zeros(2)
        mean_pairwise = var_pairwise = 0.0
        union_coverage = mean_pairwise_iou = max_pairwise_iou = overlap_ratio = 0.0

    feats.extend(mean_center)
    feats.extend(var_center)
    feats.extend(mean_size)
    feats.extend(max_size)
    feats.extend([mean_pairwise, var_pairwise])
    if len(keypoints) > 0:
        feats.extend(keypoints.mean(axis=0))
        feats.extend(keypoints.var(axis=0))
    else:
        feats.extend(np.zeros(2))
        feats.extend(np.zeros(2))
    feats.extend([union_coverage, mean_pairwise_iou, max_pairwise_iou, overlap_ratio])
    return np.asarray(feats, dtype=np.float32)


def nearest_neighbor_motion(pt_1, pt_2):
    """ Return mean/max nearest-neighbor distance between two point sets."""
    if len(pt_1) == 0 or len(pt_2) == 0:
        return 0.0, 0.0
    dists = [np.linalg.norm(pt_2 - point, axis=1).min() for point in pt_1]
    dists = np.asarray(dists)
    return dists.mean(), dists.max()


def _bbox_union_area(bboxes):
    if len(bboxes) == 0:
        return 0.0

    boxes = np.asarray(bboxes, dtype=np.float64).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, 1.0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, 1.0)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if len(boxes) == 0:
        return 0.0

    xs = np.unique(boxes[:, [0, 2]])
    area = 0.0
    for idx in range(len(xs) - 1):
        x_l, x_r = xs[idx], xs[idx + 1]
        if x_r <= x_l:
            continue
        active = boxes[(boxes[:, 0] < x_r) & (boxes[:, 2] > x_l)]
        if len(active) == 0:
            continue
        ys = sorted((y1, y2) for _, y1, _, y2 in active)
        cur_y1, cur_y2 = ys[0]
        covered = 0.0
        for y1, y2 in ys[1:]:
            if y1 <= cur_y2:
                cur_y2 = max(cur_y2, y2)
            else:
                covered += cur_y2 - cur_y1
                cur_y1, cur_y2 = y1, y2
        covered += cur_y2 - cur_y1
        area += (x_r - x_l) * covered
    return area


def _pairwise_iou_stats(bboxes):
    if len(bboxes) < 2:
        return 0.0, 0.0

    vals = []
    boxes = np.asarray(bboxes, dtype=np.float64)
    for idx_1 in range(len(boxes)):
        x1a, y1a, x2a, y2a = boxes[idx_1]
        area_a = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
        for idx_2 in range(idx_1 + 1, len(boxes)):
            x1b, y1b, x2b, y2b = boxes[idx_2]
            area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
            ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
            ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = area_a + area_b - inter
            vals.append(0.0 if union <= 0 else inter / union)

    vals = np.asarray(vals, dtype=np.float32)
    return vals.mean(), vals.max()


def _overlap_ratio(bboxes):
    if len(bboxes) == 0:
        return 0.0

    boxes = np.asarray(bboxes, dtype=np.float64).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, 1.0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, 1.0)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if len(boxes) == 0:
        return 0.0

    sum_area = np.sum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    if sum_area <= 0:
        return 0.0
    return max(0.0, sum_area - _bbox_union_area(boxes)) / sum_area

# endregion

#286(2,1,)
