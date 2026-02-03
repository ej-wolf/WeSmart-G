import json
from pathlib import Path
import numpy as np

# from my_local_utils import load_json_frames

BB_KP_TAG = 'bbs_list_of_keypoints'
# 'list_of_bbs_keypoints'
N_Keypoints = 13

#* --------------------------------------------------
#* Step 1: load JSON and extract frame-level geometry
#* --------------------------------------------------
#
# def load_frames(json_path: str | Path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     frames = data['frames']
#     return frames

def extract_frame_geometry(frame):
    """  From a single frame dict, extract bounding-box and keypoint geometry.
    Returns:
        bb_centers: (N, 2)
        bb_sizes:   (N, 2)
        keypoints:  (M, 2)  -- flattened over all people
    """
    bb_centers = []
    bb_sizes = []
    keypoints = []

    for bb in frame.get(BB_KP_TAG, []):
        # bounding box
        x1, y1, x2, y2 = bb[2], bb[3], bb[4], bb[5]
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2
        w = x2 - x1
        h = y2 - y1

        bb_centers.append([cx, cy])
        bb_sizes.append([w, h])

        #* keypoints (13 points, aligned)
        kps = bb[6]
        for i in range(0, len(kps), 2):
            x, y = kps[i], kps[i + 1]
            # JS:0 is treated as missing
            if x == 0 and y == 0:
                continue
            keypoints.append([x, y])

    return (
        np.asarray(bb_centers, dtype=np.float32),
        np.asarray(bb_sizes, dtype=np.float32),
        np.asarray(keypoints, dtype=np.float32),
    )


# --------------------------------------------------
# Step 2: compute per-frame aggregate descriptors
# --------------------------------------------------

def frame_aggregates(bb_centers, bb_sizes, keypoints):
    """
    Compute order-free, tracking-free aggregates for one frame.
    Returns a 1D feature vector.
    """
    feats = []

    # bounding boxes
    if len(bb_centers) > 0:
        mean_center = bb_centers.mean(axis=0)
        var_center = bb_centers.var(axis=0)
        mean_size = bb_sizes.mean(axis=0)
        max_size = bb_sizes.max(axis=0)

        # pairwise distances between centers
        if len(bb_centers) > 1:
            diffs = bb_centers[:, None, :] - bb_centers[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            # take upper triangle without diagonal
            iu = np.triu_indices(len(bb_centers), k=1)
            pairwise = dists[iu]
            mean_pairwise = pairwise.mean()
            var_pairwise = pairwise.var()
        else:
            mean_pairwise = 0.0
            var_pairwise = 0.0
    else:
        mean_center = var_center = mean_size = max_size = np.zeros(2)
        mean_pairwise = var_pairwise =0.0
        # var_pairwise = 0.0

    feats.extend(mean_center)
    feats.extend(var_center)
    feats.extend(mean_size)
    feats.extend(max_size)
    feats.append(mean_pairwise)
    feats.append(var_pairwise)

    # keypoints
    if len(keypoints) > 0:
        kp_mean = keypoints.mean(axis=0)
        kp_var = keypoints.var(axis=0)
    else:
        kp_mean = kp_var = np.zeros(2)

    feats.extend(kp_mean)
    feats.extend(kp_var)

    return np.asarray(feats, dtype=np.float32)


# --------------------------------------------------
# Step 3: tracking-free motion between frames
# --------------------------------------------------
def nearest_neighbor_motion(A, B):
    """
    A, B: (N, 2) point sets from consecutive frames
    Returns mean and max nearest-neighbor distance.
    """
    if len(A) == 0 or len(B) == 0:
        return 0.0, 0.0

    dists = []
    for a in A:
        dist = np.linalg.norm(B - a, axis=1)
        dists.append(dist.min())

    dists = np.asarray(dists)
    return float(dists.mean()), float(dists.max())


def compute_motion_sequence(frames):
    """    Converts a list of frames into a T x C motion feature sequence.    """
    frame_feats = []
    raw_points = []

    for frame in frames:
        bb_centers, bb_sizes, keypoints = extract_frame_geometry(frame)
        agg = frame_aggregates(bb_centers, bb_sizes, keypoints)
        frame_feats.append(agg)
        raw_points.append((bb_centers, keypoints))

    frame_feats = np.stack(frame_feats)  # T x C

    motion_feats = []
    for t in range(len(frames) - 1):
        delta_agg = frame_feats[t + 1] - frame_feats[t]

        bb_c_t, kp_t = raw_points[t]
        bb_c_tp1, kp_tp1 = raw_points[t + 1]

        bb_mean, bb_max = nearest_neighbor_motion(bb_c_t, bb_c_tp1)
        kp_mean, kp_max = nearest_neighbor_motion(kp_t, kp_tp1)

        motion_vec = np.concatenate([delta_agg,
                                     np.array([bb_mean, bb_max, kp_mean, kp_max], dtype=np.float32),])
        motion_feats.append(motion_vec)

    return np.stack(motion_feats)  # (T-1) x C'


# --------------------------------------------------
#  Testing
# --------------------------------------------------
import random


def generate_random_frame(n_bb=10):
    bbs = []
    for _ in range(n_bb):
        x1 = random.uniform(0.1, 0.6)
        y1 = random.uniform(0.1, 0.6)
        w  = random.uniform(0.1, 0.2)
        h  = random.uniform(0.1, 0.2)
        x2 = min(x1 + w, 0.95)
        y2 = min(y1 + h, 0.95)

        keypoints = []
        for _ in range(N_Keypoints):
            keypoints +=[random.uniform(x1, x2), random.uniform(y1, y2)]
        # * (0) Individual annotation (ignored), (1) confidence
        bbs += [[0, 0.9, x1, y1, x2, y2, keypoints]]

    return{'f': 0, 't': 0.0, 'individual_events': [], 'group_events': [], BB_KP_TAG: bbs}



def shift_frame(frame, dx=0.1, dy=0.1):
    shifted = {BB_KP_TAG: []}

    for bb in frame[BB_KP_TAG]:
        x1, y1, x2, y2 = bb[2] + dx, bb[3] + dy, bb[4] + dx, bb[5] + dy

        kps = bb[6]
        shifted_kps = []
        for i in range(0, len(kps), 2):
            shifted_kps.extend([kps[i] + dx, kps[i + 1] + dy ])

        shifted[BB_KP_TAG].append([bb[0], bb[1], x1, y1, x2, y2, shifted_kps ])

    return shifted


def controlled_motion_test(dx=0.1, dy=0.1, eps=1e-5, **kwargs):
    frame_0 = generate_random_frame(n_bb=kwargs.get('n_bb',10))
    frame_1 = shift_frame(frame_0, dx=dx, dy=dy)
    #* --- geometry extraction ---
    bb0, _, _ = extract_frame_geometry(frame_0)
    bb1, _, _ = extract_frame_geometry(frame_1)
    #* --- aggregate check ---
    agg0 = frame_aggregates(bb0, np.zeros_like(bb0), np.zeros((0, 2)))
    agg1 = frame_aggregates(bb1, np.zeros_like(bb1), np.zeros((0, 2)))

    delta = agg1 - agg0
    #* indices depend on feature order:
    delta_mean_center = delta[0:2]    #* mean_center = feats[0:2]

    # pairwise distance stats
    mean_pairwise_idx = 8
    var_pairwise_idx = 9

    # --- nearest neighbor motion ---
    nn_mean, nn_max = nearest_neighbor_motion(bb0, bb1)

    print("* Controlled_Motion_Test")
    print("Δmean_center:", delta_mean_center)
    print("pairwise mean delta:", delta[mean_pairwise_idx])
    print("pairwise var delta :", delta[var_pairwise_idx])
    print("NN mean motion     :", nn_mean)
    print("NN max motion      :", nn_max)
    # --- assertions ---
    # assert abs(delta_mean_center[0] - dx) < tol, "mean_center x shift mismatch"
    # assert abs(delta_mean_center[1] - dy) < tol, "mean_center y should not change"
    # assert abs(delta[mean_pairwise_idx]) < eps, "pairwise mean should stay constant"
    # assert abs(delta[var_pairwise_idx] ) < eps, "pairwise var should stay constant"
    # # assert abs(nn_mean - dx) < tol, "NN mean motion mismatch"
    # assert abs(nn_max - dx) < tol, "NN max motion mismatch"

    expected_mag = np.sqrt(dx*dx + dy*dy)
    assert np.allclose(delta_mean_center, [dx, dy], atol=eps)
    assert abs(delta[mean_pairwise_idx]) < eps, "pairwise mean should stay constant"
    assert abs(delta[var_pairwise_idx] ) < eps, "pairwise var should stay constant"

    assert abs(nn_max - expected_mag) < eps
    assert nn_mean <= expected_mag + eps

    print("✔ Controlled motion test PASSED\n")

def crowd_compression_test(scale=0.5, **kwargs): #72
    """ Sanity check for crowd compression (no global translation).
        People move toward their centroid, reducing pairwise distances.
    """

    frame_0 = generate_random_frame(n_bb=kwargs.get('n_bb', 10))
    # extract centers
    bb0, _, _ = extract_frame_geometry(frame_0)
    centroid = bb0.mean(axis=0)

    # build compressed frame
    compressed = {'f': frame_0.get('f', 0),
                  't': frame_0.get('t', 0.0),
                  'individual_events': [],
                  'group_events': [],
                  BB_KP_TAG: []}

    for bb in frame_0[BB_KP_TAG]:
        x1, y1, x2, y2 = bb[2], bb[3], bb[4], bb[5]
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2

        # move center toward centroid
        new_cx = centroid[0] + scale*(cx - centroid[0])
        new_cy = centroid[1] + scale*(cy - centroid[1])
        w = x2 - x1
        h = y2 - y1

        nx1 = new_cx - w/2
        ny1 = new_cy - h/2
        nx2 = new_cx + w/2
        ny2 = new_cy + h/2

        # shift keypoints accordingly
        shifted_kps = []
        for i in range(0, len(bb[6]), 2):
            kx, ky = bb[6][i], bb[6][i + 1]
            dkx = kx - cx
            dky = ky - cy
            shifted_kps.extend([new_cx + scale*dkx,  new_cy + scale*dky ])
        compressed[BB_KP_TAG].append([bb[0], bb[1], nx1, ny1, nx2, ny2, shifted_kps])

    bb1, _, _ = extract_frame_geometry(compressed)

    agg0 = frame_aggregates(bb0, np.zeros_like(bb0), np.zeros((0, 2)))
    agg1 = frame_aggregates(bb1, np.zeros_like(bb1), np.zeros((0, 2)))

    mean_pairwise_idx = 8
    var_pairwise_idx = 9

    delta_mean_center = agg1[0:2] - agg0[0:2]

    print("* Crowd Compression Test")
    print("Δmean_center:", delta_mean_center)
    print("pairwise mean before:", agg0[mean_pairwise_idx])
    print("pairwise mean after :", agg1[mean_pairwise_idx])

    assert np.linalg.norm(delta_mean_center) < kwargs.get('eps', 1e-5)
    assert agg1[mean_pairwise_idx] < agg0[mean_pairwise_idx]

    print("✔ Crowd compression test PASSED\n")



def generate_static_json(n_frm=100, n_bb=10): #55
    """  Generate a JSON-like dict where all frames are identical.
         Useful for zero-motion sanity checks.
    """

    # fixed BBs and keypoints (normalized coords)
    # bbs = []
    # for _ in range(n_bb):
    #     x1 = random.uniform(0.1, 0.6)
    #     y1 = random.uniform(0.1, 0.6)
    #     w  = random.uniform(0.1, 0.2)
    #     h  = random.uniform(0.1, 0.2)
    #     x2 = min(x1 + w, 0.95)
    #     y2 = min(y1 + h, 0.95)
    #
    #     # 13 keypoints → 26 values
    #     keypoints = []
    #     for _ in range(N_Keypoints):
    #         keypoints +=[random.uniform(x1, x2), random.uniform(y1, y2)]
    #
    #     bb = [0, 0.9, x1, y1, x2, y2, keypoints]
    #     bbs.append(bb)
    #     #* (0) Individual annotation (ignored), (1) confidence
    #     # bbs2 += [[0, 0.9, x1, y1, x2, y2, keypoints]]

    # identical frame template
    # frame_template = {'f': 0, 't': 0.0, 'individual_events': [], 'group_events': [], BB_KP_TAG: bbs}
    frame_template = generate_random_frame(n_bb=n_bb)

    frames = []
    for i in range(n_frm):
        frame = frame_template.copy()
        frame['f'] = 5*i
        frame['t'] = i/3  #* 0.33*i
        frames.append(frame)

    return {'video': "static_test", 'fps': 15.0, 'step': 5, 'frames': frames}

def test_motion_sequence(json_example, eps= 1e-5):

    # * load the example file
    with open(json_example, 'r') as f:
        data = json.load(f)
    frames = data['frames']
    motion_seq = compute_motion_sequence(frames)
    # * check (1): shape is stable
    print(f"Number of frames: {len(frames)}\nMotion Shape: {motion_seq.shape}")
    print(f"Correct shape:",  len(frames) - 1 ==  motion_seq.shape[0])
    # print(motion_seq.shape)
    # * check (2): calculation  consistency
    ms2 = compute_motion_sequence(frames)
    assert np.allclose(motion_seq, ms2)
    # * check (3): Zero motion
    static_example = generate_static_json()
    # json.dump(static_example, static_json_)
    ms_0 = compute_motion_sequence(generate_static_json()['frames'])
    # print("Zero motion test:", ms_0)
    print(f"\n* Static motion tensor: mean = {ms_0.mean()}, max = {abs(ms_0.max())}, sum = {ms_0.sum()}\n")
    assert ms_0.max() < eps
    controlled_motion_test()
    crowd_compression_test()

    pass


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == '__main__':
    json_example = "/mnt/local-data/Projects/Wesmart/data/usual_jsons_from_events/event_18.json"
    static_json_ = "/mnt/local-data/Python/Projects/weSmart/data/json_data/static_clip.json"
    eps = 1e-5
    test_motion_sequence(json_example,eps)

    # #* load the example file
    # with open(json_example, 'r') as f:
    #     data = json.load(f)
    # frames = data['frames']
    # # frames = load_json_frames(json_example['frames'])
    # motion_seq = compute_motion_sequence(frames)
    # #* check (1): shape is stable
    # print(motion_seq.shape)
    # #* check (2): calculation  consistency
    # ms2 = compute_motion_sequence(frames)
    # assert np.allclose(motion_seq, ms2)
    # # * check (3): Zero motion
    # static_example =generate_static_json()
    # # json.dump(static_example, static_json_)
    # print("Zero motion test:",compute_motion_sequence(generate_static_json()['frames']) )
    # assert motion_seq.max() < eps
    # controlled_motion_test()

    pass

