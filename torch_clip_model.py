""" Train and test a clip-level MLP classifier from cached NPZ features.
    Expected cache format:
    - `X`: float array with shape (N, C)
    - `y`: int labels with shape (N,), values in {0, 1}
    - `meta`: clip metadata (necessary for video-wise testing)
    Main API:
    - run_training(...) trains and saves model/config/log/TensorBoard files.
    - run_testing(...) runs inference and saves raw predictions to NPZ.
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

#* Project import
from common.my_local_utils import print_color
# from evaluation_core import analyze_clip_test, analyze_video_test
from evaluation_core import analyze_clip_test, analyze_video_test
from evaluation_cli import print_test_report
from stream_analysis import analyze_stream_test
from motion_feature_schema import (
    assert_feature_schema_match,
    load_cache_contract_compact,
    schema_has_na,
    temporal_schema_compatible,
)

#* config constants ToDo: make proper config file
DEFAULT_WORKDIR = "work_dirs/json_models"
LOCAL_CONFIG    = "config.json"
LOCAL_LOG       = "log.json"

DEFAULT_VALID_RATIO = 0.85
DEFAULT_VALID_SEED  = 42
DEFAULT_BATCH_SIZE  = 256
#* training parameters
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN_DIM  = 64
#* epochs
DEFAULT_MIN_EPOCHS = 30
DEFAULT_MAX_EPOCHS = 150
DEFAULT_SAVE_EVERY = 10
#*
DEFAULT_PATIENCE = 30
DEFAULT_MIN_DELTA = 0.002
DEFAULT_WINDOW_TOLERANCE = 0.25
DEFAULT_STRIDE_TOLERANCE = 0.25

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------
#* Dataset
# --------------------------------------------------
class ClipFeatureDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        self.X = data['X'].astype(np.float32)
        self.y = data['y'].astype(np.int64)
        assert len(self.X) == len(self.y), 'X/y size mismatch'

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]),
                torch.tensor(self.y[idx], dtype=torch.float32),)

# --------------------------------------------------
# * Minimal classifier
# --------------------------------------------------

class ClipMLP(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, 1),)
    def forward(self, x):
        return self.net(x).squeeze(1)

# * Local helpers  --------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(DEFAULT_DEVICE)
        y = y.to(DEFAULT_DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*x.size(0)

    return total_loss/len(loader.dataset)


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEFAULT_DEVICE)
            y = y.to(DEFAULT_DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()*x.size(0)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return total_loss/len(loader.dataset), preds, targets


def _infer_hidden_dim_from_state(state: dict) -> int:
    """Infer the hidden layer width from one saved MLP state dict."""
    weight = state.get('net.0.weight', None)
    if weight is None or getattr(weight, 'ndim', None) != 2:
        raise ValueError("Could not infer hidden_dim from model state")
    return int(weight.shape[0])


def _labels_from_dataset(ds) -> np.ndarray:
    """ Return label array for ClipFeatureDataset or Subset(ClipFeatureDataset)."""
    if hasattr(ds, 'y'):
        return ds.y
    if isinstance(ds, Subset) and hasattr(ds.dataset, 'y'):
        idx = np.asarray(ds.indices, dtype=np.int64)
        return ds.dataset.y[idx]
    raise ValueError("Unsupported dataset type for class-balance computation")


def _binary_auc(y_true, y_score):
    """ Compute ROC-AUC from binary labels and scores without extra dependencies. """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            mean_rank = np.mean(ranks[order[i:j]])
            ranks[order[i:j]] = mean_rank
        i = j

    pos_ranks = ranks[y_true == 1]
    return (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _training_contract_from_cache(train_cache: str | Path, **kwargs) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
    """Load and validate the canonical feature/temporal contract for one train cache."""
    contract, used_legacy_fallback = load_cache_contract_compact(train_cache)
    train_caches = [dict(item) for item in contract["source_caches"]]
    if not train_caches:
        raise ValueError(f"{train_cache} did not provide any source cache provenance")

    canonical_feature_schema = dict(contract["feature_schema"])
    canonical_temporal_schema = dict(train_caches[0]["temporal_schema"])
    window_tolerance = float(kwargs.get("window_tolerance", DEFAULT_WINDOW_TOLERANCE))
    stride_tolerance = float(kwargs.get("stride_tolerance", DEFAULT_STRIDE_TOLERANCE))
    if used_legacy_fallback:
        print_color(
            f"[WARN] {train_cache} is missing cache metadata; training will continue with 'N/A' contract fields.",
            "o",
        )

    for index, cache_rec in enumerate(train_caches):
        feature_schema = dict(cache_rec["feature_schema"])
        temporal_schema = dict(cache_rec["temporal_schema"])
        if schema_has_na(canonical_feature_schema) or schema_has_na(feature_schema):
            if index > 0:
                print_color(
                    f"[WARN] Skipping strict feature-schema validation for {train_cache} source_caches[{index}] because metadata is incomplete.",
                    "o",
                )
        else:
            assert_feature_schema_match(canonical_feature_schema,
                                        feature_schema,
                                        context=f"{train_cache} source_caches[{index}]")
        if schema_has_na(canonical_temporal_schema) or schema_has_na(temporal_schema):
            if index > 0:
                print_color(
                    f"[WARN] Skipping strict temporal-schema validation for {train_cache} source_caches[{index}] because metadata is incomplete.",
                    "o",
                )
        elif not temporal_schema_compatible(canonical_temporal_schema,
                                            temporal_schema,
                                            window_tolerance=window_tolerance,
                                            stride_tolerance=stride_tolerance):
            raise ValueError(
                f"Temporal schema mismatch for {train_cache} source_caches[{index}]: "
                f"expected target around window={canonical_temporal_schema['window']} stride={canonical_temporal_schema['stride']}, "
                f"got window={temporal_schema['window']} stride={temporal_schema['stride']}"
            )

    temporal_profile = {
        "target_window": canonical_temporal_schema["window"],
        "target_stride": canonical_temporal_schema["stride"],
        "window_tolerance": window_tolerance,
        "stride_tolerance": stride_tolerance,
    }
    return train_caches, canonical_feature_schema, temporal_profile

# --------------------------------------------------
# * main/ API functions
# --------------------------------------------------

def run_training(train_cache:str|Path, valid_cache:str|Path|None=None, **kwargs):
    """ Train a clip classifier from cached features and save run artifacts.
        Path to the run directory that contains `model.pt`, `config.json`, `log.json`,
        and TensorBoard event files.
        :param train_cache  : path to train cache NPZ.
        :param valid_cache  : optional validation cache NPZ. If omitted, runtime split is applied.
        :param kwargs:
            work_dir, tag   : directory for output files and their tag, by default
                              both work dir & tag are generated from the cache and model properties
            lr, batch_size, hidden_dim, save_every :
                             Training params, if passed they overwrite the defaults/config settings
            max_epochs, patience, min_delta :
                             E"arly-stop settings; `epochs` is kept as alias for `max_epochs`
            valid_ratio, valid_seed : ratio and seed for runtime train/valid split
        :return:              path for the effective work_dir
    """

    #* Paths and runtime parameters.
    train_npz = Path(train_cache)
    valid_npz = Path(valid_cache) if valid_cache is not None else None
    work_dir = Path(kwargs.get('work_dir', DEFAULT_WORKDIR))
    run_dir = work_dir/f"{datetime.now().strftime('%y%m%d_%H-%M-%S')}_{kwargs.get('tag', train_npz.stem)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    lr = kwargs.get('lr', DEFAULT_LR)
    batch_size  = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
    hidden_dim  = kwargs.get('hidden_dim', DEFAULT_HIDDEN_DIM)
    save_every  = kwargs.get('save_every', DEFAULT_SAVE_EVERY)
    max_epochs = kwargs.get('max_epochs', kwargs.get('epochs', DEFAULT_MAX_EPOCHS))
    min_epochs = DEFAULT_MIN_EPOCHS
    patience = kwargs.get('patience', DEFAULT_PATIENCE)
    min_delta = kwargs.get('min_delta', DEFAULT_MIN_DELTA)

    if max_epochs < 1:
        raise ValueError(f"Invalid max_epochs={max_epochs}. Expected positive integer.")
    min_epochs = min(min_epochs, max_epochs)
    train_caches, feature_schema, temporal_profile = _training_contract_from_cache(train_npz, **kwargs)

    # Build train/valid datasets;
    full_train_ds = ClipFeatureDataset(train_npz)
    if valid_npz is not None:
        train_ds = full_train_ds
        valid_ds = ClipFeatureDataset(valid_npz)
        valid_used = str(valid_npz)
    else:
        #* split train cache only when valid cache is not given.
        valid_ratio = kwargs.get('valid_ratio', DEFAULT_VALID_RATIO)
        valid_seed = kwargs.get('valid_seed', DEFAULT_VALID_SEED)
        if not (0.0 < valid_ratio < 1.0): # or len(full_train_ds) < 2
            raise ValueError(f" Invalid valid_ratio= {valid_ratio}. Expected value in (0, 1).")
        n_total = len(full_train_ds)
        n_train = int( max(1, np.floor(n_total*valid_ratio)) )
        n_valid = n_total - n_train
        gen = torch.Generator().manual_seed(valid_seed)
        train_ds, valid_ds = random_split(full_train_ds, [n_train, n_valid], generator=gen)
        valid_used =  f"Runtime valid split: ratio{valid_ratio}, seed{valid_seed}"

    #* Save lightweight run config for later testing/evaluation.
    run_cfg = {'train_cache': str(train_npz), 'valid_cache': valid_used, #* datasets configs
               'train_caches': train_caches,
               'feature_schema': feature_schema,
               'temporal_profile': temporal_profile,
               'min_epochs': min_epochs, 'max_epochs': max_epochs,  #* epoch related configs
               'save_every': save_every,
               'patience': patience, 'min_delta': min_delta,  #* early stop condition
               'batch_size': batch_size, 'lr':lr, 'hidden_dim':hidden_dim, #* Training configs
               }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    tb_writer = SummaryWriter(log_dir=str(run_dir))

    sample_x, _ = train_ds[0]
    model = ClipMLP(int(sample_x.numel()), hidden_dim=hidden_dim).to(DEFAULT_DEVICE)

    #* Positive class weight for BCE (neg/pos) to reduce imbalance bias.
    pos_weight = None
    train_y = _labels_from_dataset(train_ds)
    n_pos = (train_y == 1).sum()
    n_neg = (train_y == 0).sum()
    if n_pos > 0:
        pos_weight = torch.tensor(n_neg/max(n_pos, 1), device=DEFAULT_DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = None
    best_epoch = 0
    best_metric = 'val_auc'
    best_path = None
    stale_epochs = 0
    train_log = []
    for epoch in range(1, max_epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_probs, val_targets = eval_one_epoch(model, valid_loader, criterion)
        val_auc = _binary_auc(val_targets.numpy(), val_probs.numpy())
        monitor_score = val_auc if val_auc is not None else -val_loss
        if val_auc is None:
            best_metric = '-val_loss'

        is_better = best_score is None or monitor_score > best_score + min_delta
        if is_better:
            best_score = monitor_score
            best_epoch = epoch
            stale_epochs = 0
            if best_path is not None and best_path.is_file():
                best_path.unlink()
            best_path = run_dir/f"best_model.{epoch:03d}.pt"
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1

        train_log += [{'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                       'val_auc': val_auc, 'best_epoch': best_epoch, 'best_score': best_score, }]
        tb_writer.add_scalar('loss/train', train_loss, epoch)
        tb_writer.add_scalar('loss/valid', val_loss, epoch)
        if val_auc is not None:
            tb_writer.add_scalar('auc/valid', val_auc, epoch)

        if save_every and epoch % save_every == 0:
            torch.save(model.state_dict(), run_dir/f"checkpoint_ep-{epoch:03d}.pt")

        auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val auc: {auc_str}")

        if patience and epoch >= min_epochs and stale_epochs >= patience:
            print(f"Early stop at epoch {epoch:03d} | best {best_metric}: {best_score:.4f} at epoch {best_epoch:03d}")
            break

    #* Save model and run logs.
    if best_path is not None and best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=DEFAULT_DEVICE), strict=True)
    torch.save(model.state_dict(), run_dir/'model.pt')

    with open(run_dir/LOCAL_LOG, 'w') as f:
        json.dump(train_log, f, indent=2)
    
    with open(run_dir/LOCAL_CONFIG, 'w') as f:
        json.dump(run_cfg, f, indent=2)
    tb_writer.close()

    print(f"Training complete\n Files saved to {run_dir} ")

    return run_dir    # return model, train_log


def run_testing(test_model:str|Path, test_cache:str|Path, vid_info=False, video_mode=False, **kwargs):
    """Run model inference on a cache NPZ and save raw prediction arrays.
        By default, the saved NPZ uses one unified format that includes any available
        grouping/timing metadata needed for clip/video/stream analysis.
        `pure_clips=True` saves one minimal clip-only payload for batch throughput.
        parameters:
        :param test_cache : path to test cache NPZ.
        :param test_model : path to `model.pt`
        :param vid_info   : legacy no-op flag kept for compatibility.
        :param video_mode : legacy no-op flag kept for compatibility.
        :param kwargs     : batch_size, out_dir, out_name, pure_clips
        :return           : Dict with saved `path` and in-memory prediction arrays.
    """
    model_path = Path(test_model)
    test_npz = Path(test_cache)

    batch_size = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)

    state = torch.load(model_path, map_location=DEFAULT_DEVICE)
    # Rebuild model shape from the training config saved with the checkpoint.
    cfg_path = model_path.parent/LOCAL_CONFIG
    cfg = {}
    hidden_dim = None
    if cfg_path.is_file():
        try:
            with cfg_path.open('r') as f:
                cfg = json.load(f)
            hidden_dim = int(cfg['hidden_dim'])
        except Exception as e:
            raise ValueError(f"Invalid hidden_dim value in run_config.json: {cfg.get('hidden_dim')}") from e
    else:
        print_color(f"{LOCAL_CONFIG} is missing, inferring hidden_dim from model state", 'o')
        hidden_dim = _infer_hidden_dim_from_state(state)

    test_ds = ClipFeatureDataset(test_npz)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    sample_x, _ = test_ds[0]
    model = ClipMLP(int(sample_x.numel()), hidden_dim=hidden_dim).to(DEFAULT_DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    probs_all, y_true_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEFAULT_DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            y_true_all.append(y.numpy().astype(np.int64))

    y_prob = np.concatenate(probs_all, axis=0)
    y_true = np.concatenate(y_true_all, axis=0)

    y_true = y_true.astype(np.int64)

    # Save raw per-clip predictions.
    # out_name = kwargs.get('out_name', f"{model_path.stem}_{test_npz.stem}-test.npz")
    out_name = kwargs.get('output_tag', f"{model_path.stem}_{test_npz.stem}-tst.npz")
    out_path = Path(kwargs.get('out_dir', model_path.parent))/str(out_name)
    out_path = out_path.with_suffix('.npz') if out_path.suffix.lower() != '.npz' else out_path

    save_payload = {'model_path': str(model_path), 'test_cache': str(test_npz),
                    'y_true': y_true, 'y_prob': y_prob,
                    'cache_index': np.arange(len(y_true), dtype=np.int64)}
    pure_clips = bool(kwargs.get('pure_clips', False))

    test_data = np.load(test_npz, allow_pickle=True)
    has_meta = 'meta' in test_data.files
    if has_meta and not pure_clips:
        meta = test_data['meta']
        if len(meta) != len(y_true):
            raise ValueError(f"meta length mismatch: {len(meta)} vs {len(y_true)} predictions")
        meta_video = np.asarray([Path(item['video']).stem for item in meta], dtype=str)
        meta_t_start = np.asarray([item['t_start'] for item in meta], dtype=np.float32)
        meta_t_end = np.asarray([item['t_end'] for item in meta], dtype=np.float32)
        meta_n_frames = np.asarray([int(item.get('n_frames', -1)) if isinstance(item, dict) else -1
                                    for item in meta], dtype=np.int64)

        save_payload['meta_video'] = meta_video
        save_payload['meta_t_start'] = meta_t_start
        save_payload['meta_t_end'] = meta_t_end
        save_payload['meta_n_frames'] = meta_n_frames
        save_payload['video_name'] = meta_video
        save_payload['time_stamp'] = meta_t_end

    np.savez_compressed(out_path, **save_payload)

    print(f"\n=== Testing run complete ===\n"
          f"\tTested model : {test_model}\n"
          f"\tTested set   : {test_cache}\n"
          f"\tPredictions npz: {out_path.name}\n")
    return {'path': str(out_path), **save_payload}


def run_stream_testing(test_model:str|Path, X:np.ndarray, y:np.ndarray, meta:np.ndarray,
                       stream_name:str, **kwargs):
    """Run model inference directly on extracted stream features."""
    model_path = Path(test_model)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or not len(X):
        raise ValueError(f"No stream feature rows available for {stream_name}")
    if len(X) != len(y) or len(meta) != len(y):
        raise ValueError("Stream X/y/meta size mismatch")

    state = torch.load(model_path, map_location=DEFAULT_DEVICE)
    cfg_path = model_path.parent/LOCAL_CONFIG
    if cfg_path.is_file():
        with cfg_path.open('r', encoding='utf-8') as f:
            hidden_dim = int(json.load(f)['hidden_dim'])
    else:
        hidden_dim = _infer_hidden_dim_from_state(state)

    model = ClipMLP(X.shape[1], hidden_dim=hidden_dim).to(DEFAULT_DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float()),
        batch_size=kwargs.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=False)

    probs = []
    with torch.no_grad():
        for batch_x, _ in loader:
            probs.append(torch.sigmoid(model(batch_x.to(DEFAULT_DEVICE))).cpu().numpy())
    y_prob = np.concatenate(probs, axis=0)

    meta_video = np.asarray([Path(item['video']).stem for item in meta], dtype=str)
    meta_t_start = np.asarray([item['t_start'] for item in meta], dtype=np.float32)
    meta_t_end = np.asarray([item['t_end'] for item in meta], dtype=np.float32)
    meta_n_frames = np.asarray([int(item.get('n_frames', -1)) for item in meta], dtype=np.int64)
    save_payload = {
        'model_path': str(model_path),
        'test_cache': stream_name,
        'y_true': y,
        'y_prob': y_prob,
        'cache_index': np.arange(len(y), dtype=np.int64),
        'meta_video': meta_video,
        'meta_t_start': meta_t_start,
        'meta_t_end': meta_t_end,
        'meta_n_frames': meta_n_frames,
        'video_name': meta_video,
        'time_stamp': meta_t_end,
    }
    out_name = kwargs.get('output_tag', f"{model_path.stem}_{Path(stream_name).stem}-tst.npz")
    out_path = Path(kwargs.get('out_dir', model_path.parent))/str(out_name)
    out_path = out_path.with_suffix('.npz') if out_path.suffix.lower() != '.npz' else out_path
    np.savez_compressed(out_path, **save_payload)

    print(f"\n=== Stream testing run complete ===\n"
          f"\tTested model : {model_path}\n"
          f"\tTested stream: {stream_name}\n"
          f"\tPredictions npz: {out_path.name}\n")
    return {'path': str(out_path), **save_payload}

# --------------------------------------------------
#* Training scripts and unit testing
# --------------------------------------------------

def test_test(test_cache:str|Path, test_model:str|Path, **kwargs):
    """ Small helper that tests the testing tools"""
    #* return run_testing(test_cache, tst_model, **kwargs)
    # res = run_testing(test_cache, test_model, **kwargs)
    res = run_testing(test_model, test_cache, **kwargs)
    if res is None: return
    eval_mode = kwargs.get('eval_mode', None)
    if eval_mode is None:
        eval_mode = 'video' if kwargs.get('video_mode', False) else 'clip'

    if eval_mode == 'clip':
        report = analyze_clip_test(res['path'], show_roc=kwargs.get('show', False))
    elif eval_mode == 'video':
        report = analyze_video_test(res['path'], show_roc=kwargs.get('show', False))
    else:
        report = analyze_stream_test(res['path'], show_roc=kwargs.get('show', False))
    print_test_report(report)

#*
def train_rwd_n_rlvs():
    """ Example script: train/test on RWF and RLVS caches separately."""
    d = Path("data/cache/")
    #* train on RWF data
    output_path = run_training(d/"RWF_train.npz", tag="TMS-18f_RW", valid_ratio=0.85, valid_seed=21)
    #* Test on RWF test-set
    res = run_testing(d/'RWF_test.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_clip_test(res['path'], show_roc=True, print=True)
    #* Test on RLVS train-set
    res = run_testing(d/'RLVS_train.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_clip_test(res['path'], show_roc=True)

    #* train on RLVS data
    output_path = run_training(d/"RLVS_train.npz", tag="TMS-18f_RLVS", valid_ratio=0.85, valid_seed=21)
    #* Test on RLVS test-set
    res = run_testing(d/'RLVS_test.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_clip_test(res['path'], show_roc=True, print=True)
    # * Test on RLVS train-set
    res = run_testing(d/'RWF_train.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_clip_test(res['path'], show_roc=True)


def train_joint():
    """Example script: merge RWF+RLVS caches, then train/test a joint model."""
    from  precompute_clips import merge_cache_npz, cache_info
    d = Path("data/cache/")
    #* train on RWF data
    tr_1, tst_1 = d/"RWF_train.npz",  d/"RWF_test.npz"
    tr_2, tst_2 = d/"RLVS_train.npz", d/"RLVS_test.npz"
    tr_j, tst_j = d/"Joint_RWFLV_train.npz", d/"Joint_RWFLV_test.npz"

    print(f"all data sets exists {tr_j.is_file() and tst_1.is_file() and tr_2.is_file() and tst_2.is_file()}")

    merge_cache_npz([tr_1 , tr_2 ], tr_j)
    merge_cache_npz([tst_1, tst_2], tst_j)
    # cache_info(tr_1)
    # cache_info(tr_2)
    cache_info(tr_j)
    output_path = run_training(tr_j, tag="TMS-18f_Jn", valid_ratio=0.85, valid_seed=42)
    #* Test on RWF test-set
    res = run_testing(tst_j, output_path/'model.pt')
    if res is not None:
        report = analyze_clip_test(res['path'], show_roc=True, print=True)


if __name__ == '__main__':
    pass
    # Example:
    # model, hist = run_training('data/cache/RWF_train.npz', 'data/cache/RWF_valid.npz')
    # model, hist = run_training('data/cache/RWF_train.npz', valid_ratio=0.85, valid_seed=42)

    # train_rwd_n_rlvs()
    # train_joint()
    tst_mdl = "work_dirs/json_models/260331-0233_J-RWFLV-25ft/best_model.085.pt"
    tst_mdl = "work_dirs/json_models/draft/260331-0233_J-RWFLV-25ft/best_model.085.pt"
    tst_ch =  "data/cache/J_RWFLV_25ft_test.npz"

    test_test(test_model=tst_mdl, test_cache=tst_ch, out_name='tvt_J25ft-4v',vid_info=True)

# 318(2,4,2)-> 300(2,,)
