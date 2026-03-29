""" Train and test a clip-level MLP classifier from cached NPZ features.
    Expected cache format:
    - `X`: float array with shape (N, C)
    - `y`: int labels with shape (N,), values in {0, 1}
    - `meta`: clip metadata (necessary for video-wise testing)
    Main API:
    - run_training(...) trains and saves model/config/log files.
    - run_testing(...) runs inference and saves raw predictions to NPZ.
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split

#* Project import
from common.my_local_utils import print_color
from evaluation_tools import analyze_test_results, print_test_report, plot_roc_curve

#* config constants ToDo: make proper config file
DEFAULT_WORKDIR = "work_dirs/json_models"
LOCAL_CONFIG    = "config.json"
LOCAL_LOG       = "log.json"

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN_DIM  = 64
DEFAULT_SAVE_EVERY = 10
DEFAULT_SPLIT_RATIO = 0.85
DEFAULT_SPLIT_SEED  = 42
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
                                 nn.Linear(hidden_dim, 1),
                                 )
    def forward(self, x):
        return self.net(x).squeeze(1)

# * Local helpers  --------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*x.size(0)

    return total_loss/len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()*x.size(0)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return total_loss/len(loader.dataset), preds, targets


def _labels_from_dataset(ds) -> np.ndarray:
    """Return label array for ClipFeatureDataset or Subset(ClipFeatureDataset)."""
    if hasattr(ds, 'y'):
        return ds.y
    if isinstance(ds, Subset) and hasattr(ds.dataset, 'y'):
        idx = np.asarray(ds.indices, dtype=np.int64)
        return ds.dataset.y[idx]
    raise ValueError("Unsupported dataset type for class-balance computation")

# --------------------------------------------------
# * main/ API functions
# --------------------------------------------------

def run_training(train_cache:str|Path, valid_cache:str|Path|None=None, **kwargs):
    """ Train a clip classifier from cached features and save run artifacts.
        Path to the run directory that contains `model.pt`, `config.json`, and `log.json`.
        :param train_cache  : path to train cache NPZ.
        :param valid_cache  : optional validation cache NPZ. If omitted, runtime split is applied.
        :param kwargs:
            work_dir, tag   : directory for output files and their tag, by default
                              both work dir & tag are generated from the cache and model properties
            lr, epochs, batch_size, hidden_dim, save_every :
                             Training params, if passed they overwrite the defaults/config settings
            split_ratio, split_seed : ratio and seed for runtime splitting (if relevant)
        :return:              path for the effective work_dir
    """

    #* Paths and runtime parameters.
    train_npz = Path(train_cache)
    valid_npz = Path(valid_cache) if valid_cache is not None else None
    work_dir = Path(kwargs.get('work_dir', DEFAULT_WORKDIR))
    run_dir = work_dir/f"{datetime.now().strftime('%y%m%d-%H%M')}_{kwargs.get('tag','')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    lr = kwargs.get('lr', DEFAULT_LR)
    epochs = kwargs.get('epochs', DEFAULT_EPOCHS)
    batch_size  = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
    hidden_dim  = kwargs.get('hidden_dim', DEFAULT_HIDDEN_DIM)
    save_every  = kwargs.get('save_every', DEFAULT_SAVE_EVERY)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build train/valid datasets;
    full_train_ds = ClipFeatureDataset(train_npz)
    if valid_npz is not None:
        train_ds = full_train_ds
        valid_ds = ClipFeatureDataset(valid_npz)
        valid_used = str(valid_npz)
    else:
        #* split train cache only when valid cache is not given.
        split_ratio = kwargs.get('split_ratio', DEFAULT_SPLIT_RATIO)
        split_seed = kwargs.get('split_seed', DEFAULT_SPLIT_SEED)
        if not (0.0 < split_ratio < 1.0): # or len(full_train_ds) < 2
            raise ValueError(f" Invalid split_ratio= {split_ratio}. Expected value in (0, 1).")
        n_total = len(full_train_ds)
        n_train = int( max(1, np.floor(n_total*split_ratio)) )
        n_valid = n_total - n_train
        gen = torch.Generator().manual_seed(split_seed)
        train_ds, valid_ds = random_split(full_train_ds, [n_train, n_valid], generator=gen)
        valid_used =  f"Runtime split: ratio{split_ratio}, seed{split_seed}"

    #* Save lightweight run config for later testing/evaluation.
    run_cfg = {'batch_size': batch_size, 'epochs': epochs,
               'lr': lr, 'hidden_dim' : hidden_dim,
               'save_every': save_every,
               'train_cache': str(train_npz),
               'valid_cache': valid_used, }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    sample_x, _ = train_ds[0]
    model = ClipMLP(int(sample_x.numel()), hidden_dim=hidden_dim).to(device)

    #* Positive class weight for BCE (neg/pos) to reduce imbalance bias.
    pos_weight = None
    train_y = _labels_from_dataset(train_ds)
    n_pos = (train_y == 1).sum()
    n_neg = (train_y == 0).sum()
    if n_pos > 0:
        pos_weight = torch.tensor(n_neg/max(n_pos, 1), device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_log = []
    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = eval_one_epoch(model, valid_loader, criterion, device)
        train_log += [{ 'epoch': epoch, 'train_loss': train_loss, 'val_loss':val_loss }]

        if save_every and epoch % save_every == 0:
            torch.save(model.state_dict(), run_dir/f"model_ep{epoch:03d}.pt")

        print(f'Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}')

    #* Save model and run logs.
    torch.save(model.state_dict(), run_dir/'model.pt')

    with open(run_dir/LOCAL_LOG, 'w') as f:
        json.dump(train_log, f, indent=2)
    
    with open(run_dir/LOCAL_CONFIG, 'w') as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Training complete\n Files saved to {run_dir} ")

    return run_dir    # return model, train_log


def run_testing(test_cache:str|Path, test_model: str | Path, **kwargs):
    """     Run model inference on a cache NPZ and save predictions NPZ file.
        NPZ file stores: `cache_index`, `y_true`, `y_pred`, `y_prob`.
        model setting are loaded from config.json (file
        parameters:
        :param test_cache: path to test cache NPZ.
        :param test_model: path to `model.pt` produced by `run_training`.
        :param kwargs    :  batch_size, threshold, out_dir, out_name.
        :return:    Dict with saved `path` and in-memory prediction arrays.
    """
    test_npz = Path(test_cache)
    model_path = Path(test_model)

    batch_size = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
    threshold = float(kwargs.get('threshold', 0.5))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rebuild model shape from the training config saved with the checkpoint.
    cfg_path = model_path.parent/LOCAL_CONFIG
    if not cfg_path.is_file():
        # raise FileNotFoundError(f"{LOCAL_CONFIG} is missing")
        print_color(f"{LOCAL_CONFIG} is missing",'o')
        return None
    try:
        with cfg_path.open('r') as f:
            cfg = json.load(f)
        hidden_dim = int(cfg['hidden_dim'])
    except Exception as e:
        raise ValueError(f"Invalid hidden_dim value in run_config.json: {cfg.get('hidden_dim')}") from e

    state = torch.load(model_path, map_location=device)
    test_ds = ClipFeatureDataset(test_npz)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    sample_x, _ = test_ds[0]
    model = ClipMLP(int(sample_x.numel()), hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    probs_all, y_true_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            y_true_all.append(y.numpy().astype(np.int64))

    y_prob = np.concatenate(probs_all, axis=0)
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cache_index = np.arange(len(y_true), dtype=np.int64)
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    # Save raw per-clip predictions.
    out_name = kwargs.get('out_name', f"{model_path.stem}_{test_npz.stem}-test.npz")
    out_path = Path(kwargs.get('out_dir', model_path.parent))/str(out_name)
    out_path = out_path.with_suffix('.npz') if out_path.suffix.lower() != '.npz' else out_path

    save_payload = {'cache_index':cache_index, 'y_true':y_true, 'y_pred':y_pred, 'y_prob':y_prob}
    np.savez_compressed(out_path, **save_payload)

    print(f'Testing run complete\npredictions saved to "{out_path}" ')
    return {'path': str(out_path), **save_payload}

# --------------------------------------------------
#* Training scripts and unit testing
# --------------------------------------------------

def test_test(test_cache: str | Path, test_model: str | Path, **kwargs):
    """ Small helper that tests the testing tools"""
    #* return run_testing(test_cache, tst_model, **kwargs)
    res = run_testing(test_cache, test_model, **kwargs)
    if res is not None:
        report = analyze_test_results(res['path'], show_roc=kwargs.get('show',False))
        print_test_report(report)

#*
def train_rwd_n_rlvs():
    """ Example script: train/test on RWF and RLVS caches separately."""
    d = Path("data/cache/")
    #* train on RWF data
    output_path = run_training(d/"RWF_train.npz", tag="TMS-18f_RW", split_ratio=0.85, split_seed=21)
    #* Test on RWF test-set
    res = run_testing(d/'RWF_test.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_test_results(res['path'], show_roc=True, print=True)
    #* Test on RLVS train-set
    res = run_testing(d/'RLVS_train.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_test_results(res['path'], show_roc=True)

    #* train on RLVS data
    output_path = run_training(d/"RLVS_train.npz", tag="TMS-18f_RLVS", split_ratio=0.85, split_seed=21)
    #* Test on RLVS test-set
    res = run_testing(d/'RLVS_test.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_test_results(res['path'], show_roc=True, print=True)
    # * Test on RLVS train-set
    res = run_testing(d/'RWF_train.npz', output_path/'model.pt')
    if res is not None:
        report = analyze_test_results(res['path'], show_roc=True)


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
    output_path = run_training(tr_j, tag="TMS-18f_Jn", split_ratio=0.85, split_seed=42)
    #* Test on RWF test-set
    res = run_testing(tst_j, output_path/'model.pt')
    if res is not None:
        report = analyze_test_results(res['path'], show_roc=True, print=True)


if __name__ == '__main__':
    pass
    # Example:
    # model, hist = run_training('data/cache/RWF_train.npz', 'data/cache/RWF_valid.npz')
    # model, hist = run_training('data/cache/RWF_train.npz', split_ratio=0.85, split_seed=42)

    # train_rwd_n_rlvs()
    train_joint()

# 318(2,4,2)-> 300(2,,)
