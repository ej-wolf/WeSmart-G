"""
    Minimal PyTorch Dataset + classifier for cached clip-level features.
    Assumes cached .npz files created by `precompute_clip_features` with keys:
    - 'X'    : (N, C) float features
    - 'y'    : (N, ) int labels {0,1}
    - 'meta' : optional metadata (ignored here)
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split

#* config constants ToDo: make proper config file
DEFAULT_WORKDIR = "work_dirs/json_models"
LOCAL_CONFIG    = "run_config.json"
LOCAL_LOG       = "log.json"

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN_DIM  = 64
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

# --------------------------------------------------
# * Training / evaluation utilities
# --------------------------------------------------

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
    """Return labels array for Dataset or Subset(ClipFeatureDataset)."""
    if hasattr(ds, 'y'):
        return ds.y
    if isinstance(ds, Subset) and hasattr(ds.dataset, 'y'):
        idx = np.asarray(ds.indices, dtype=np.int64)
        return ds.dataset.y[idx]
    raise ValueError("Unsupported dataset type for class-balance computation")


def run_training(train_cache:str|Path, valid_cache:str|Path|None=None, **kwargs):
    """ Run training on cached clip-level features and save artifacts.
        If `valid_cache` is None, split `train_cache` into train/valid for this run only.
    """

    #* Argument normalization:
    #* dirs and files
    train_npz = Path(train_cache)
    valid_npz = Path(valid_cache) if valid_cache is not None else None
    work_dir = Path(kwargs.get('work_dir', DEFAULT_WORKDIR))
    run_dir = work_dir/f"train_{datetime.now().strftime('%y%m%d-%H%M')}_{kwargs.get('tag','')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    #* training params
    lr = kwargs.get('lr', DEFAULT_LR)
    epochs = kwargs.get('epochs', DEFAULT_EPOCHS)
    batch_size  = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
    hidden_dim  = kwargs.get('hidden_dim', DEFAULT_HIDDEN_DIM)
    split_ratio = kwargs.get('split_ratio', DEFAULT_SPLIT_RATIO)
    split_seed  = kwargs.get('split_seed', DEFAULT_SPLIT_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #* # datasets & loaders
    full_train_ds = ClipFeatureDataset(train_npz)
    if valid_npz is not None:
        train_ds = full_train_ds
        valid_ds = ClipFeatureDataset(valid_npz)
        # split_used = 'external_valid_cache'
    else:
        if not (0.0 < split_ratio < 1.0): # or len(full_train_ds) < 2
            raise ValueError(f" Invalid split_ratio= {split_ratio}. Expected value in (0, 1).")
        n_total = len(full_train_ds)
        # n_train = int(round(n_total * split_ratio))
        # n_train = max(1, min(n_total - 1, n_train))
        n_train = int( max(1, np.floor(n_total*split_ratio)) )
        n_valid = n_total - n_train
        gen = torch.Generator().manual_seed(split_seed)
        train_ds, valid_ds = random_split(full_train_ds, [n_train, n_valid], generator=gen)
        # split_used = 'runtime_random_split'

    # save run config (lightweight)
    run_cfg = {'batch_size': batch_size, 'epochs': epochs,
               'lr': lr, 'hidden_dim' : hidden_dim,
               'train_cache': str(train_npz),
               'valid_cache': str(valid_npz) if valid_npz is not None else f"Runtime split: ratio{split_ratio}, seed{split_seed}",
               }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    sample_x, _ = train_ds[0]
    model = ClipMLP(int(sample_x.numel()), hidden_dim=hidden_dim).to(device)

    #* handle class imbalance
    pos_weight = None
    train_y = _labels_from_dataset(train_ds)
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    if n_pos > 0:
        pos_weight = torch.tensor(n_neg/max(n_pos, 1), device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #* training loop
    train_log = []
    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = eval_one_epoch(model, valid_loader, criterion, device)
        train_log += [{ 'epoch': epoch, 'train_loss': train_loss, 'val_loss':val_loss }]
        
        print(f'Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}')

    #* save files 
    torch.save(model.state_dict(), run_dir/'model.pt')
    
    with open(run_dir/LOCAL_LOG, 'w') as f:
        json.dump(train_log, f, indent=2)
    
    with open(run_dir/LOCAL_CONFIG, 'w') as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Training complete\n Files saved to {run_dir} ")

    return model, train_log


def run_testing(test_cache:str|Path, tst_model: str|Path, **kwargs):
    """ Run model inference on a cached NPZ file and save raw per-clip results as NPZ."""
    test_npz = Path(test_cache)
    model_path = Path(tst_model)
    # work_ dir = Path(kwargs.get('work_ dir', DEFAULT_WORKDIR))
    # work_ dir.mkdir(parents=True, exist_ok=True)

    batch_size = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
    threshold = float(kwargs.get('threshold', 0.5))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = kwargs.get('device', DEFAULT_DEVICE)
    # if not isinstance(device, torch.device):
    #     device = torch.device(device)

    #* Resolve hidden_dim strictly from run_config.json.
    cfg_path = model_path.parent/LOCAL_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{LOCAL_CONFIG} is missing")
    try:
        with cfg_path.open('r') as f:
            cfg = json.load(f)
        hidden_dim = int(cfg['hidden_dim'])
    except Exception as e:
        raise ValueError(f"Invalid hidden_dim value in run_config.json: {cfg.get('hidden_dim')}") from e

    # if not cfg_path.is_file():
    #     raise FileNotFoundError(f"run_config.json not found near model: {cfg_path}")
    # try:
    #     with cfg_path.open('r') as f:
    #         cfg = json.load(f)
    # except Exception as e:
    #     raise ValueError(f"Failed to parse run_config.json: {cfg_path}") from e
    #
    # if 'hidden_dim' not in cfg:
    #     raise KeyError(f"'hidden_dim' not found in run_config.json: {cfg_path}")
    # try:
    #     hidden_dim = int(cfg['hidden_dim'])
    # except Exception as e:
    #     raise ValueError(f"Invalid hidden_dim value in run_config.json: {cfg.get('hidden_dim')}") from e

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

    #* saving the result
    # out_name = kwargs.get('out_name', None)
    # if out_name is None:
    #     out_name = f"test_result_{model_path.stem}_{test_npz.stem}.npz"
    out_name = kwargs.get('out_name', f"{model_path.stem}_{test_npz.stem}-test.npz")
    out_path = Path(kwargs.get('out_dir', model_path.parent))/str(out_name)
    out_path = out_path.with_suffix('.npz') if out_path.suffix.lower() != '.npz' else out_path

    save_payload = {'cache_index': cache_index, 'y_true': y_true, 'y_pred': y_pred,}
    np.savez_compressed(out_path, **save_payload)
    #
    # results = {'path': str(out_path),
    #            'num_samples': len(y_true),
    #            'cache_index': cache_index,
    #            'y_true': y_true, 'y_pred': y_pred,}
    print(f"Testing complete\n Raw results saved to {out_path}")
    # return results
    return {'path': str(out_path), **save_payload}


def run_test(test_cache: str | Path, tst_model: str | Path, **kwargs):
    """Alias for run_testing."""
    from  evaluation_tools import analyze_test_results, print_test_report
    # return run_testing(test_cache, tst_model, **kwargs)
    res = run_testing(test_cache, tst_model, **kwargs)
    report = analyze_test_results(res, out_path=tst_model.parent)
    print_test_report(report)
    analyze_test_results(res['path'])


# --------------------------------------------------
# Example training script
# --------------------------------------------------

if __name__ == '__main__':
    pass

    # Example:
    # model, hist = run_training('data/cache/RWF_train.npz', 'data/cache/RWF_valid.npz')
    # model, hist = run_training('data/cache/RWF_train.npz', split_ratio=0.85, split_seed=42)
    d = Path("data/cache/")
    # m, l = run_training("data/cache/RWF_train.npz", tag='RWF_tms_f18' )
    tst_model = Path("work_dirs/json_models/train_260323-0314_RWF_tms_f18/model.pt")
    run_test(d/'RWF_valid.npz', tst_model)

#143 (,12,4)
# 184(,5,3)
