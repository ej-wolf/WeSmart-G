"""
    Minimal PyTorch Dataset + classifier for cached clip-level features.
    Assumes cached .npz files created by `precompute_clip_features` with keys:
    - 'X'    : (N, C) float features
    - 'y'    : (N, ) int labels {0,1}
    - 'meta' : optional metadata (ignored here)
"""
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# * Dataset
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


def run_training(cache_dir:str|Path, **kwargs):
    """ Run training on cached clip-level features and save artifacts. """

    # * work/run dirs and files
    cache_dir = Path(cache_dir)
    train_npz = cache_dir/'train_feats.npz'
    val_npz = cache_dir/'val_feats.npz'

    work_dir = Path(kwargs.get('work_dir', "work_dirs/json_model"))
    # ts = datetime.now().strftime('%y%m%d-%H%M')
    run_dir = work_dir/f"train_{datetime.now().strftime('%y%m%d-%H%M')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    #* training params
    batch_size = kwargs.get('batch_size', 256)
    epochs = kwargs.get('epochs', 30)
    lr = kwargs.get('lr', 1e-3)
    hidden_dim = kwargs.get('hidden_dim', 64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save run config (lightweight)
    run_cfg = {'batch_size':batch_size, 'epochs':epochs, 'lr':lr,
               'hidden_dim':hidden_dim, 'cache_dir': str(cache_dir)}

    #* # datasets & loaders
    train_ds = ClipFeatureDataset(train_npz)
    valid_ds = ClipFeatureDataset(val_npz)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader   = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # in_dim = train_ds.X.shape[1]
    model = ClipMLP(train_ds.X.shape[1], hidden_dim=hidden_dim).to(device)

    #* handle class imbalance
    pos_weight = None
    n_pos = int((train_ds.y == 1).sum())
    n_neg = int((train_ds.y == 0).sum())
    if n_pos > 0:
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #* training loop
    history = []
    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = eval_one_epoch(model, valid_loader, criterion, device)

        history += [{ 'epoch': epoch, 'train_loss': float(train_loss), 'val_loss': float(val_loss)}]

        print(f'Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}')

    #* save artifacts
    model_path   = run_dir/'model.pt'
    history_path = run_dir/'history.json'
    cfg_path     =  run_dir/'run_config.json'

    torch.save(model.state_dict(), model_path)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    with open(cfg_path, 'w') as f:
        json.dump(run_cfg, f, indent=2)

    print(f'Saved model to   : {model_path}')
    print(f'Saved history to : {history_path}')

    return model, history

# --------------------------------------------------
# Example training script
# --------------------------------------------------

if __name__ == '__main__':
    pass
    DATA_DIR = Path("./data/json_data")
    # OUT_DIR = DATA_DIR / 'cache'
    model, hist = run_training(DATA_DIR/'cache')
    pass
#143 (,12,4)
