"""  NO-AGGREGATION SANITY TEST
===============================

Purpose:  Verify that the trained model produces non-degenerate predictions
*per clip*, without any MMAction test-time aggregation.

This script intentionally bypasses:
- average_clips
- temporal_pool
- evaluation metrics

If this test passes, training + model are proven correct and
any collapse must originate from MMAction test-time logic.
"""

# IMPORTANT: register all MMAction modules so Recognizer2D is known

from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)
# from mmengine.config import Config
# from mmengine.runner import load_checkpoint
# from mmaction.registry import MODELS
# from mmaction.datasets import build_dataset, build_dataloader
# from mmengine.registry import MODELS
# from mmengine.dataset import build_dataset, build_dataloader
from mmengine.registry import MODELS, DATASETS
from mmengine.runner import  load_checkpoint, Runner
# from mmengine.runner import load_checkpoint
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.logging.history_buffer import HistoryBuffer

import torch
from torch.utils.data import DataLoader
# torch.serialization.add_safe_globals([HistoryBuffer])

from utils.torch_safe_load import enable_checkpoint_loading
enable_checkpoint_loading()

from collections import Counter
# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CFG_PATH  = 'configs/tsm_R50_MMA_RWF.py'
CKPT_PATH = 'work_dirs/tsm_R50_MMA_nc2-l4-b4-v/best.pth'
DEVICE = 'cuda'

# -----------------------------------------------------------------------------
# LOAD CONFIG
# -----------------------------------------------------------------------------

cfg = Config.fromfile(CFG_PATH)
cfg.model.test_cfg = None   #* CRITICAL: disable any test-time aggregation logic

# -----------------------------------------------------------------------------
# BUILD MODEL
# -----------------------------------------------------------------------------

model = MODELS.build(cfg.model)
model.to(DEVICE)
model.eval()

load_checkpoint( model, CKPT_PATH,  map_location=DEVICE)

# -----------------------------------------------------------------------------
# BUILD DATASET & DATALOADER (UNCHANGED)
# -----------------------------------------------------------------------------

# dataset = build_dataset(cfg.test_dataloader.dataset)
# dataloader = build_dataloader( dataset,  **cfg.test_dataloader)
dataset = DATASETS.build(cfg.test_dataloader.dataset)
sampler = DefaultSampler(dataset, shuffle=False)
dataloader = DataLoader( dataset, sampler=sampler,
                         batch_size =cfg.test_dataloader.batch_size,
                         num_workers=cfg.test_dataloader.num_workers,
                         pin_memory =cfg.test_dataloader.pin_memory,
                         collate_fn=dataset.collate_fn
                        )

# -----------------------------------------------------------------------------
# RAW INFERENCE (NO AGGREGATION)
# -----------------------------------------------------------------------------

all_preds = []
all_targets = []
all_probs = []

print("\nRunning NO-AGG sanity test...\n")

with torch.no_grad():
    for i, data in enumerate(dataloader):
        inputs = data['inputs'].to(DEVICE)   # (B, C, T, H, W)
        targets = data['labels'].to(DEVICE)  # (B,)

        logits = model(inputs)               # EXPECTED: (B, num_classes)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_probs.append(probs.cpu())

        # Print first few batches for shape + sanity inspection
        if i < 3:
            print(f"Batch {i}")
            print(" inputs :", tuple(inputs.shape))
            print(" logits :", tuple(logits.shape))
            print(" targets:", tuple(targets.shape))
            uniq, cnt = preds.unique(return_counts=True)
            print(" preds  :", {int(u): int(c) for u, c in zip(uniq, cnt)})
            print("-")

# -----------------------------------------------------------------------------
# DISTRIBUTION INSPECTION (NO METRICS)
# -----------------------------------------------------------------------------

print("\n=== FINAL DISTRIBUTIONS ===")
print("Predictions:", Counter(all_preds))
print("Targets     :", Counter(all_targets))

all_probs = torch.cat(all_probs, dim=0)
print("Mean softmax confidence per class:")
print(all_probs.mean(dim=0))

print("\nNO-AGG sanity test completed.\n")

#120(5,,3)
