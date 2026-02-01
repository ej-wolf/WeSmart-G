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
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.logging.history_buffer import HistoryBuffer


import torch
from torch.utils.data import DataLoader, default_collate
# torch.serialization.add_safe_globals([HistoryBuffer])

from common.torch_safe_load import enable_checkpoint_loading
enable_checkpoint_loading()
from my_local_utils import correct_path

from collections import Counter

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CFG_PATH  = 'configs/tsm_R50_MMA_RWF.py'
CKPT_PATH = 'work_dirs/tsm_R50_MMA_nc2-l4-b4-v/best.pth'
# CFG_PATH  = '../configs/tsm_R50_MMA_RWF.py'
# CKPT_PATH = '../work_dirs/tsm_R50_MMA_nc2-l4-b4-v/best.pth'
DEVICE = 'cuda'

#* LOAD CONFIG
cfg = Config.fromfile(CFG_PATH)
cfg.model.test_cfg = None   #* CRITICAL: disable any test-time aggregation logic

# cfg.data_root = correct_path(cfg.data_root)
ds = cfg.test_dataloader.dataset
ds.ann_file = correct_path(ds.ann_file)
ds.data_prefix['video'] = correct_path(ds.data_prefix['video'])

# -----------------------------------------------------------------------------
# BUILD MODEL
# -----------------------------------------------------------------------------

model = MODELS.build(cfg.model)
model.to(DEVICE)
model.eval()

# load_checkpoint( model, CKPT_PATH,  map_location=DEVICE)
# NOTE: PyTorch >=2.6 defaults to weights_only=True, which breaks MMEngine checkpoints.

# For this sanity test we explicitly load the full checkpoint (trusted, local).
checkpoint = torch.load( CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])


# -----------------------------------------------------------------------------
# BUILD DATASET & DATALOADER (UNCHANGED)
# -----------------------------------------------------------------------------

# dataset = build_dataset(cfg.test_dataloader.dataset)
# dataloader = build_dataloader( dataset,  **cfg.test_dataloader)
dataset = DATASETS.build(cfg.test_dataloader.dataset)
sampler = DefaultSampler(dataset, shuffle=False)

#* pin_memory, num_workers, batch_size may be absent in minimal configs
# pin_memory  = bool(getattr(cfg.test_dataloader, 'pin_memory', False))
# num_workers = int(getattr(cfg.test_dataloader, 'num_workers', 0))
# batch_size  = int(getattr(cfg.test_dataloader, 'batch_size', 1))
#
# dataloader = DataLoader( dataset,
#                          sampler=sampler,
#                          batch_size =cfg.test_dataloader.batch_size,
#                          num_workers=cfg.test_dataloader.num_workers,
#                          pin_memory =cfg.test_dataloader.pin_memory,
#                          collate_fn=dataset.collate_fn
#                         )

dataloader = DataLoader( dataset, sampler = sampler,
                         batch_size = int(getattr(cfg.test_dataloader, 'batch_size', 1)),
                         num_workers= int(getattr(cfg.test_dataloader, 'num_workers', 0)),
                         pin_memory = bool(getattr(cfg.test_dataloader, 'pin_memory', False)),
                         collate_fn = getattr(dataset, 'collate_fn', pseudo_collate)
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
        # inputs = data['inputs'].to(DEVICE)   # (B, C, T, H, W)
        # # targets = data['labels'].to(DEVICE)  # (B,)
        # targets = torch.tensor([int(ds.gt_label.item()) for ds in data['data_samples']], device=DEVICE)


        # if isinstance(data, list):
        #     inputs = torch.stack([d['inputs'] for d in data], dim=0).to(DEVICE)
        #     targets = torch.tensor([int(d['labels']) for d in data], device=DEVICE)
        # else:
        #     with torch.no_grad():
        #         for i, data in enumerate(dataloader):
        #             # pseudo_collate ALWAYS returns a list
        #             # inputs = torch.stack([d['inputs'] for d in data], dim=0).to(DEVICE)
        #             # targets = torch.tensor([int(d['labels']) for d in data], device=DEVICE)
        #             inputs = data['inputs'].to(DEVICE)
        #             # labels are stored inside ActionDataSample objects
        #             targets = torch.tensor( [int(ds.gt_label.item()) for ds in data['data_samples']],
        #                                     device=DEVICE)
        #             logits = model(inputs)  # (B, num_classes)
        #             probs = torch.softmax(logits, dim=1)
        #             preds = torch.argmax(probs, dim=1)
        #
        #             all_preds.extend(preds.cpu().tolist())
        #             all_targets.extend(targets.cpu().tolist())
        #             all_probs.append(probs.cpu())
        #
        #             if i < 3:
        #                 print(f"Batch {i}\n inputs :{tuple(inputs.shape)}\n "
        #                       f" logits :{tuple(logits.shape)}\n"
        #                       f" targets: {tuple(targets.shape)}")
        #                 # print(f"Batch {i}")
        #                 # print(" inputs :", tuple(inputs.shape))
        #                 # print(" logits :", tuple(logits.shape))
        #                 # print(" targets:", tuple(targets.shape))
        #                 uniq, cnt = preds.unique(return_counts=True)
        #                 print(" preds  :", {int(u): int(c) for u, c in zip(uniq, cnt)})
        #                 print("-")
        # # pseudo_collate ALWAYS returns a list of samples


        # inputs = torch.stack([d for d in data['inputs']], dim=0).to(DEVICE)
        # inputs = data['inputs'].to(DEVICE).float()
        #
        # targets = torch.tensor([int(d.gt_label.item()) for d in data['data_samples']], device=DEVICE)
        # data is ALWAYS a list when using raw DataLoader + pseudo_collate
        inputs = torch.stack([d for d in data['inputs']], dim=0).to(DEVICE)

        targets = torch.tensor(
            [int(d['data_samples'].gt_label.item()) for d in data],
            device=DEVICE
        )

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
