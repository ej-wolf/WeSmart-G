# Minimal TSN toy config (rawframes) — runs in seconds
# Path assumptions (relative to your project root):
#   data/rawframes_dummy/{classX_clipY/img_00001.jpg ...}
#   data/rawframes_dummy/train.txt / val.txt   (made by scripts/make_dummy_rawframes.py)


dataset_name = 'RawframeDataset'
data_dir = 'dummy_test/rawframes_dummy'
data_root = data_dir
ann_train = f"{data_dir}/train.txt"
ann_val   = f"{data_dir}/val.txt"

default_scope = 'mmaction'

# ----- Model -----
model = dict(
    type='Recognizer2D',
    backbone=dict(type='ResNet', depth=18, pretrained=None),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus'),
        dropout_ratio=0.4),
    test_cfg=dict(average_clips='prob'),
)

# ----- Pipelines (rawframes; no Decord needed) -----
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 160)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
]
val_pipeline = train_pipeline
test_pipeline = train_pipeline

# ----- Dataloaders -----
train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(type=dataset_name, ann_file=ann_train, data_prefix=dict(img=data_root), pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(type=dataset_name, ann_file=ann_val, data_prefix=dict(img=data_root), pipeline=val_pipeline))

test_dataloader = val_dataloader

# ----- Evaluators (required if val/test enabled) -----
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')

# ----- Optimization & schedule -----
optim_wrapper = dict( optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4))
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[1])

# ----- Loops -----
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ----- Hooks (make logging chatty + save one checkpoint) -----
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
)

# Disable auto LR scaling (tiny toy)
auto_scale_lr = dict(enable=False)
log_level = 'INFO'

# ⬇ turn off validation in the toy run
val_dataloader, val_cfg, val_evaluator = None, None, None
test_dataloader, test_cfg, test_evaluator = None, None, None
