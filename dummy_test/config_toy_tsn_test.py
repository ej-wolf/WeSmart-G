# tiny TSN on rawframes (no heavy decoding)
default_scope = 'mmaction'

data_dir = 'data/rawframes_dummy'

model = dict(
    type='Recognizer2D',
    backbone=dict(type='ResNet', depth=18, pretrained=False),
    cls_head=dict(type='TSNHead', num_classes=2, in_channels=512, spatial_type='avg', consensus=dict(type='AvgConsensus'), dropout_ratio=0.4),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'RawframeDataset'
data_root = data_dir
ann_train = f"{data_dir}/train.txt"
ann_val   = f"{data_dir}/val.txt"

# --- Pipelines:  for rawframes ---
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 160)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
]
val_pipeline = train_pipeline
test_pipeline = train_pipeline  # if you ever run tes

train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(type=dataset_type, ann_file=ann_train, data_prefix=dict(img=data_root), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(type=dataset_type, ann_file=ann_val, data_prefix=dict(img=data_root), pipeline=val_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4))
param_scheduler = dict(type='MultiStepLR', milestones=[1], by_epoch=True)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
)
# --- Loops ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

#--- Evaluators ---
val_evaluator = dict(type='AccMetric')   # top-1/5 accuracy for classification
test_evaluator = dict(type='AccMetric')  # optional

auto_scale_lr = dict(enable=False)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'))
log_level = 'INFO'