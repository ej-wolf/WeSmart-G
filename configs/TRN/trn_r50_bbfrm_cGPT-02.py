# TRN on box-based raw frames (violence vs non-violence)
# This config is written for mmengine-style Runner.from_cfg.
# Paths are relative to extern/mmaction2 as working directory.



default_scope = 'mmaction'
dataset_type = 'RawframeDataset'

# Paths (from CWD = weSmart/extern/mmaction2)
DATA_ROOT = '../../data/cache'



data_root = DATA_ROOT
data_root_val = DATA_ROOT
ann_file_train = DATA_ROOT + '/train.txt'
ann_file_val = DATA_ROOT + '/val.txt'
ann_file_test = DATA_ROOT + '/val.txt'  # reuse val as test for now

# ----------------------------------------------------------------------
# Model: TRN + ResNet50, binary classification (violence / non-violence)
# ----------------------------------------------------------------------

num_classes  = 2
num_segments = 8   # frames sampled per window

model = dict( type='Recognizer2D',
        backbone=dict( type='ResNet',
        depth=50,
        pretrained='torchvision://resnet50',
        out_indices=(3, ),
        ),
        cls_head=dict( type='TRNHead',
        num_classes=num_classes,
        in_channels=2048,
        num_segments=num_segments,
        spatial_type='avg',
        relation_type='TRNMultiScale',  # or 'TRN' for simpler variant

        hidden_dim=256,
        dropout_ratio=0.8,
        init_std=0.001,
    ),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
)

# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std= [58.395, 57.12, 57.375],
                    to_bgr=False,
                    )

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_segments),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
    ]

val_pipeline = [
    dict( type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_segments,
        test_mode=True, ),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
    ]

test_pipeline = val_pipeline

# ----------------------------------------------------------------------
# NEW mmengine-style sections (Runner.from_cfg expects these)
# ----------------------------------------------------------------------

# Dataloaders
train_dataloader = dict(# try 16; drop to 8 if OOM
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type,
                 ann_file=ann_file_train,
                 data_prefix=data_root,
                 pipeline=train_pipeline,),
    )
val_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type,
                 ann_file=ann_file_val,
                 data_prefix=data_root_val,
                 pipeline=val_pipeline,),
    )
test_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type,
                 ann_file=ann_file_test,
                 data_prefix=data_root_val,
                 pipeline=test_pipeline,
                ),
    )

# Optimizer wrapper (replaces old optimizer + optimizer_config)
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=40, norm_type=2),
                    )

# LR scheduler (replaces old lr_config)
total_epochs = 50

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=total_epochs,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1,
    )
]

# Loops
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=total_epochs,
    val_interval=5,   # run validation every 5 epochs
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks (rough equivalent of old log_config, checkpoint_config, etc.)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='auto',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Evaluation (metric(s) to compute on val/test)
val_evaluator = dict(
    type='AccMetric',
    metric_list=['top_k_accuracy', 'mean_class_accuracy'],
)

test_evaluator = val_evaluator

# Environment + logging level
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'

# Work dir relative to extern/mmaction2
work_dir = '../../work_dirs/trn_r50_bbrfm_c2_s8'

load_from = None
resume_from = None
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# ----------------------------------------------------------------------
# OLD-STYLE FIELDS (kept for reference, now unused by mmengine Runner)
# ----------------------------------------------------------------------

# data = dict(
#     videos_per_gpu=16,
#     workers_per_gpu=4,
#     train=dict(
#         type=dataset_type,
#         ann_file=ann_file_train,
#         data_prefix=data_root,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=ann_file_val,
#         data_prefix=data_root_val,
#         pipeline=val_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=ann_file_test,
#         data_prefix=data_root_val,
#         pipeline=test_pipeline),
# )

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# lr_config = dict(
#     policy='step',
#     step=[20, 40],
# )

# checkpoint_config = dict(interval=5)
# evaluation = dict(
#     interval=5,
#     metrics=['top_k_accuracy', 'mean_class_accuracy'],
# )

# log_config = dict(
#     interval=20,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook'),
#     ],
# )

# dist_params = dict(backend='nccl')
# workflow = [('train', 1)]
