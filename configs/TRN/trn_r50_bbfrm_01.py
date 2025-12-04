"""
    Config file for the TRN training
"""
# ----- basic dataset + path settings -----
print(f"\nThis is the Config for Version 2.x")

default_scope = 'mmaction'
dataset_type = 'RawframeDataset'

#* Paths (assuming CWD = ./extern/mmaction2)
Current_Data_Root = '../../data/cache'
Valid_File = "val.txt"
Train_File = "train.txt"

data_root = Current_Data_Root
data_root_val  = data_root
ann_file_train = data_root + '/' + Train_File
ann_file_valid = data_root + '/' + Valid_File
ann_file_test  = data_root + '/' + Valid_File   #* reuse val as test for now

# -------------------------------------------------------------------------------
#* --- Model: TRN with ResNet50 backbone, 2 classes (violence/ no-violence) -----
# -------------------------------------------------------------------------------
num_classes  = 2   #*  2 classes (violence / no-violence)
num_segments = 8   #*  number of segments (frames) TRN sees per window

model = dict(type='Recognizer2D',
             backbone= dict(type='ResNet',
                       depth=50, out_indices=(3, ),
                       pretrained='torchvision://resnet50',
                       ),
             cls_head= dict(type='TRNHead',
                       num_classes =num_classes,
                       num_segments=num_segments,
                       in_channels=2048,
                       spatial_type='avg',
                       relation_type='TRNMultiScale', #* or 'TRN' for simpler relation
                       # consensus=dict(type='TRN', num_segments=num_segments),
                       hidden_dim=256,
                       dropout_ratio=0.8,             #* 0.8 is TRN’s default
                       init_std=0.001,
                       ),
             # This is sometimes only in TSN configs; harmless to keep
             train_cfg=None,
             test_cfg=dict(average_clips='prob')
             )

# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=  [58.395,  57.12, 57.375],
                    to_rgb=True)

#* Each window has ~15–20 frames; TRN will sample 8 of them
train_pipeline = [    #* 8 segments per window
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_segments),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
    ]

val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_segments,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
    ]

test_pipeline = val_pipeline

# ----------------------------------------------------------------------
# NEW mmengine-style sections (Runner.from_cfg expects these)
# ----------------------------------------------------------------------

#* --- DataLoaders (mmengine-style) ---
train_dataloader = dict(
        batch_size=16, num_workers=4, persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(type=dataset_type,
                     ann_file=ann_file_train,
                     data_prefix=dict(img=data_root),
                     pipeline=train_pipeline,)
        )

val_dataloader = dict(
        batch_size=16, num_workers=4, persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type=dataset_type,
                     ann_file=ann_file_valid,
                     data_prefix=dict(img=data_root_val),
                     pipeline=val_pipeline)
        )

test_dataloader = dict(
        batch_size=16, num_workers=4,  persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type=dataset_type,
                     ann_file=ann_file_test,
                     data_prefix=dict(img=data_root_val),
                     pipeline=test_pipeline)
        )

#* ----- training schedule / runtime settings (you can keep defaults or tweak) -----

# --- Optimizer wrapper ---
optim_wrapper = dict( #* weights update (SGD + lr + momentum).
                optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
                clip_grad=dict(max_norm=40, norm_type=2),)

lr_config = dict( policy='step', step=[20, 40] ) #* learning rate schedule (decays at epochs 20 & 40).

total_epochs = 50

checkpoint_config = dict(interval=5)    #* checkpoints saving frequency

evaluation = dict( interval=5,  metrics=['top_k_accuracy', 'mean_class_accuracy'])


param_scheduler = [dict(type='MultiStepLR',
                        begin=0, end=total_epochs, by_epoch=True,
                        milestones=[20, 40],gamma=0.1
                        )]

#* Loops
train_cfg = dict( type='EpochBasedTrainLoop',  max_epochs=total_epochs, val_interval=5)
            #* run validation every 5 epochs
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks (rough equivalent of old log_config, checkpoint_config, etc.)
default_hooks = dict(
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook', interval=20),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook',interval=5, save_best='auto',),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                )

# Evaluation (metric(s) to compute on val/test)

val_evaluator = dict(type='AccMetric',
                metric_list='top_k_accuracy',
                )

test_evaluator = val_evaluator

# Environment + logging level
env_cfg = dict(cudnn_benchmark=False,
          mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
          dist_cfg=dict(backend='nccl'),
          )

log_level = 'INFO'
#* evaluation frequency and metrics

dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = '../../work_dirs/R50_bbrfm_01'
load_from = None
resume_from = None
vis_backends = [dict(type='LocalVisBackend')]
visualizer =  dict(type='ActionVisualizer', vis_backends=vis_backends)
# workflow = [('train', 1)]
