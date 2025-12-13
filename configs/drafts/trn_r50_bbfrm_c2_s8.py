"""
    Config file for the TRN training
"""

print(f"\nThis is the Config I would like to run")
# ----- basic dataset + path settings -----

# from extern.mmaction2.mmaction.utils import register_all_modules
# from mmaction.utils import register_all_modules
# register_all_modules(init_default_scope=True)

default_scope = 'mmaction'
dataset_type = 'RawframeDataset'
#* project paths
Current_Data_Root = '../../data/cache'
Valid_File = "val.txt"
Train_File = "train.txt"

#* Paths for model
data_root = Current_Data_Root
data_root_val  = data_root
ann_file_train = data_root + '/' + Train_File
ann_file_valid = data_root + '/' + Valid_File
ann_file_test  = data_root + '/' + Valid_File

#* ----- model: TRN with ResNet-50 backbone, 2 classes (violence / no-violence) -----
num_classes  = 2   #*  2 classes (violence / no-violence)
num_segments = 8   #*  number of segments (frames) TRN sees per window

model = dict( type='Recognizer2D',
        backbone= dict(type='ResNet', depth=50,
                  pretrained='torchvision://resnet50',
                  out_indices=(3, )
                  ),
        cls_head= dict( type='TRNHead',
                  num_classes=num_classes,
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

#* ----- data pipeline -----  *#

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=  [58.395,  57.12, 57.375],
                    to_bgr=False)

#* Each window has ~15–20 frames; TRN will sample 8 of them
train_pipeline = [    #* 8 segments per window
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_segments),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
    ]

val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_segments,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
    ]

test_pipeline = val_pipeline

data = dict(      # you might need to lower this if you hit OOM
    videos_per_gpu = 16,
    workers_per_gpu= 4,
    train=dict( type=dataset_type, ann_file=ann_file_train, data_prefix=data_root, pipeline=train_pipeline),
    val =dict( type=dataset_type, ann_file=ann_file_valid, data_prefix=data_root_val, pipeline=val_pipeline),
    test=dict( type=dataset_type, ann_file=ann_file_test, data_prefix=data_root_val, pipeline=test_pipeline),
    )

#* ----- training schedule / runtime settings (you can keep defaults or tweak) -----

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
#* optimizer controls the weights update (SGD + lr + momentum).
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

lr_config = dict( policy='step', step=[20, 40] )
#* learning rate schedule (decays at epochs 20 & 40).

total_epochs = 50

checkpoint_config = dict(interval=5)    #* checkpoints saving frequency

evaluation = dict( interval=5,  metrics=['top_k_accuracy', 'mean_class_accuracy'])
#* evaluation frequency and metrics

log_config = dict( interval=20, hooks=[ dict(type='TextLoggerHook'),
                                      # dict(type='TensorboardLoggerHook'),
                                      ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../../work_dirs/trn_r50_bbrfm_c2_s8'
load_from = None
resume_from = None
workflow = [('train', 1)]
