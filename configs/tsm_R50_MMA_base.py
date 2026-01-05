"""   Config file for the TSM with video training   """

# -------------------------------------------------------------------------------
# ----- basic dataset & path settings -----
# -------------------------------------------------------------------------------
default_scope = 'mmaction'
dataset_type = 'VideoDataset'

#* Paths (assuming CWD = ./extern/mmaction2)
Current_Data_Root = '../../data/video'
# All_Ann_File = "all.txt"
# Valid_File = "val.txt"
# Train_File = "train.txt"

data_root = Current_Data_Root
# data_root_val  = data_root
# ann_file_train = data_root + '/' + Train_File
# ann_file_valid = data_root + '/' + Valid_File
# ann_file_test  = data_root + '/' + All_Ann_File   #* reuse val as test for now

# work_dir = '../../work_dirs'

# -------------------------------------------------------------------------------
# --- Model: TSM with ResNet50 backbone, 2 classes (violence/ no-violence) -----
# -------------------------------------------------------------------------------

num_classes  = 2   #*  2 classes (violence / no-violence)
img_sz = 224

#* clip params
clip_len = 4
frame_interval = 1
num_clips = 2

#* model parmas
Resnet_Depth = 50
Out_Layer = 4
in_channels  = 2048
dropout = 0.5

model = dict( type='Recognizer2D',
              data_preprocessor=dict(type='ActionDataPreprocessor',
                            format_shape='NCHW',
                            mean=[123.675, 116.28, 103.53],
                            std =[58.395 ,  57.12, 57.375],
                            ),
              backbone=dict(# type='ResNet',
                            type='ResNetTSM',
                            depth=Resnet_Depth,  # <-- key difference vs TSN/TRN
                            pretrained='torchvision://resnet50',
                            out_indices=(Out_Layer-1,),
                            norm_eval=False,
                            shift_div=8,
                            # plugins=[dict(type='TemporalShift', shift_div=8, position='after_conv1')]
                            # style='pytorch',shift_div=8, #* typical TSM setting
                            ),
              cls_head=dict(type='TSMHead',
                            num_classes=num_classes,            # violence / non-violence
                            in_channels=in_channels,
                            spatial_type='avg',
                            # consensus=dict(type='AvgConsensus', dim=1),
                            dropout_ratio=dropout,
                            init_std=0.01,
                            is_shift=True,
                            temporal_pool=False,
                            average_clips='score',
                            topk=(),
                            ),
             )


# ----------------------------------------------------------------------
# --- Pipelines
# ----------------------------------------------------------------------

# dataset_type = 'VideoDataset'
train_pipeline = [dict(type='DecordInit'),
                  dict(type='SampleFrames',
                       clip_len=clip_len,
                       frame_interval=frame_interval,
                       num_clips=num_clips),
                  dict(type='DecordDecode'),
                  dict(type='Resize', scale=(-1, 256)),
                  dict(type='RandomResizedCrop'),
                  #dict(type='Resize', scale=(img_sz, img_sz), keep_ratio=False),
                  dict(type='Flip', flip_ratio=0.5),
                  dict(type='FormatShape', input_format='NCHW'),
                  dict(type='PackActionInputs')
                 ]

val_pipeline = [dict(type='DecordInit'),
                dict(type='SampleFrames',
                     clip_len=clip_len,
                     frame_interval= frame_interval,
                     num_clips=num_clips,
                     test_mode=True),
                dict(type='DecordDecode'),
                dict(type='Resize', scale=(-1, 256)),
                dict(type='CenterCrop', crop_size=img_sz),
                dict(type='FormatShape', input_format='NCHW'),
                dict(type='PackActionInputs'),
                ]

test_pipeline = val_pipeline


# ----------------------------------------------------------------------
# --- DataLoaders (mmengine-style) -------
# ----------------------------------------------------------------------

#* consts for all dataloaders
Batch_sz = 2
N_Workers = 4

train_dataloader = dict(batch_size=Batch_sz, num_workers=N_Workers, persistent_workers=True, #)
                        # dataset=dict(type=data)) # to be overridden in child
                        dataset=dict(type=dataset_type,
                                     ann_file='',
                                     data_prefix=dict(video=''),
                                     pipeline=train_pipeline,
                                     test_mode=False
                                     )
                        )

val_dataloader   = dict(batch_size=Batch_sz, num_workers=N_Workers, persistent_workers=True,
                        dataset=dict(type=dataset_type,
                                     ann_file='',
                                     data_prefix=dict(video=''),
                                     pipeline=val_pipeline,
                                     test_mode=True)
                        )

test_dataloader  = val_dataloader

# ----------------------------------------------------------------------
# ---------------- Training, Loops, Optimizer, Evaluation --------------
# ----------------------------------------------------------------------

Max_Epochs = 60
Val_Frq = 5         #* run validation every 5 epochs
ChP_Frq = 10

#* Optimizer wrapper
optim_wrapper = dict(type='AmpOptimWrapper',
                     #* weights update (SGD + lr + momentum).
                     optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
                     loss_scale='dynamic'
                    ) # clip_grad=None)

# lr_config = dict( policy='step', step=[20, 40] ) #* learning rate schedule (decays at epochs 20 & 40).

checkpoint_config = dict(interval=ChP_Frq)    #* checkpoints saving frequency

#* ----- training schedule / runtime settings (you can keep defaults or tweak) -----
param_scheduler = [dict(type='MultiStepLR',
                        begin=0, end=Max_Epochs, by_epoch=True,
                        milestones=[20, 40], gamma=0.1
                        )]

#* --- Loops
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=Max_Epochs, val_interval=Val_Frq)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


#* --- Evaluation Metrics to compute on val/test
# evaluation = dict( interval=5,  metrics=['top_k_accuracy', 'mean_class_accuracy'])
val_evaluator = dict(type='AccMetric',
                     metric_list=('top_k_accuracy',),       #* avoid mean_class_accuracy bug
                     metric_options=dict(top_k_accuracy=dict(topk=(1,)) ) #* top-1 only
                    )
test_evaluator = val_evaluator

# ----------------------------------------------------------------------
# ---------------- Logging / misc ----------------
# ----------------------------------------------------------------------

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# Default hooks (rough equivalent of old log_config, checkpoint_config, etc.)
default_hooks = dict( # optimizer=dict(type='OptimizerHook', grad_clip=None),
                     checkpoint=dict(type='CheckpointHook', interval=ChP_Frq, save_best='auto'),
                     logger=dict(type='LoggerHook', interval=20),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     )

# ----------------------------------------------------------------------
# work_dir = f"../../work_dirs/tsm_R50_MMA_nc{num_clips}_b{Batch_sz}"
work_dir = f"../../work_dirs/tsm_R50_MMA_nc{num_clips}-cl{clip_len}-b{Batch_sz}"