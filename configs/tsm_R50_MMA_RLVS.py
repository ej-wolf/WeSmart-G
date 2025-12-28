# weSmart/configs/tsm_R50_MMA_RLVS.py

_base_ = ['./tsm_R50_MMA_base.py']
if False:  from .tsm_R50_MMA_base import train_pipeline, val_pipeline

dataset_type = 'VideoDataset'

# from mmaction2/extern, ../../data is weSmart/data
data_root = "../../data/video/RLVS"
All_Ann_File = "all.txt"
Train_File = "train.txt"
Valid_File = "val.txt"

work_dir = "../../work_dirs/tsm_R50_MMA_RLVS"

#* --- Model
Classes = ['NonViolence', 'Violence']

model = dict(cls_head=dict( num_classes=len(Classes),))

#*  --- DataLoaders -------
# Batch_sz = 8
# N_Workers = 4

train_dataloader = dict( dataset=dict(type=dataset_type,    #* redundant but explicit
                                      ann_file=f'{data_root}/train.txt',
                                      data_prefix=dict(video=data_root))
                        )
val_dataloader  = dict(dataset=dict(type=dataset_type,
                                    ann_file=f'{data_root}/val.txt',
                                    data_prefix=dict(video=data_root),)
                      )

test_dataloader = val_dataloader
