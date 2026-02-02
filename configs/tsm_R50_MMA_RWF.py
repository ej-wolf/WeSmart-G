# weSmart/configs/tsm_R50_MMA_RWF.py

_base_ = ['./tsm_R50_MMA_base.py']
if False:  from .tsm_R50_MMA_base import train_pipeline, val_pipeline

dataset_type = 'VideoDataset'

#* from mmaction2/extern, ../../data is video/RWF-mart/data
data_root = "../../data/video/RWF-2000"
All_Ann_File = "all.txt"
Train_File = "train.txt"
Valid_File = "val.txt"

#* --- Model
Classes = ['NonFight', 'Fight']
model = dict(cls_head=dict( num_classes=len(Classes) ))

#*  --- DataLoaders -----
train_dataloader = dict( dataset=dict(type=dataset_type,    #* redundant but explicit
                                      ann_file=f'{data_root}/{Train_File}',
                                      data_prefix=dict(video=data_root))
                        )
val_dataloader  = dict(dataset=dict(type=dataset_type,
                                    ann_file=f'{data_root}/{Valid_File}',
                                    data_prefix=dict(video=data_root))
                      )

test_dataloader = val_dataloader
