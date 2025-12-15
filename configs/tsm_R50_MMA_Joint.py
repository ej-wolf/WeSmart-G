# weSmart/configs/tsm_R50_MMA_JOINT.py

_base_ = ['./tsm_R50_MMA_base.py']


dataset_type = 'VideoDataset'

# Joint root: both datasets live under data/video
data_root = '../../data/video'
All_Ann_File = "joint_all.txt"
Train_File = "joint_train.txt"
Valid_File = "joint_val.txt"

work_dir = '../../work_dirs/tsm_R50_MMA_JOINT'

#* --- Model
Classes = ['NonViolence', 'Violence']
# num_classes = 2

model = dict(cls_head=dict( num_classes=len(Classes) ))

#*  --- DataLoaders -----
train_dataloader = dict( dataset=dict( type=dataset_type,
                                       ann_file=f'{data_root}/joint_train.txt',
                                       data_prefix=dict(video=data_root)
                                     ))

val_dataloader  = dict( dataset=dict( type=dataset_type,
                                      ann_file=f'{data_root}/joint_val.txt',
                                      data_prefix=dict(video=data_root),
                                    ))

# For now, test = val; you can change later
test_dataloader = val_dataloader
