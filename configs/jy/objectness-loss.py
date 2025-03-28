'''
classification 16 output
split 15, 1(objectness) channel
indicator focal, bce objectness loss
'''


_base_ = [ './prototype4.py']

auto_resume = False
find_unused_parameters = True

work_dir = './work_dirs/objectness-ver2'
dataset_type = 'DOTADataset'
data_root = 'data/split_ms_dota2_2/'
samples_per_gpu = 8
workers_per_gpu = 4

max_epochs = 12
# # test
# data_root = 'data/debug/'
# samples_per_gpu = 1
# work_dir = './work_dirs/debug'

model = dict(
    bbox_head=dict(
        type='RotatedYOLOv8Head',
        loss_cls=dict(type='ObjectnessLoss', loss_weight=1.0, obj_loss_weight=1.0, ver=2),
        )
    )

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
