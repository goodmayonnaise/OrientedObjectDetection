_base_ = [ './prototype4.py']

auto_resume = False
find_unused_parameters = True

work_dir = './work_dirs/objectness3-ver1'
dataset_type = 'DOTADataset'
data_root = 'data/split_ms_dota2_2/'
samples_per_gpu = 8
workers_per_gpu = 4

max_epochs = 12

# # test
# data_root = 'data/debug/'
# samples_per_gpu = 1
# work_dir = './work_dirs/debug'

num_classes = 15

model = dict(
    bbox_head=dict(
        type='RotatedDecoupled1x1ObjHead',
        num_classes=num_classes,
        loss_cls=dict(type='ObjectnessLoss3', loss_weight=1.0, obj_loss_weight=1.0, ver=1),
        train_cfg=dict(
            assigner=dict(
                type='OBBLabelAssigner',
                num_classes=num_classes))),
    train_cfg=dict(
        assinger=dict(
            type='OBBLabelAssigner',
            num_classes=num_classes))
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


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PolyRandomRotate',
         rotate_ratio=0.6,
         angles_range=180,
         auto_bound=False,
         rect_classes=[9, 11],
         version='le90'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))


