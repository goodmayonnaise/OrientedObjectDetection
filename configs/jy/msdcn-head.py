auto_resume = False
find_unused_parameters = True
persistent_workers = False

_base_ = [ '../_base_/datasets/dotav1.py', 
          '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs/msdcn-head'
dataset_type = 'DOTADataset'
data_root = 'data/split_ms_dota2_2/'
samples_per_gpu = 8
workers_per_gpu = 4

# test
# data_root = 'data/dummy3/'
# samples_per_gpu = 1
# work_dir = './work_dirs/_test'

matching = 0 # base LA
num_classes = 15
epoch = 50
regress_ranges= ((-1, 96), (96, 192), (192, 384))

deepen_factor = 0.67
widen_factor = 0.75
last_stage_out_channels = 768
angle_version = 'le90'

load_from = '/mmrotate/configs/jy/pretrained/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_223400-f40abfcd.pth' 
weight_decay = 0.05
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
strides = [8, 16, 32]

model = dict(
    type='RotatedYOLOv6',
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg ,
        act_cfg=dict(type='SiLU', inplace=True)),
    
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),  
      
    bbox_head=dict(
        type='RotatedMSDCNHead',
        num_classes=15,
        in_channels=[256, 512, 768],
        regress_ranges=regress_ranges,

        widen_factor=0.75,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        featmap_strides=[8, 16, 32],
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        matching = matching, ## jh
        train_cfg=dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=15,
                use_ciou=True,
                topk=10,
                alpha=0.5,
                beta=6.0,
                eps=1e-09))),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000)
    )

    
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)


optim_wrapper = dict(
    optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))

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

workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=epoch)
