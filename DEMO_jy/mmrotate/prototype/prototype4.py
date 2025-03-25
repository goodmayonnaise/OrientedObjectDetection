dataset_type = 'DOTADataset'
data_root = 'data/split_ms_dota2_2/'
angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PolyRandomRotate',
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='DOTADataset',
        ann_file='data/split_ms_dota2_2/train/annfiles/',
        img_prefix='data/split_ms_dota2_2/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(1024, 1024)),
            dict(type='RRandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='PolyRandomRotate',
                rotate_ratio=0.6,
                angles_range=180,
                auto_bound=False,
                rect_classes=[9, 11],
                version='le90'),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='DOTADataset',
        ann_file='data/split_ms_dota2_2/val/annfiles/',
        img_prefix='data/split_ms_dota2_2/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='DOTADataset',
        ann_file='data/split_ms_dota2_2/test/images/',
        img_prefix='data/split_ms_dota2_2/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    pin_memory=True)
evaluation = dict(interval=1, metric='mAP', save_best='mAP')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=1e-05)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'configs/test_jh/pretrained/best_coco_bbox_mAP_epoch_273_cspnext.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = './work_dirs/prototype4-10'
gpu_ids = range(0, 2)
epoch = 50
find_unused_parameters = True
num_training_classes = 15
deepen_factor = 0.67
widen_factor = 0.75
last_stage_out_channels = 768
bbox_in_channels = [256, 512, 768]
regress_ranges = ((-1, 96), (96, 192), (192, 384))
featmap_strides = [8, 16, 32]
samples_per_gpu = 8
model = dict(
    type='RotatedYOLOv6',
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        last_stage_out_channels=768,
        deepen_factor=0.67,
        widen_factor=0.75,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        stage_aux=None,
        reverse=True,
        cspnext_block=True),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=0.67,
        widen_factor=0.75,
        in_channels=[256, 512, 768],
        out_channels=[256, 512, 768],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RotatedYOLOv8Head',
        num_classes=15,
        in_channels=[256, 512, 768],
        regress_ranges=((-1, 96), (96, 192), (192, 384)),
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
        train_cfg=dict(
            assigner=dict(
                type='OBBLabelAssigner',
                regress_ranges=((-1, 96), (96, 192), (192, 384)),
                featmap_strides=[8, 16, 32],
                num_classes=15,
                topk=15))),
    train_cfg=dict(
        assigner=dict(
            type='OBBLabelAssigner',
            regress_ranges=((-1, 96), (96, 192), (192, 384)),
            featmap_strides=[8, 16, 32],
            num_classes=15,
            topk=15)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.1,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
auto_resume = False
