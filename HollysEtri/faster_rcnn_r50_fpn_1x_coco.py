_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'EndoCVdataset'
data_root = 'data/EndoCV2021/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/image/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/image/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'test/image/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
# runtime settings
total_epochs =  100
evaluation = dict(interval=1, metric=['bbox'])
