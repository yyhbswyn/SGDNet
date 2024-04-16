num_classes = 32
model = dict(
    type='FastRCNN',
    init_cfg='your-path',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),

    roi_head=dict(
        type='TITANRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D_CSATT',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True,
            temporal_pool_mode='max'),
        shared_head=dict(type='ACRNHead', in_channels=4608, out_channels=2304),
        bbox_head=dict(
            type='BBoxHeadTITAN',
            dropout_ratio=0.5,
            in_channels=2304,
            num_classes=num_classes,
            multilabel=True)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict( #分配器
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=dict(action_thr=0.002)))

dataset_type = 'AVADataset'
data_root_train = 'your_dataset_path'

data_root_val = 'your_dataset_path/dataset/TITAN/rawframes_test'
anno_root = 'your_dataset_path/dataset/TITAN/annotations'

ann_file_train = f'{anno_root}/train_val.csv'

ann_file_val = f'{anno_root}/test.csv'

exclude_file_train = f'{anno_root}/train_excluded_timestamps.csv'

exclude_file_val = f'{anno_root}/test_excluded_timestamps.csv'

label_file = f'{anno_root}/action_list_train.pbtxt'

proposal_file_train = (f'{anno_root}/dense_proposals_train_val.pkl')

proposal_file_val = f'{anno_root}/dense_proposals_test.pkl'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=48, frame_interval=4),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.8),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'entity_ids'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=48, frame_interval=4),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
        nested=True)
]

data = dict(
    # videos_per_gpu=9,
    # workers_per_gpu=2,
    videos_per_gpu=6,
    workers_per_gpu=16,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        person_det_score_thr=0.9,
        num_classes=num_classes,
        data_prefix=data_root_train,
        start_index=1, ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.5,
        num_classes=num_classes,
        data_prefix=data_root_val,
        start_index=1, ))
data['test'] = data['val']

# optimizer = dict(type='SGD', lr=0.1125, momentum=0.9, weight_decay=0.00001)
optimizer = dict(type='SGD', lr=0.0125, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10,
    warmup_ratio=0.1)
# total_epochs = 20
total_epochs = 50
checkpoint_config = dict(interval=2)
workflow = [('train', 1)]
evaluation = dict(interval=2, save_best='mAP@0.5IOU')
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('your-path')
load_from = ('your-path')
resume_from = None
find_unused_parameters = False

