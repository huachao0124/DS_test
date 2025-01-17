_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

crop_size = (1024, 512)

pretrained = 'ckpts/swin_base_patch4_window12_384_22k.pth'  # noqa
lang_model_name = './bert-base-uncased'


image_size = (1024, 1024)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)


model = dict(
    type='GroundingDINOPTSegRoI',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=data_preprocessor,
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHeadPTSegRoI',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0),
        mask_head=dict(type='MaskRCNNHead',
                        mask_roi_extractor=dict(
                            type='SingleRoIExtractor',
                            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                            out_channels=256,
                            featmap_strides=[4, 8, 16, 32]),
                        mask_head=dict(
                            type='FCNMaskHead',
                            num_convs=4,
                            in_channels=256,
                            conv_out_channels=256,
                            num_classes=1,
                            loss_mask=dict(
                                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                        ),
                        train_cfg=dict(
                                        # assigner=dict(
                                        #     type='HungarianAssigner',
                                        #     match_costs=[
                                        #         dict(type='FocalLossCost', weight=2.0),
                                        #         dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                                        #         dict(type='IoUCost', iou_mode='giou', weight=2.0)
                                        #     ]),
                                        # sampler=dict(type='PseudoSampler'),
                                        assigner=dict(
                                            type='MaxIoUAssigner',
                                            pos_iou_thr=0.5,
                                            neg_iou_thr=0.5,
                                            min_pos_iou=0.5,
                                            match_low_quality=True,
                                            ignore_iof_thr=-1),
                                        sampler=dict(
                                            type='RandomSampler',
                                            num=512,
                                            pos_fraction=0.25,
                                            neg_pos_ub=-1,
                                            add_gt_as_proposals=True),
                                        mask_size=28,
                                        pos_weight=-1,
                                        debug=False
                        ),
                        test_cfg=dict(
                                        score_thr=0.05,
                                        nms=dict(type='nms', iou_threshold=0.5),
                                        max_per_img=100,
                                        mask_thr_binary=0.5)
        )
    ),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ]),
        ),
    test_cfg=dict(max_per_img=300),
    )


embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0),
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters = 368750
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[327778, 355092],
    gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=max_iters + 2,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='AddPrompt'),
    dict(type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]


test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddPrompt'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]


# dataset settings
class_name = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle')
palette = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
            (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
            (107, 142, 35), (152, 251, 152), (70, 130, 180),
            (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

metainfo = dict(classes=class_name, palette=palette)


train_dataset_type = 'CityscapesWithCocoDataset'
train_data_root = 'data/cityscapes/'
test_dataset_type = 'RoadAnomalyDataset'
test_data_root = 'data/RoadAnomaly'
# test_dataset_type = 'FSLostAndFoundDataset'
# test_data_root = 'data/FS_LostFound'
# test_data_root = 'data/FS_Static'
# test_dataset_type = 'SMIYCDataset'
# test_data_root = 'data/SMIYC/dataset_AnomalyTrack'
# test_dataset_type = 'LostAndFoundDataset'
# test_data_root = 'data/LostAndFound'


class_name = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle')
palette = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
            (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
            (107, 142, 35), (152, 251, 152), (70, 130, 180),
            (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

metainfo = dict(classes=class_name, palette=palette)

train_dataloader = dict(_delete_=True,
                        batch_size=2,
                        num_workers=2,
                        # sampler=dict(type='DefaultSampler', shuffle=True),
                        # batch_sampler=dict(type='AspectRatioBatchSampler'),
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        # batch_sampler=dict(type='InfiniteBatchSampler'),
                        dataset=dict(type=train_dataset_type, 
                                     coco_file_path='data/coco/',
                                     data_root=train_data_root,
                                     data_prefix=dict(
                                        img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                                     pipeline=train_pipeline))
# val_dataloader = dict(dataset=dict(type=test_dataset_type,
#                                      data_root=test_data_root,
#                                      pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(_delete_=True,
                                    type=test_dataset_type, 
                                    data_root=test_data_root, 
                                    pipeline=test_pipeline, 
                                    # img_suffix='.webp',
                                    # img_suffix='.jpg',
                                    data_prefix=dict(
                                        # img_path='images', seg_map_path='labels_masks'),))
                                        # img_path='original', seg_map_path='labels'),))
                                        img_path='leftImg8bit/test', seg_map_path='gtCoarse/test'),))
test_dataloader = val_dataloader
# val_evaluator = dict(type='AnomalyMetricRbA')
val_evaluator = dict(type='BlankMetric')
# val_evaluator = dict(type='AnomalyIoUMetric')
# val_evaluator = dict(type='AnomalyMetricLoad')
test_evaluator = val_evaluator

# training schedule for 90k
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='GroundingVisualizationHook', draw=True, interval=5, score_thr=0.0))

vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='VisualizerHeatMap', vis_backends=vis_backends, name='visualizer')
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)

load_from = '/home/arima/mmdetection/iter_5000.pth'