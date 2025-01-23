_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'ckpts/swin_base_patch4_window12_384_22k.pth'
lang_model_name = './bert-base-uncased'

model = dict(
    type='GroundingDINOPTSegSAM',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
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
        out_indices=(1, 2, 3),
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
        type='GroundingDINOHeadPT',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
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
            ])),
    test_cfg=dict(max_per_img=300))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(
    #     type='RandomSamplingNegPos',
    #     tokenizer_name=lang_model_name,
    #     num_sample_negative=85,
    #     max_tokens=256),
    dict(type='ReplacePrompt'),
    dict(type='ConcatPrompt'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='ConcatPrompt'),
    # dict(type='UnifyGT', label_map={0: 0, 2: 1}),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]



train_dataset_type = 'CityscapesWithCocoDataset'
train_data_root = 'data/cityscapes/'
# test_dataset_type = 'RoadAnomalyDataset'
# test_data_root = 'data/RoadAnomaly'
test_dataset_type = 'FSLostAndFoundDataset'
test_data_root = 'data/FS_LostFound'
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
                                        img_path='images', seg_map_path='labels_masks'),))
                                        # img_path='original', seg_map_path='labels'),))
                                        # img_path='leftImg8bit/test', seg_map_path='gtCoarse/test'),))
test_dataloader = val_dataloader
# val_evaluator = dict(type='AnomalyMetricRbA')
# val_evaluator = dict(type='BlankMetric')
val_evaluator = dict(type='AnomalyIoUMetric2')
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
    visualization=dict(type='GroundingVisualizationHook', draw=False, interval=1, score_thr=0.0))

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