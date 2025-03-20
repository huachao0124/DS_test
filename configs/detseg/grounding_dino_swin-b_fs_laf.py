_base_ = '../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

load_from = '/mnt/nj-public02/usr/xiangyiwei/zhuhuachao/DS_test/ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth'  # noqa

crop_size = (1024, 512)
lang_model_name = './bert-base-uncased'

model = dict(
    use_autocast=True,
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
        _delete_=True,
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
        init_cfg=None),
    neck=dict(in_channels=[256, 512, 1024]),
)

dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
class_name = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
              'bicycle')
palette = [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
           (0, 80, 100), (0, 0, 230), (119, 11, 32)]

metainfo = dict(classes=class_name, palette=palette)

train_pipeline = [
    dict(type='LoadImageFromFile'),
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
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
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
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='ConcatPrompt'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

train_dataloader = dict(
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=train_pipeline,
            return_classes=True,
            data_prefix=dict(img='leftImg8bit/train/'),
            ann_file='annotations/instancesonly_filtered_gtFine_train.json')))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        metainfo=metainfo,
        data_root=data_root,
        return_classes=True,
        ann_file='annotations/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img='leftImg8bit/val/')))
test_dataloader = val_dataloader


# val_evaluator = dict(type='AnomalyMetricRbA')
val_evaluator = dict(type='BlankMetric')
# val_evaluator = dict(type='AnomalyIoUMetric')
# val_evaluator = dict(type='AnomalyMetricLoad')
test_evaluator = val_evaluator

# training schedule for 90k
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=5000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationWithResizeHook', draw=True, interval=1))
    visualization=dict(type='GroundingVisualizationHook', draw=True, interval=30, score_thr=0.2))

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
