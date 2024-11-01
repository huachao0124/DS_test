_base_ = '../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

data_root = 'data/coco/'

model = dict(language_model=dict(name='./bert-base-uncased'))
# model = dict(bbox_head=dict(with_iou_pred=True))

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

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoClassAgnosticDataset',
        # type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/'),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(data_prefix=dict(img='images/val2017/'),))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

# learning policy
# max_epochs = 12
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]
train_cfg = dict(max_epochs=6, val_interval=6)
# default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='auto'))

# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', interval=1000, by_epoch=False, max_keep_ckpts=1, save_best='auto'))
# train_cfg = dict(
#     _delete_=True, 
#     type='IterBasedTrainLoop', max_iters=5000, val_interval=5000)
# log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)



load_from = 'ckpts/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
