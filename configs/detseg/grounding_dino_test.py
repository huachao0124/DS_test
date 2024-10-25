_base_ = './grounding_dino_swin-t_finetune_16xb4_1x_coco.py'

# model = dict(
#              language_model=dict(name='./bert-base-uncased'),
#              bbox_head=dict(type='GroundingDINOHeadTwoB'))

model = dict(type='GroundingDINO',
            #  backbone=dict(
            #     out_indices=(0, 1, 2, 3)),
            #  language_model=dict(name='./bert-base-uncased'),
             bbox_head=dict(type='GroundingDINOHead'))

test_dataset_type = 'FSLostAndFoundDataset'
test_data_root = 'data/FS_LostFound'
# test_data_root = 'data/FS_Static'

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
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(_delete_=True, dataset=dict(type=test_dataset_type, 
                                                data_root=test_data_root, 
                                                pipeline=test_pipeline, 
                                                #  img_suffix='.jpg',
                                                data_prefix=dict(
                                                        img_path='images',
                                                        seg_map_path='labels_masks')))
test_dataloader = val_dataloader
