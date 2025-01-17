from glob import glob
import os

images = glob('data/FS_LostFound/images/*.png')
# images = glob('data/FS_Static/images/*.jpg')
# images = glob('data/LostAndFound/leftImg8bit/train/01_Hanns_Klemm_Str_45/*.png')
# images = glob('data/RoadAnomaly/frames/*.webp')
# images = glob('data/LostAndFound/leftImg8bit/test/02_Hanns_Klemm_Str_44/*.png')
# images = ['data/FS_LostFound/images/3.png']

# images = ['./data/LostAndFound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000001_000220_leftImg8bit.png']
# images = ['./data/LostAndFound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000250_leftImg8bit.png',
        #   './data/LostAndFound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000270_leftImg8bit.png',
        #   './data/LostAndFound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000001_000240_leftImg8bit.png']
images.sort()

for image in images[:]:
    # image = 'data/objects365v1/val/obj365_val_000000354609.jpg'
    # os.system(f"python demo/image_demo.py \
    #                         {image} \
    #                         configs/anomaly_detection/detsegmask_swin-t_cityscapes_test.py \
    #                         --pred-score-thr 0 \
    #                         --weights work_dirs/detsegmask_swin-t_cityscapes_anomaly_detseg/iter_10000.pth \
    #                         --texts 'person. rider. car. truck. bus. train. motorcycle. bicycle. traffic light. traffic sign'")

    # os.system(f"python demo/image_demo.py \
    #                         {image} \
    #                         configs/anomaly_detection/glip_atss_swin-t_cityscapes_test.py \
    #                         --pred-score-thr 0.05 \
    #                         --weights ckpts/glip_l_mmdet-abfe026b.pth \
    #                         --texts '$: coco'")

    # os.system(f"python demo/image_demo.py \
    #                         {image} \
    #                         configs/anomaly_detection/grounding_dino_test.py \
    #                         --pred-score-thr 0.0 \
    #                         --weights work_dirs/grounding_dino_swin-t_coco_bbyy/epoch_2.pth \
    #                         --texts 'bbyy'")
    
    # configs/detseg/grounding_dino_bbyy_swin-b_finetune_obj365.py \
    # configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_all.py \
    # configs/detseg/grounding_dino_bbyy_swin-b_detseg.py
    
    # ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
    # work_dirs/grounding_dino_bbyy_swin-b_finetune_obj365/iter_5000.pth
    
    #  os.system(f"python demo/image_demo.py \
    #                         {image} \
    #                         configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_all.py \
    #                         --pred-score-thr 0.2 \
    #                         --weights ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth \
    #                         --texts ''")
    
    
    
    os.system(f"python demo/image_demo.py \
                            {image} \
                            configs/detseg/grounding_dino_bbyy_swin-b_sam_seg_test.py \
                            --pred-score-thr 0.2 \
                            --weights iter_5000.pth \
                            --texts 'road. sidewalk. building. wall. fence. pole. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. train. motorcycle. bicycle'")
    
    
    # os.system(f"python demo/image_demo.py \
    #                         {image} \
    #                         configs/anomaly_detection/detsegmask_swin-t_cityscapes_test.py \
    #                         --pred-score-thr 0.0 \
    #                         --weights work_dirs/detsegmask_swin-t_cityscapes_anomaly_detseg/iter_10000.pth \
    #                         --texts 'road. sidewalk. building. wall. fence. pole. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. train. motorcycle. bicycle'")

