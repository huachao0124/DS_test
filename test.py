from glob import glob
import os

images = glob('data/FS_LostFound/images/*.png')

for image in images[:4]:
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
    
    
    os.system(f"python demo/image_demo.py \
                            {image} \
                            configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_all.py \
                            --pred-score-thr 0.3 \
                            --weights ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth \
                            --texts 'road. sidewalk. building. wall. fence. pole. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. train. motorcycle. bicycle'")
    
    
    # os.system(f"python demo/image_demo.py \
    #                         {image} \
    #                         configs/anomaly_detection/detsegmask_swin-t_cityscapes_test.py \
    #                         --pred-score-thr 0.0 \
    #                         --weights work_dirs/detsegmask_swin-t_cityscapes_anomaly_detseg/iter_10000.pth \
    #                         --texts 'road. sidewalk. building. wall. fence. pole. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. train. motorcycle. bicycle'")