from glob import glob
import os
import multiprocessing
from itertools import cycle

def process_image(args):
    image, gpu_id = args
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = f"CUDA_VISIBLE_DEVICES={str(gpu_id)} python demo/image_demo.py \
            {image} \
            configs/detseg/grounding_dino_bbyy_swin-b_sam_seg_test.py \
            --pred-score-thr 0.2 \
            --weights iter_5000.pth\
            --texts 'road. sidewalk. building. wall. fence. pole. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. train. motorcycle. bicycle'"
    
    os.system(cmd)

def main():
    # 获取所有图片
    # images = glob('data/FS_LostFound/images/*.png')
    # images.sort()
    # images = ['data/FS_LostFound/images/4.png', 'data/FS_LostFound/images/1.png', 'data/FS_LostFound/images/2.png', 'data/FS_LostFound/images/3.png']
    # images = glob('/mnt/nj-public02/usr/xiangyiwei/zhuhuachao/DS_test/data/RoadAnomaly/frames/*.webp')
    images = glob('/mnt/nj-public02/usr/xiangyiwei/zhuhuachao/DS_test/data/SMIYC/dataset_ObstacleTrack/images/validation*')
    # 创建GPU ID列表
    gpu_ids = [0, 1, 2, 3]  # 4张GPU
    
    # 将图片和GPU ID配对
    args = list(zip(images, cycle(gpu_ids)))
    
    # 创建进程池
    num_processes = len(gpu_ids)
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 并行处理图片
    pool.map(process_image, args)
    
    # 关闭进程池
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
