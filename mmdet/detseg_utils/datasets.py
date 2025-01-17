from mmdet.registry import DATASETS, TRANSFORMS
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets.cityscapes import CityscapesDataset

import mmcv
from mmcv.transforms.base import BaseTransform
import mmengine
import mmengine.fileio as fileio
from mmengine.fileio import list_from_file, get_local_path
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.dataset import BaseDataset, Compose, force_full_init
from ..datasets.coco import CocoDataset
from ..datasets.base_det_dataset import BaseDetDataset

import random
import os
import os.path as osp
import pickle
import copy
import cv2
import math
import numpy as np
import logging
from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torchvision import transforms
from torch.utils.data import Dataset
import json
import glob
from PIL import Image


@TRANSFORMS.register_module()
class ConcatPrompt(BaseTransform):
    def transform(self, results: dict) -> dict:
        if (isinstance(results['text'], str)):
            # results['text'] = 'All foreground objects. ' + results['text']
            results['text'] = 'objects. ' + results['text']
            if 'tokens_positive' in results:
                if results['tokens_positive'] is not None:
                    for k, v in results['tokens_positive'].items():
                        results['tokens_positive'][k] = (np.array(v) + 9).tolist()
                        # results['tokens_positive'][k] = (np.array(v) + 24).tolist()
                        # results['tokens_positive'][k] = [[v[0][0] + 24, v[0][1] + 24]]
        else:
            # results['text'] = 'All foreground objects. ' + '. '.join(results['text'])
            # results['text'] = ('All foreground objects', ) + results['text']
            results['text'] = ('objects', ) + results['text']
            if 'tokens_positive' in results:
                if results['tokens_positive'] is not None:
                    for k, v in results['tokens_positive'].items():
                        results['tokens_positive'][k] = (np.array(v) + 9).tolist()
        return results


@TRANSFORMS.register_module()
class AddPrompt(BaseTransform):
    def __init__(self, text = ('objects', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                                'motorcycle', 'bicycle')):
        self.text = text

    def transform(self, results: dict) -> dict:
        results['text'] = self.text
        results['gt_bboxes_labels'] = np.zeros_like(results['gt_bboxes_labels'])
        return results


@TRANSFORMS.register_module()
class ReplacePrompt(BaseTransform):
    def transform(self, results: dict) -> dict:
        if (isinstance(results['text'], str)):
            results['text'] = 'road. sidewalk. building. wall. fence. pole. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. train. motorcycle. bicycle'
        else:
            results['text'] = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                                'motorcycle', 'bicycle')
        results['gt_bboxes_labels'] = np.zeros_like(results['gt_bboxes_labels'])
        return results


@DATASETS.register_module()
class RoadAnomalyDataset(Dataset):
    # METAINFO = dict(
    #     classes=('not anomaly', 'anomaly'),
    #     palette=[[128, 64, 128], [244, 35, 232]])
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                 (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                 (107, 142, 35), (152, 251, 152), (70, 130, 180),
                 (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                 (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])

    def __init__(self, 
                 data_root: str = None, 
                 pipeline: List[Union[dict, Callable]] = [], 
                 caption_prompt = None, 
                 **kwargs):
        with open(os.path.join(data_root, 'frame_list.json'), 'r') as f:
            self.img_list = json.load(f)
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.caption_prompt = caption_prompt
        self.metainfo = RoadAnomalyDataset.METAINFO
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        data_info = {'img_path': os.path.join(self.data_root, 'frames', self.img_list[idx])}
        data_info['reduce_zero_label'] = False
        data_info['seg_map_path'] = os.path.join(self.data_root, 'frames', \
                        self.img_list[idx].replace('webp', 'labels'), 'labels_semantic.png')
        data_info['seg_fields'] = []
        data_info['text'] = self.metainfo['classes']
        data_info = self.pipeline(data_info)
        return data_info




@DATASETS.register_module()
class FSLostAndFoundDataset(BaseSegDataset):
    # METAINFO = dict(
    # classes=('normal', 'anomaly'),
    # palette=[[128, 64, 128], [244, 35, 232]])

    # METAINFO = dict(
    #     classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    #              'traffic light', 'traffic sign', 'vegetation', 'terrain',
    #              'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    #              'motorcycle', 'bicycle', 'bbyy'),
    #     palette=[(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    #              (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    #              (107, 142, 35), (152, 251, 152), (70, 130, 180),
    #              (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    #              (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 255, 0)])

    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                 (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                 (107, 142, 35), (152, 251, 152), (70, 130, 180),
                 (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                 (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])

    # METAINFO = dict(
    #     classes=('bbyy', ),
    #     palette=[(0, 255, 0)])
    
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png', 
                 caption_prompt = None, 
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        self.caption_prompt = caption_prompt
    
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        
        data_info['text'] = self.metainfo['classes']
        data_info['caption_prompt'] = self.caption_prompt
        data_info['custom_entities'] = True
        data_info['img_id'] = idx


        return data_info


@DATASETS.register_module()
class CityscapesDatasetDetSeg(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle'),
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    }

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        self.img_suffix = img_suffix
        super().__init__(seg_map_suffix=seg_map_suffix, **kwargs)
        

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            all_is_crowd = all([
                instance['ignore_flag'] == 1
                for instance in data_info['instances']
            ])
            if filter_empty_gt and (img_id not in ids_in_cat or all_is_crowd):
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].replace(self.img_suffix, self.seg_map_suffix))
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

# @DATASETS.register_module()
# class LostAndFoundDataset(BaseSegDataset):

#     METAINFO = dict(
#         classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
#                  'traffic light', 'traffic sign', 'vegetation', 'terrain',
#                  'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
#                  'motorcycle', 'bicycle'),
#         palette=[(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
#                  (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
#                  (107, 142, 35), (152, 251, 152), (70, 130, 180),
#                  (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
#                  (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])
    
#     def __init__(self,
#                  img_suffix='_leftImg8bit.png',
#                  seg_map_suffix='_gtCoarse_labelTrainIds.png', 
#                  caption_prompt = None,
#                  return_classes = False,
#                  sequences_split_num = 10,
#                  use_sequence_group_flag = True,
#                  **kwargs) -> None:
#         self.caption_prompt = caption_prompt
#         self.return_classes = return_classes
#         self.sequences_split_num = sequences_split_num
#         self.use_sequence_group_flag = use_sequence_group_flag
#         super().__init__(
#             img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        
            
#     @force_full_init
#     def get_data_info(self, idx: int) -> dict:
#         """Get annotation by index and automatically call ``full_init`` if the
#         dataset has not been fully initialized.

#         Args:
#             idx (int): The index of data.

#         Returns:
#             dict: The idx-th annotation of the dataset.
#         """
#         if self.serialize_data:
#             start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
#             end_addr = self.data_address[idx].item()
#             bytes = memoryview(
#                 self.data_bytes[start_addr:end_addr])  # type: ignore
#             data_info = pickle.loads(bytes)  # type: ignore
#         else:
#             data_info = copy.deepcopy(self.data_list[idx])
#         # Some codebase needs `sample_idx` of data information. Here we convert
#         # the idx to a positive number and save it in data information.
#         if idx >= 0:
#             data_info['sample_idx'] = idx
#         else:
#             data_info['sample_idx'] = len(self) + idx
        
#         if self.return_classes:
#             data_info['text'] = self.metainfo['classes']
#             data_info['caption_prompt'] = self.caption_prompt
#             data_info['custom_entities'] = True
#         data_info['img_id'] = idx

#         return data_info
    
#     def _set_sequence_group_flag(self):
#         """
#         Set each sequence to be a different group
#         """
#         res = []

#         curr_sequence = 0
        
#         for idx in range(len(self.data_list)):
#             if idx != 0 and self.data_list[idx]['video_id'] != self.data_list[idx - 1]['video_id']:
#                 # Not first frame and # of sweeps is 0 -> new sequence
#                 curr_sequence += 1
#             res.append(curr_sequence)

#         self.flag = np.array(res, dtype=np.int64)
        
#         if self.sequences_split_num != 1:
#             if self.sequences_split_num == 'all':
#                 self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
#             else:
#                 bin_counts = np.bincount(self.flag)
#                 new_flags = []
#                 curr_new_flag = 0
#                 for curr_flag in range(len(bin_counts)):
#                     curr_sequence_length = np.array(
#                         list(range(0, 
#                                 bin_counts[curr_flag], 
#                                 math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
#                         + [bin_counts[curr_flag]])

#                     for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
#                         for _ in range(sub_seq_idx):
#                             new_flags.append(curr_new_flag)
#                         curr_new_flag += 1
                
#                 assert len(new_flags) == len(self.flag)
#                 assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
#                 self.flag = np.array(new_flags, dtype=np.int64)
    
#     def full_init(self):
#         """Load annotation file and set ``BaseDataset._fully_initialized`` to
#         True.

#         If ``lazy_init=False``, ``full_init`` will be called during the
#         instantiation and ``self._fully_initialized`` will be set to True. If
#         ``obj._fully_initialized=False``, the class method decorated by
#         ``force_full_init`` will call ``full_init`` automatically.

#         Several steps to initialize annotation:

#             - load_data_list: Load annotations from annotation file.
#             - filter data information: Filter annotations according to
#               filter_cfg.
#             - slice_data: Slice dataset according to ``self._indices``
#             - serialize_data: Serialize ``self.data_list`` if
#               ``self.serialize_data`` is True.
#         """
#         if self._fully_initialized:
#             return
#         # load data information
#         self.data_list = self.load_data_list()
#         # filter illegal data, such as data that has no annotations.
#         self.data_list = self.filter_data()
#         # Get subset data according to indices.
#         if self._indices is not None:
#             self.data_list = self._get_unserialized_subset(self._indices)

        
#         for data_info in self.data_list:
#             data_info['video_id'] = data_info['img_path'].split('/')[-1].split('_')[0]
#         if self.use_sequence_group_flag:
#             self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
        
#         # serialize data_list
#         if self.serialize_data:
#             self.data_bytes, self.data_address = self._serialize_data()

#         self._fully_initialized = True


class CocoSemSeg(Dataset):
    def __init__(self, data_root: str, proxy_size: int = 5000, split: str = "train",
                 transform: Optional[Callable] = None, shuffle=True) -> None:
        """
        COCO dataset loader
        """
        self.data_root = data_root
        self.coco_year = '2017'
        self.split = split + self.coco_year
        self.images = []
        self.targets = []
        self.transform = transform
                
        for filename in glob.glob(os.path.join(data_root, "annotations", "ood_seg_" + self.split, '*.png')):
            self.targets.append(filename)
            self.images.append(filename.replace('annotations/', '').replace('ood_seg_', '').replace('.png', '.jpg'))
        
        if shuffle:
            zipped = list(zip(self.images, self.targets))
            random.shuffle(zipped)
            self.images, self.targets = zip(*zipped)

        self.images = list(self.images[:proxy_size])
        self.targets = list(self.targets[:proxy_size])
        
    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Return raw image and ground truth in PIL format or as torch tensor"""
        image = cv2.imread(self.images[idx])
        target = cv2.imread(self.targets[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __repr__(self):
        """Return number of images in each dataset."""

        fmt_str = 'Number of COCO Images: %d\n' % len(self.images)
        return fmt_str.strip()


@DATASETS.register_module()
class CityscapesWithCocoDataset(CityscapesDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    
    
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle', 'anomaly'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], 
                 [0, 255, 0]])
    def __init__(self, 
                 coco_file_path,
                 return_classes: bool = True,
                 img_size = (1024, 2048),
                 **kwargs) -> None:
        self.coco_dataset = CocoSemSeg(coco_file_path)
        self.return_classes = return_classes
        super().__init__(**kwargs)        
    
    def filter_data(self) -> List[dict]:
        return self.data_list
    
    def get_data_info(self, idx: int) -> dict:
        
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        
        data_info['coco_data'] = self.coco_dataset[np.random.randint(0, len(self.coco_dataset))]
        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = None
            data_info['custom_entities'] = True
        
        return data_info
    
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
    

@TRANSFORMS.register_module()
class PasteCocoObjects(BaseTransform):
    
    def __init__(self, 
                 mix_ratio=0.2):
        super().__init__()
        self.mix_ratio = mix_ratio
        
    #Source: https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """        
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x2 += 1
                y2 += 1
            else:
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)
    
    
    def mix_object(self, current_labeled_image, current_labeled_mask, cut_object_image, cut_object_mask):
        train_id_out = 254

        cut_object_mask[cut_object_mask == train_id_out] = 254

        mask = cut_object_mask == 254

        ood_mask = np.expand_dims(mask, axis=2)
        ood_boxes = self.extract_bboxes(ood_mask)
        ood_boxes = ood_boxes[0, :]  # (y1, x1, y2, x2)
        y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
        cut_object_mask = cut_object_mask[y1:y2, x1:x2]
        cut_object_image = cut_object_image[y1:y2, x1:x2, :]

        mask = cut_object_mask == 254

        idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

        if mask.shape[0] != 0:
            h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
            h_end_point = h_start_point + cut_object_mask.shape[0]
            w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
            w_end_point = w_start_point + cut_object_mask.shape[1]
        else:
            h_start_point = 0
            h_end_point = 0
            w_start_point = 0
            w_end_point = 0

        current_labeled_image[h_start_point:h_end_point, w_start_point:w_end_point, :][np.where(idx == 254)] = \
            cut_object_image[np.where(idx == 254)]
        current_labeled_mask[h_start_point:h_end_point, w_start_point:w_end_point][np.where(cut_object_mask == 254)] = \
            cut_object_mask[np.where(cut_object_mask == 254)]
        current_labeled_mask[current_labeled_mask == 254] = 19
        return current_labeled_image, current_labeled_mask
    
    def transform(self, results: dict) -> dict:
        
        if np.random.uniform() < self.mix_ratio:
            coco_img, coco_gt = results['coco_data']
            img, sem_seg_gt = self.mix_object(current_labeled_image=results['img'], \
                current_labeled_mask=results['gt_seg_map'], cut_object_image=coco_img, cut_object_mask=coco_gt)
            results['img'] = img
            results['gt_seg_map'] = sem_seg_gt
        
        r = random.randint(0, 10000)
        if r < 100:
            img = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
            Image.fromarray(img).save(f'samples/{r}.jpg')
        
        return results
    

@TRANSFORMS.register_module()
class UnifyGT(BaseTransform):
    def __init__(self, label_map={0: 0, 1: 1}):
        super().__init__()
        self.label_map = label_map
    
    def transform(self, results: dict) -> dict:
        new_gt_seg_map = np.zeros_like(results['gt_seg_map'])
        for k, v in self.label_map.items():
            new_gt_seg_map[results['gt_seg_map'] == k] = v
        results['gt_seg_map'] = new_gt_seg_map
        return results



@DATASETS.register_module()
class SMIYCDataset(BaseDataset):
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                 (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                 (107, 142, 35), (152, 251, 152), (70, 130, 180),
                 (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                 (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])
    # METAINFO = dict(
    #     classes=('not anomaly', 'anomaly'),
    #     palette=[[128, 64, 128], [244, 35, 232]])
    def __init__(self,
                 data_root: str = None, 
                 data_info: str = None, 
                 img_suffix: str = '.jpg', 
                 split: str = 'validation',
                 **kwargs) -> None:
        super().__init__(lazy_init=True, serialize_data=False, data_root=data_root, **kwargs)
        self.img_list = glob.glob(os.path.join(data_root, 'images', f'*{img_suffix}'))
        if split == 'validation':
            self.img_list = [img_path for img_path in self.img_list if 'validation' in img_path]
        self.img_suffix = img_suffix

    def full_init(self):
        pass
    
    def __len__(self):
        return len(self.img_list)
    
    def get_data_info(self, idx: int) -> dict:
        data_info = dict()
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        
        data_info['img_path'] = self.img_list[idx]
        data_info['reduce_zero_label'] = False
        data_info['seg_map_path'] = self.img_list[idx].replace('images', 'labels_masks').replace(self.img_suffix, '_labels_semantic.png')
        # if not os.path.exists(data_info['seg_map_path']):
        # img = cv2.imread(data_info['img_path'])
        # cv2.imwrite(data_info['seg_map_path'], np.zeros((*img.shape[:2], 1)))
        data_info['text'] = self.metainfo['classes']
        data_info['seg_fields'] = []

        return data_info


@TRANSFORMS.register_module()
class GetAnomalyScoreMap(BaseTransform):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
    
    def transform(self, results: dict) -> dict:
        anomaly_score_map = np.load(os.path.join(self.data_path, os.path.basename(results['img_path'])).replace('webp', 'jpg').replace('image_', '') + '.npy')
        # anomaly_score_map = np.load(os.path.join(self.data_path, os.path.basename(results['img_path'])).replace('webp', 'jpg').replace('image_', '').replace('.png', '.npy').replace('_leftImg8bit', ''))
        results['anomaly_score_map'] = anomaly_score_map
        return results



@DATASETS.register_module()
class LostAndFoundDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtCoarse_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        
        data_info['text'] = self.metainfo['classes']
        data_info['custom_entities'] = True
        data_info['img_id'] = idx


        return data_info



@DATASETS.register_module()
class SODADDataset(CocoDataset):
    METAINFO = {
        'classes':
        ('people', 'rider', 'bicycle', 'motor', 'vehicle',
        'traffic-sign', 'traffic-light', 'traffic-camera',
        'warning-cone'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192)]
    }

    def __init__(self,
                 ori_ann_file,
                 **kwargs):
        super(SODADDataset, self).__init__(**kwargs)
        # self.ori_infos = self.load_ori_annotations(ori_ann_file)
        # self.ori_coco = COCO(ori_ann_file)
        # self.ori_img_ids = self.ori_coco.getImgIds()
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def load_ori_annotations(self, ori_ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.ori_coco = COCO(ori_ann_file)
        cat_ids = self.ori_coco.getCatIds(catNms=self.CLASSES)
        cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.ori_img_ids = self.ori_coco.getImgIds()
        ori_infos = []
        for i in self.ori_img_ids:
            info = self.ori_coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            ori_infos.append(info)
        return ori_infos

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos



    # def _get_ori_img_ids(self):
    #     ids = set()
    #     for i, class_id in enumerate(self.cat_ids):
    #         ids |= set(self.ori_coco.cat_img_map[class_id])
    #     self.ori_img_ids = list(ids)

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self.ori_img_ids)):
            img_id = self.ori_img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        # assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def translate(self, bboxes, x, y):
        dim = bboxes.shape[-1]
        translated = bboxes + \
                     np.array([x, y] * int(dim / 2), dtype=np.float32)
        return translated

    def merge_dets(self,
                           results,
                           with_merge=True,
                           nms_iou_thr=0.5,
                           nproc=10,
                           save_dir=None,
                           **kwargs):
        if mmcv.is_list_of(results, tuple):
            dets, segms = results
        else:
            dets = results

        # get patch results for evaluating
        if not with_merge:
            results = [(data_info['id'], result)
                       for data_info, result in zip(self.data_infos, results)]
            # TODO:
            if save_dir is not None:
                pass
            return results

        print('\n>>> Merge detected results of patch for whole image evaluating...')
        start_time = time.time()
        collector = defaultdict(list)
        # ensure data_infos and dets have the same length
        for data_info, result in zip(self.data_infos, dets):
            x_start, y_start = data_info['start_coord']
            new_result = []
            for i, res in enumerate(result):
                bboxes, scores = res[:, :-1], res[:, [-1]]
                bboxes = self.translate(bboxes, x_start, y_start)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(np.concatenate(
                    [labels, bboxes, scores], axis=1
                ))

            new_result = np.concatenate(new_result, axis=0)
            collector[data_info['ori_id']].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=nms_iou_thr)
        if nproc > 1:
            pool = Pool(nproc)
            merged_results = pool.map(merge_func, list(collector.items()))
            pool.close()
        else:
            merged_results = list(map(merge_func, list(collector.items())))

        # TODO:
        if save_dir is not None:
            pass

        stop_time = time.time()
        print('Merge results completed, it costs %.1f seconds.' % (stop_time - start_time))
        return merged_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 with_merge=True):
        """Evaluation in COCO protocol.
        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        merged_results = self.merge_dets(
            results=results,
            with_merge=with_merge,
            nms_iou_thr=0.5,
            nproc=8,
            save_dir=None
        )

        img_ids = [result[0] for result in merged_results]
        results = [result[1] for result in merged_results]

        # sort the detection results based on the original annotation order which
        # conform to COCO API.
        empty_result = [np.empty((0, 5), dtype=np.float32) for cls in self.CLASSES]
        sort_results = []
        for ori_img_id in self.ori_img_ids:
            if ori_img_id in img_ids:
                sort_results.append(results[img_ids.index(ori_img_id)])
            else:
                sort_results.append(empty_result)
        results = sort_results

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        """ Evaluating detection results on top of COCO API """
        eval_results = {}
        cocoGt = self.ori_coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            SODADEval = SODADeval(cocoGt, cocoDt, iou_type)
            SODADEval.params.catIds = self.cat_ids  # self.cat_ids
            SODADEval.params.imgIds = self.ori_img_ids
            SODADEval.params.maxDets = list(proposal_nums)
            SODADEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            tod_metric_names = {
                'AP': 0,
                'AP_50': 1,
                'AP_75': 2,
                'AP_eS': 3,
                'AP_rS': 4,
                'AP_gS': 5,
                'AP_Normal': 6,
                'AR@100': 7,
                'AR@300': 8,
                'AR@1000': 9,
                'AR_eS@1000': 10,
                'AR_rS@1000': 11,
                'AR_gS@1000': 12,
                'AR_Normal@1000': 13
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in tod_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                SODADEval.params.useCats = 0
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS',
                        'AP_rS', 'AP_gS', 'AP_Normal'
                    ]

                for item in metric_items:
                    val = float(
                        f'{SODADEval.stats[tod_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()

                if True:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = SODADEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.ori_coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS',
                        'AP_rS', 'AP_gS', 'AP_Normal'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{SODADEval.stats[tod_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = SODADEval.stats[:7]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} '
                    f'{ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f} '
                    f'{ap[6]:.3f} '
                )

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

def _merge_func(info, CLASSES, iou_thr):
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)
    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    ori_img_results = []
    for i in range(len(CLASSES)):
        cls_dets = dets[labels == i]
        if len(cls_dets) == 0:
            ori_img_results.append(np.empty((0, dets.shape[1]), dtype=np.float32))
            continue
        bboxes, scores = cls_dets[:, :-1], cls_dets[:, [-1]]
        bboxes = torch.from_numpy(bboxes).to(torch.float32).contiguous()
        scores = torch.from_numpy(np.squeeze(scores, 1)).to(torch.float32).contiguous()
        results, inds = nms(bboxes, scores, iou_thr)
        # If scores.shape=(N, 1) instead of (N, ), then the results after NMS
        # will be wrong. Suppose bboxes.shape=[N, 4], the results has shape
        # of [N**2, 5],
        results = results.numpy()
        ori_img_results.append(results)
    return img_id, ori_img_results
