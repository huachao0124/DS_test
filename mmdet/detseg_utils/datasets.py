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
            # results['text'] = ('objects', ) + results['text']
            pass
        return results



@DATASETS.register_module()
class RoadAnomalyDataset(Dataset):
    # METAINFO = dict(
    #     classes=('anomaly', 'not anomaly'),
    #     palette=[[128, 64, 128], [244, 35, 232]])
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
                 data_root: str = None, 
                 pipeline: List[Union[dict, Callable]] = [], 
                 caption_prompt = None, 
                 **kwargs):
        with open(os.path.join(data_root, 'frame_list.json'), 'r') as f:
            self.img_list = json.load(f)
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.caption_prompt = caption_prompt
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        data_info = {'img_path': os.path.join(self.data_root, 'frames', self.img_list[idx])}
        data_info['reduce_zero_label'] = False
        data_info['seg_map_path'] = os.path.join(self.data_root, 'frames', \
                        self.img_list[idx].replace('jpg', 'labels'), 'labels_semantic.png')
        data_info['seg_fields'] = []
        
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

@DATASETS.register_module()
class LostAndFoundDataset(BaseSegDataset):

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
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtCoarse_labelTrainIds.png', 
                 caption_prompt = None,
                 return_classes = False,
                 sequences_split_num = 10,
                 use_sequence_group_flag = True,
                 **kwargs) -> None:
        self.caption_prompt = caption_prompt
        self.return_classes = return_classes
        self.sequences_split_num = sequences_split_num
        self.use_sequence_group_flag = use_sequence_group_flag
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
        
        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True
        data_info['img_id'] = idx

        return data_info
    
    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        
        for idx in range(len(self.data_list)):
            if idx != 0 and self.data_list[idx]['video_id'] != self.data_list[idx - 1]['video_id']:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)
        
        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1
                
                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)
    
    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        
        for data_info in self.data_list:
            data_info['video_id'] = data_info['img_path'].split('/')[-1].split('_')[0]
        if self.use_sequence_group_flag:
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
        
        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True


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
            self.images.append(filename.replace('annotations', 'images').replace('ood_seg_', '').replace('.png', '.jpg'))
        
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
    
    