import os
import time
import pickle
import numpy as np
import cv2
import random
from PIL import Image
import torch
import threading
from tqdm import tqdm

import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmseg.structures import SegDataSample

from mmdet.visualization.local_visualizer import DetLocalVisualizer
from mmengine.dist import master_only
from mmdet.registry import VISUALIZERS

@VISUALIZERS.register_module()
class VisualizerHeatMap(DetLocalVisualizer):
    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)
            if 'gt_sem_seg' in data_sample:
                gt_img_data = self._draw_sem_seg(gt_img_data,
                                                 data_sample.gt_sem_seg,
                                                 classes, palette)

            if 'gt_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_panoptic_seg(
                    gt_img_data, data_sample.gt_panoptic_seg, classes, palette)

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)

            if 'pred_sem_seg' in data_sample:
                pred_img_data = self._draw_sem_seg(pred_img_data,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette)

            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes, palette)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        seg_logits = data_sample.seg_logits.data.cpu()
        # heatmap = -seg_logits[:19].tanh().sum(dim=0).numpy()
        heatmap = 1 - torch.max(seg_logits[:19], dim=0)[0].numpy()
        # heatmap = seg_logits[1].numpy() - seg_logits[0].numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
        drawn_img = np.concatenate((drawn_img, heatmap), axis=1)

        # drawn_img = heatmap
        # drawn_img = self.plot_mask_on_img(image, data_sample.pred_sem_seg.sem_seg.cpu().numpy())
        
        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)


    def plot_mask_on_img(self, img, mask):
        red_mask = np.zeros_like(img)
        mask = mask.squeeze(0)

        red_mask[:, :, :1][mask == 1] = 255  # 设置红色通道为1
        red_mask_on_img = img.copy()
        red_mask_on_img[:, :, :1][mask == 1] = 0.1 * img[:, :, :1][mask == 1] + 0.9 * red_mask[:, :, :1][mask == 1]
        # red_mask_on_img[:, :, 1:2][mask == 1] = 0 * img[:, :, 1:2][mask == 1] + 1 * red_mask[:, :, 1:2][mask == 1]
        # red_mask_on_img[:, :, 1:2][mask == 1] = 0 * img[:, :, 1:2][mask == 1] + 1 * red_mask[:, :, 1:2][mask == 1]
        return red_mask_on_img