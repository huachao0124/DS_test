import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmdet.registry import METRICS
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode


@METRICS.register_module()
class AnomalyMetric(BaseMetric):
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
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            self.results.append((data_sample['seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['sem_seg'].cpu().numpy()))
        
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 2
        
        seg_logits = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])
                
        has_anomaly = np.array([(1 in np.unique(gt_anomaly_map)) for gt_anomaly_map in gt_anomaly_maps]).astype(np.bool_)
        
        # seg_logits = seg_logits[has_anomaly]
        # gt_anomaly_maps = gt_anomaly_maps[has_anomaly].flatten()        
                
        # pred_anomaly_maps = seg_logits[:, 19, :, :].flatten()
        pred_anomaly_maps = (1 - np.max(seg_logits[:, :19, :, :], axis=1)).flatten()
        # pred_anomaly_maps = seg_logits[:, 19, :, :].flatten() * (1 - np.max(seg_logits[:, :19, :, :], axis=1)).flatten()
        # pred_anomaly_maps = seg_logits[:, -1, :, :].flatten() / np.max(seg_logits[:, :19, :, :], axis=1).flatten()
        gt_anomaly_maps = gt_anomaly_maps.flatten()
        
        # assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
        ood_mask = (gt_anomaly_maps == 1)
        ind_mask = (gt_anomaly_maps == 0)

        ood_out = pred_anomaly_maps[ood_mask]
        ind_out = pred_anomaly_maps[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        fpr, tpr, _ = roc_curve(val_label, val_out)    
        roc_auc = auc(fpr, tpr)
        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


@METRICS.register_module()
class AnomalyMetricRbA(BaseMetric):
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
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            self.results.append((data_sample['seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['sem_seg'].cpu().numpy()))
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 2
        
        seg_logits = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])
                
        has_anomaly = np.array([(1 in np.unique(gt_anomaly_map)) for gt_anomaly_map in gt_anomaly_maps]).astype(np.bool_)
        
        # seg_logits = seg_logits[has_anomaly]
        # gt_anomaly_maps = gt_anomaly_maps[has_anomaly]    
        
                
        pred_anomaly_maps = -torch.from_numpy(seg_logits[:, :19]).tanh().sum(dim=1).numpy().flatten()
        gt_anomaly_maps = gt_anomaly_maps.flatten()
        
        # assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
        ood_mask = (gt_anomaly_maps == 1)
        ind_mask = (gt_anomaly_maps == 0)

        ood_out = pred_anomaly_maps[ood_mask]
        ind_out = pred_anomaly_maps[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        fpr, tpr, _ = roc_curve(val_label, val_out)    
        roc_auc = auc(fpr, tpr)
        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'FS_LF'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['sem_seg'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['sem_seg'].squeeze().to(
                    pred_label)
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics

@METRICS.register_module()
class AnomalyIoUMetric(BaseMetric):
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
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            self.results.append((data_sample['pred_masks']['sem_seg'].cpu(), data_sample['gt_sem_seg']['sem_seg'].cpu(), data_sample['pred_instances']))
    
    def calculate_miou(self, gt_anomaly_map, pred_anomaly_map):
        ious = []
        f1_scores = []
        for i in range(len(gt_anomaly_map)):
            intersection = torch.logical_and(gt_anomaly_map[i]==1, pred_anomaly_map[i]==1).sum().item()
            union = torch.logical_or(gt_anomaly_map[i]==1, pred_anomaly_map[i]==1).sum().item()
            union = torch.logical_and(gt_anomaly_map[i]!=255, torch.logical_or(gt_anomaly_map[i]==1, pred_anomaly_map[i]==1)).sum().item()
            # union = torch.logical_or(gt_anomaly_map[i]==1, pred_anomaly_map[i]==1).sum().item()
            if union == 0:
                iou = 0
                f1_score = 0
            else:
                iou = intersection / union
                f1_score = (2 * intersection) / (intersection + union)
            ious.append(iou)
            f1_scores.append(f1_score)
        return np.mean(ious), np.mean(f1_scores)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        # assert len(results) == 2
        
        pred_anomaly_maps = results[0]
        gt_anomaly_maps = torch.stack(results[1])
        pred_instances = results[2]
        
        ious = []
        f1_scores = []
        Best_miou = 0
        Best_threshold = 0
        threshold_list = []
        num_set = 0
        step = 0.01

        anomaly_list = []

        # for masks, single_pred_instances in zip(pred_anomaly_maps, pred_instances):
        #     anomaly = torch.zeros_like(masks[0]).float()
        #     for mask, score in zip(masks, single_pred_instances['scores']):
        #         anomaly[mask > 0] = score #       score
        #     anomaly_list.append(anomaly)
        
        # anomaly = torch.stack(anomaly_list)

        anomaly = torch.stack(pred_anomaly_maps)
        left, right = anomaly.min(), anomaly.max()

        for threshold in np.arange(left - step, right + step, step):
            num_set += 1
            threshold_list.append(threshold)
            anomaly_mask = torch.zeros_like(anomaly)
            anomaly_mask[anomaly > threshold] = 1
            iou, f1_score = self.calculate_miou(gt_anomaly_maps, anomaly_mask)
            if iou > Best_miou:
                Best_miou = iou
                Best_threshold = threshold
            ious.append(iou)
            f1_scores.append(f1_score)   
        f1_score = np.mean(f1_scores)
        area = np.trapz(ious, threshold_list)

        # summary
        metrics = dict()
        for key, val in zip(('IoU', 'F1'), (Best_miou, area)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'FS_LF'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


@METRICS.register_module()
class AnomalyIoUMetric2(BaseMetric):
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
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            self.results.append((data_sample['pred_sem_seg']['sem_seg'].cpu(), data_sample['gt_sem_seg']['sem_seg'].cpu(), data_sample['pred_instances']))
    
    def calculate_miou(self, gt_anomaly_map, pred_anomaly_map):
        ious = []
        f1_scores = []
        for i in range(len(gt_anomaly_map)):
            intersection = torch.logical_and(gt_anomaly_map[i]==1, pred_anomaly_map[i]==1).sum().item()
            # union = torch.logical_or(gt_anomaly_map[i]==1, pred_anomaly_map[i]==1).sum().item()
            union = torch.logical_and(gt_anomaly_map[i]!=255, torch.logical_or(gt_anomaly_map[i]==1, pred_anomaly_map[i]==1)).sum().item()
            if union == 0:
                iou = 0
                f1_score = 0
            else:
                iou = intersection / union
                f1_score = (2 * intersection) / (intersection + union)
            ious.append(iou)
            f1_scores.append(f1_score)
            print(i, iou)
        return np.mean(ious), np.mean(f1_scores)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        # assert len(results) == 2
        
        pred_anomaly_maps = torch.stack(results[0])
        gt_anomaly_maps = torch.stack(results[1])
        pred_instances = results[2]
        
        iou, f1_score = self.calculate_miou(gt_anomaly_maps, pred_anomaly_maps)
        # summary
        metrics = dict()
        for key, val in zip(('IoU', 'F1'), (iou, f1_score)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'FS_LF'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics




@METRICS.register_module()
class AnomalyMetricLoad(BaseMetric):
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
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            self.results.append((data_sample['anomaly_scores']['data'].cpu().numpy(), data_sample['gt_sem_seg']['sem_seg'].cpu().numpy()))
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        
        pred_anomaly_maps = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])

        pred_anomaly_maps = pred_anomaly_maps.flatten()
        gt_anomaly_maps = gt_anomaly_maps.flatten()
                
        ood_mask = (gt_anomaly_maps == 1)
        ind_mask = (gt_anomaly_maps == 0)

        ood_out = pred_anomaly_maps[ood_mask]
        ind_out = pred_anomaly_maps[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        fpr, tpr, _ = roc_curve(val_label, val_out)    
        roc_auc = auc(fpr, tpr)
        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'FS_LF'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


@METRICS.register_module()
class BlankMetric(BaseMetric):
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
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        self.results.append(0)
        
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        prc_auc = 0
        fpr = 0
        roc_auc = 0

        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics
