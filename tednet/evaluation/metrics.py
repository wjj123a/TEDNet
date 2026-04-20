from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmseg.registry import METRICS

from .boundary_utils import (bfscore_from_counts, boundary_counts,
                             mean_from_counts, segmentation_counts)


def _to_numpy(data) -> np.ndarray:
    if data.__class__.__name__ == "PixelData":
        data = data.data
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    return np.asarray(data).squeeze()


def _sample_value(sample, key: str):
    if isinstance(sample, dict):
        value = sample[key]
        if isinstance(value, dict) and "data" in value:
            return value["data"]
        return value
    return getattr(sample, key)


@METRICS.register_module(force=True)
class BoundaryIoUMetric(BaseMetric):
    """Boundary-focused semantic segmentation metric.

    The metric complements mIoU with boundary IoU, boundary F-score, and a
    Cityscapes thin-object group score for pole/light/sign/person/rider/two-wheel
    classes.
    """

    default_prefix = None

    def __init__(self,
                 num_classes: Optional[int] = None,
                 ignore_index: int = 255,
                 boundary_width: int = 3,
                 thin_classes: Sequence[int] = (5, 6, 7, 11, 12, 17, 18),
                 collect_device: str = "cpu",
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_width = boundary_width
        self.thin_classes = tuple(thin_classes)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = _to_numpy(_sample_value(data_sample, "pred_sem_seg"))
            target = _to_numpy(_sample_value(data_sample, "gt_sem_seg"))
            num_classes = self._num_classes(pred, target)

            iou_intersections, iou_unions = segmentation_counts(
                pred,
                target,
                num_classes=num_classes,
                ignore_index=self.ignore_index)
            counts = boundary_counts(
                pred,
                target,
                num_classes=num_classes,
                ignore_index=self.ignore_index,
                boundary_width=self.boundary_width)
            counts["iou_intersections"] = iou_intersections
            counts["iou_unions"] = iou_unions
            self.results.append(counts)

    def compute_metrics(self, results: list) -> dict:
        if not results:
            return {}

        totals = {}
        for key in results[0]:
            totals[key] = np.sum([result[key] for result in results], axis=0)

        thin_classes = tuple(cls for cls in self.thin_classes
                             if cls < len(totals["iou_unions"]))

        metrics = dict(
            mBIoU=mean_from_counts(totals["biou_intersections"],
                                   totals["biou_unions"]),
            BFScore=bfscore_from_counts(totals["pred_matches"],
                                        totals["pred_totals"],
                                        totals["target_matches"],
                                        totals["target_totals"]),
            thin_mIoU=mean_from_counts(totals["iou_intersections"],
                                       totals["iou_unions"], thin_classes),
            thin_BIoU=mean_from_counts(totals["biou_intersections"],
                                       totals["biou_unions"], thin_classes),
            thin_BFScore=bfscore_from_counts(totals["pred_matches"],
                                             totals["pred_totals"],
                                             totals["target_matches"],
                                             totals["target_totals"],
                                             thin_classes))
        return {
            key: round(float(value) * 100.0, 2)
            for key, value in metrics.items() if not np.isnan(value)
        }

    def _num_classes(self, pred: np.ndarray, target: np.ndarray) -> int:
        if self.num_classes is not None:
            return self.num_classes
        if self.dataset_meta and "classes" in self.dataset_meta:
            return len(self.dataset_meta["classes"])
        valid_target = target[target != self.ignore_index]
        max_pred = int(pred.max()) if pred.size else 0
        max_target = int(valid_target.max()) if valid_target.size else 0
        return max(max_pred, max_target) + 1
