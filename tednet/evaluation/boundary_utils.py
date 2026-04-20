from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    kernel_size = radius * 2 + 1
    for y in range(kernel_size):
        for x in range(kernel_size):
            out &= padded[y:y + height, x:x + width]
    return out


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    kernel_size = radius * 2 + 1
    for y in range(kernel_size):
        for x in range(kernel_size):
            out |= padded[y:y + height, x:x + width]
    return out


def mask_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Return a one-pixel inner boundary for a binary segmentation mask."""
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~_binary_erode(mask, radius=radius)


def segmentation_counts(pred: np.ndarray,
                        target: np.ndarray,
                        num_classes: int,
                        ignore_index: int = 255) -> tuple[np.ndarray,
                                                          np.ndarray]:
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()
    valid = target != ignore_index
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    for cls_idx in range(num_classes):
        pred_mask = (pred == cls_idx) & valid
        target_mask = (target == cls_idx) & valid
        intersections[cls_idx] = np.logical_and(pred_mask, target_mask).sum()
        unions[cls_idx] = np.logical_or(pred_mask, target_mask).sum()
    return intersections, unions


def boundary_counts(pred: np.ndarray,
                    target: np.ndarray,
                    num_classes: int,
                    ignore_index: int = 255,
                    boundary_width: int = 3) -> dict[str, np.ndarray]:
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()
    valid = target != ignore_index

    biou_intersections = np.zeros(num_classes, dtype=np.float64)
    biou_unions = np.zeros(num_classes, dtype=np.float64)
    pred_matches = np.zeros(num_classes, dtype=np.float64)
    pred_totals = np.zeros(num_classes, dtype=np.float64)
    target_matches = np.zeros(num_classes, dtype=np.float64)
    target_totals = np.zeros(num_classes, dtype=np.float64)

    for cls_idx in range(num_classes):
        pred_mask = (pred == cls_idx) & valid
        target_mask = (target == cls_idx) & valid
        if not pred_mask.any() and not target_mask.any():
            continue

        pred_boundary = mask_boundary(pred_mask)
        target_boundary = mask_boundary(target_mask)
        pred_boundary = pred_boundary & valid
        target_boundary = target_boundary & valid

        pred_band = _binary_dilate(pred_boundary, radius=boundary_width) & valid
        target_band = _binary_dilate(
            target_boundary, radius=boundary_width) & valid

        biou_intersections[cls_idx] = np.logical_and(
            pred_boundary, target_band).sum()
        biou_unions[cls_idx] = np.logical_or(
            pred_boundary, target_boundary).sum()

        pred_matches[cls_idx] = np.logical_and(
            pred_boundary, target_band).sum()
        pred_totals[cls_idx] = pred_boundary.sum()
        target_matches[cls_idx] = np.logical_and(
            target_boundary, pred_band).sum()
        target_totals[cls_idx] = target_boundary.sum()

    return dict(
        biou_intersections=biou_intersections,
        biou_unions=biou_unions,
        pred_matches=pred_matches,
        pred_totals=pred_totals,
        target_matches=target_matches,
        target_totals=target_totals)


def mean_from_counts(values: np.ndarray,
                     totals: np.ndarray,
                     classes: Optional[Iterable[int]] = None) -> float:
    if classes is not None:
        classes = np.asarray(list(classes), dtype=np.int64)
        values = values[classes]
        totals = totals[classes]
    valid = totals > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(values[valid] / totals[valid]))


def bfscore_from_counts(pred_matches: np.ndarray,
                        pred_totals: np.ndarray,
                        target_matches: np.ndarray,
                        target_totals: np.ndarray,
                        classes: Optional[Iterable[int]] = None) -> float:
    if classes is not None:
        classes = np.asarray(list(classes), dtype=np.int64)
        pred_matches = pred_matches[classes]
        pred_totals = pred_totals[classes]
        target_matches = target_matches[classes]
        target_totals = target_totals[classes]

    valid = (pred_totals > 0) | (target_totals > 0)
    if not valid.any():
        return float("nan")

    scores = []
    for cls_idx in np.where(valid)[0]:
        precision = pred_matches[cls_idx] / max(pred_totals[cls_idx], 1.0)
        recall = target_matches[cls_idx] / max(target_totals[cls_idx], 1.0)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(float(2 * precision * recall /
                                (precision + recall)))
    return float(np.mean(scores))
