
_base_ = ["../../tednet_cityscapes-1024x1024.py"]

ablation_group = "supervision"
ablation_variant = "detail_w20"
ablation_seed = 304

work_dir = "work_dirs/ablation/cityscapes/detail_w20/seed_304"
randomness = dict(seed=304)

model = {'backbone': {'detail_mode': 'diff', 'diff_count': 2},
 'decode_head': {'detail_loss_weight': 2.0}}

val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(type="BoundaryIoUMetric", num_classes=19, ignore_index=255),
]
test_evaluator = val_evaluator
