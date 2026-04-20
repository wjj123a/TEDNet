
_base_ = ["../../tednet_cityscapes-1024x1024.py"]

ablation_group = "supervision"
ablation_variant = "no_boundary_loss"
ablation_seed = 305

work_dir = "work_dirs/ablation/cityscapes/no_boundary_loss/seed_305"
randomness = dict(seed=305)

model = {'backbone': {'detail_mode': 'diff', 'diff_count': 2},
 'decode_head': {'use_boundary_loss': False}}

val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(type="BoundaryIoUMetric", num_classes=19, ignore_index=255),
]
test_evaluator = val_evaluator
