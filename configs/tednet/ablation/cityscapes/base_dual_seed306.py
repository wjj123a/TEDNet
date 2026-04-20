
_base_ = ["../../tednet_cityscapes-1024x1024.py"]

ablation_group = "structure"
ablation_variant = "base_dual"
ablation_seed = 306

work_dir = "work_dirs/ablation/cityscapes/base_dual/seed_306"
randomness = dict(seed=306)

model = {'backbone': {'use_transformer_encoder': False,
              'attention_type': 'none',
              'detail_mode': 'none',
              'diff_count': 0},
 'decode_head': {'use_boundary_loss': False, 'use_detail_loss': False}}

val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(type="BoundaryIoUMetric", num_classes=19, ignore_index=255),
]
test_evaluator = val_evaluator
