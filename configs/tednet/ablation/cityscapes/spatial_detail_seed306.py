
_base_ = ["../../tednet_cityscapes-1024x1024.py"]

ablation_group = "structure"
ablation_variant = "spatial_detail"
ablation_seed = 306

work_dir = "work_dirs/ablation/cityscapes/spatial_detail/seed_306"
randomness = dict(seed=306)

model = {'backbone': {'use_transformer_encoder': True,
              'attention_type': 'improved',
              'detail_mode': 'spatial',
              'diff_count': 0},
 'decode_head': {'use_boundary_loss': True, 'use_detail_loss': True}}

val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(type="BoundaryIoUMetric", num_classes=19, ignore_index=255),
]
test_evaluator = val_evaluator
