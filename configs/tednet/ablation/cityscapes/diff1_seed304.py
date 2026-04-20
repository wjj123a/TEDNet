
_base_ = ["../../tednet_cityscapes-1024x1024.py"]

ablation_group = "structure"
ablation_variant = "diff1"
ablation_seed = 304

work_dir = "work_dirs/ablation/cityscapes/diff1/seed_304"
randomness = dict(seed=304)

model = {'backbone': {'use_transformer_encoder': True,
              'attention_type': 'improved',
              'detail_mode': 'diff',
              'diff_count': 1},
 'decode_head': {'use_boundary_loss': True, 'use_detail_loss': True}}

val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(type="BoundaryIoUMetric", num_classes=19, ignore_index=255),
]
test_evaluator = val_evaluator
