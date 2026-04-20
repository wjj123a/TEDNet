
_base_ = ["../../tednet_cityscapes-1024x1024.py"]

ablation_group = "attention"
ablation_variant = "attn_cbam"
ablation_seed = 305

work_dir = "work_dirs/ablation/cityscapes/attn_cbam/seed_305"
randomness = dict(seed=305)

model = {'backbone': {'use_transformer_encoder': True,
              'attention_type': 'cbam',
              'detail_mode': 'diff',
              'diff_count': 2},
 'decode_head': {'use_boundary_loss': True, 'use_detail_loss': True}}

val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(type="BoundaryIoUMetric", num_classes=19, ignore_index=255),
]
test_evaluator = val_evaluator
