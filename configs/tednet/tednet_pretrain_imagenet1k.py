_base_ = [
    "../_base_/datasets/imagenet1k_224x224.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(imports=["tednet"], allow_failed_imports=False)

norm_cfg = dict(type="BN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=(224, 224))

model = dict(
    type="ImageClassifier",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="TEDNet_Backbone_Cls",
        in_channels=3,
        channels=64,
        ppm_channels=128,
        num_heads=8,
        qk_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False),
    num_classes=1000,
    in_channels=256,
    label_smoothing=0.1,
    topk=(1, 5),
    loss=dict(type="CrossEntropyLoss", loss_weight=1.0))

train_dataloader = dict(batch_size=64, num_workers=8)
val_dataloader = dict(batch_size=64, num_workers=8)
test_dataloader = val_dataloader

max_epochs = 300

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.04))

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs - 5,
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=max_epochs,
        convert_to_iter_based=True),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_best="accuracy/top1",
        rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=False))

log_processor = dict(by_epoch=True)
val_evaluator = dict(type="Accuracy", topk=(1, 5))
test_evaluator = val_evaluator
randomness = dict(seed=42)
