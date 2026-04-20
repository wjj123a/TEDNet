import os.path as osp

_base_ = [
    "../_base_/datasets/camvid.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(imports=["tednet"], allow_failed_imports=False)

class_weight = [1.0] * 11
checkpoint = osp.abspath(
    osp.join("{{ fileDirname }}", "../../checkpoints/best.pth"))

crop_size = (960, 720)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

norm_cfg = dict(type="SyncBN", requires_grad=True)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="TEDNet_Backbone",
        in_channels=3,
        channels=64,
        ppm_channels=128,
        num_heads=8,
        qk_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint)),
    decode_head=dict(
        type="TEDNet_Head",
        in_channels=64 * 4,
        channels=128,
        aux_in_channels=64 * 2,
        boundary_in_channels=64 * 2,
        detail_channels=128,
        dropout_ratio=0.,
        num_classes=11,
        align_corners=False,
        norm_cfg=norm_cfg,
        use_detail_loss=True,
        detail_loss_weight=1.0,
        boundary_loss_weight=0.2,
        loss_decode=[
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.2),
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"))

train_dataloader = dict(batch_size=6, num_workers=4)

iters = 80000

optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=iters, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1000,
        save_best="mIoU",
        rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"))

randomness = dict(seed=304)

model_wrapper_cfg = dict(
    type="MMDistributedDataParallel", find_unused_parameters=True)
