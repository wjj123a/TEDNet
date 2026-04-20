import os.path as osp

dataset_type = "ImageNetDataset"
_project_root = osp.abspath(osp.join("{{ fileDirname }}", "../../.."))
data_root = osp.join(_project_root, "data", "imagenet")

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=(224, 224))

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RandomResizedCrop",
        size=224,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333)),
    dict(type="RandomHorizontalFlipCls", prob=0.5),
    dict(
        type="ColorJitter",
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1),
    dict(type="PackClsInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ResizeForCls", size=256),
    dict(type="CenterCrop", size=224),
    dict(type="PackClsInputs"),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix="train",
        ann_file="meta/train.txt",
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix="val",
        ann_file="meta/val.txt",
        pipeline=test_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

val_evaluator = dict(type="Accuracy", topk=(1, 5))
test_evaluator = val_evaluator
