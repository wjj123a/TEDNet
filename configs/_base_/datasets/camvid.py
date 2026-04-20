import os.path as osp

dataset_type = "CamVidDataset"
_project_root = osp.abspath(osp.join("{{ fileDirname }}", "../../.."))
data_root = osp.join(_project_root, "data", "CamVid")
crop_size = (960, 720)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize",
        scale=(960, 720),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs")
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(960, 720), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs")
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(type="Resize", scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type="RandomFlip", prob=0., direction="horizontal"),
                dict(type="RandomFlip", prob=1., direction="horizontal")
            ], [dict(type="LoadAnnotations")], [dict(type="PackSegInputs")]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="train", seg_map_path="train_labels_indexed"),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="val", seg_map_path="val_labels_indexed"),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="test", seg_map_path="test_labels_indexed"),
        pipeline=test_pipeline))

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
