# TEDNet

This repository contains the official implementation of TEDNet, the method
described in the manuscript submitted to *The Visual Computer*:

**Towards Boundary-Aware Real-Time Semantic Segmentation with Transformer and
Detail Enhancement**

TEDNet is a dual-resolution real-time semantic segmentation network that combines
Transformer-based context encoding with differential feature guidance and
detail-aware boundary recovery.

Readers who use this code should cite the related manuscript. The archived
source code and reproducibility package are available through Zenodo at
<https://doi.org/10.5281/zenodo.20065873>.

## Repository status

- GitHub: <https://github.com/wjj123a/TEDNet>
- Zenodo DOI: <https://doi.org/10.5281/zenodo.20065873>
- License: Apache-2.0
- Framework: PyTorch + OpenMMLab MMSegmentation

## Main results

The revised manuscript reports the following TEDNet results.

| Dataset | Split | Resolution | mIoU (%) | FPS | GPU | Params |
| --- | --- | --- | ---: | ---: | --- | ---: |
| Cityscapes | validation | 1024x2048 | 80.45 | 68.9 | RTX 3090 | 27.1M |
| CamVid | test | 720x960 | 83.72 | 113.2 | RTX 3090 | 27.1M |
| ADE20K | validation/test setting in manuscript | 512x512 | 37.56 | 117.9 | RTX 3090 | 27.1M |
| COCO-Stuff10K | validation/test setting in manuscript | 640x640 | 31.93 | 94.4 | RTX 3090 | 27.1M |

FPS is measured with batch size 1 on a single RTX 3090 using PyTorch
2.7.0+cu128, CUDA 12.8, and FP16 automatic mixed precision. When comparing with
published methods, note that some baselines report speed on different GPUs.

## Installation

Create a fresh environment and install the dependencies used for the revised
manuscript experiments. The reported environment uses Python 3.10.0, PyTorch
2.7.0+cu128, TorchVision 0.22.0+cu128, CUDA 12.8, cuDNN 9.7.1, MMCV 2.2.0,
MMEngine 0.10.7, MMSegmentation 1.2.2, NumPy 2.2.6, and OpenCV 4.12.0.

```bash
conda create -n tednet python=3.10 -y
conda activate tednet

pip install -r requirements.txt
```

If `mmcv` installation fails on your platform, follow the official OpenMMLab
installation guide, or install MMCV 2.2.0 from source for the CUDA and PyTorch
versions above.

## Dataset layout

The default configs read datasets from `data/`.

```text
TEDNet/
  data/
    cityscapes/
      leftImg8bit/
      gtFine/
    CamVid/
      images/
      labels/
    ADEChallengeData2016/
      images/
      annotations/
    coco_stuff10k/
      images/
      annotations/
```

Use the official dataset download pages and licenses:

- Cityscapes: <https://www.cityscapes-dataset.com/>
- CamVid: <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/>
- ADE20K: <https://groups.csail.mit.edu/vision/datasets/ADE20K/>
- COCO-Stuff10K: <https://github.com/nightrome/cocostuff10k>

Adjust paths in `configs/_base_/datasets/*.py` if your local layout differs.

## Checkpoints

The configs initialize TEDNet with the DDRNet-23 ImageNet-pretrained weights
provided by OpenMMLab:

<https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23-in1kpre_3rdparty-9ca29f62.pth>

The config files expect this checkpoint to be available at:

```text
checkpoints/best.pth
```

Download and rename it before training:

```bash
mkdir -p checkpoints
wget -O checkpoints/best.pth https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23-in1kpre_3rdparty-9ca29f62.pth
```

Large checkpoints are intentionally not tracked by git. The file
`checkpoints/best.pth` is only used for initialization; trained TEDNet
checkpoints should be saved under `work_dirs/` or another local output
directory.

## Usage

Follow this workflow to reproduce the main experiments.

### 1. Prepare the project


Place the public datasets under `data/` as shown above and download the
DDRNet-23 ImageNet initialization checkpoint to `checkpoints/best.pth`. If your
dataset or checkpoint paths differ, update the corresponding files in
`configs/_base_/` and `configs/tednet/`.

### 2. Choose a config

| Experiment | Config | Crop size | Iterations | Total batch size |
| --- | --- | ---: | ---: | ---: |
| Cityscapes | `configs/tednet/tednet_cityscapes-1024x1024.py` | `1024x1024` | 160k | 12 |
| CamVid | `configs/tednet/tednet_camvid-960x720.py` | `960x720` | 80k | 12 |
| ADE20K | `configs/tednet/tednet_ade20k-512x512.py` | `512x512` | 160k | 8 |
| COCO-Stuff10K | `configs/tednet/tednet_coco-stuff10k-640x640.py` | `640x640` | 80k | 12 |

The Cityscapes ablation configs are stored in
`configs/tednet/ablation/cityscapes/`. Each ablation variant is provided with
seeds `304`, `305`, and `306` for manuscript reproducibility.

### 3. Train, evaluate, and benchmark

Train TEDNet on Cityscapes:

```bash
python tools/train.py configs/tednet/tednet_cityscapes-1024x1024.py --work-dir work_dirs/tednet_cityscapes
```

Evaluate the best checkpoint:

```bash
python tools/test.py configs/tednet/tednet_cityscapes-1024x1024.py work_dirs/tednet_cityscapes/best_mIoU.pth
```

Measure FPS at the manuscript input resolution:

```bash
python tools/benchmark_fps.py configs/tednet/tednet_cityscapes-1024x1024.py work_dirs/tednet_cityscapes/best_mIoU.pth --shape 1024 2048 --device cuda:0
```

## Training

Single-GPU examples:

```bash
python tools/train.py configs/tednet/tednet_cityscapes-1024x1024.py
python tools/train.py configs/tednet/tednet_camvid-960x720.py
python tools/train.py configs/tednet/tednet_ade20k-512x512.py
python tools/train.py configs/tednet/tednet_coco-stuff10k-640x640.py
```

Specify an output directory:

```bash
python tools/train.py configs/tednet/tednet_cityscapes-1024x1024.py --work-dir work_dirs/tednet_cityscapes
```

Resume training:

```bash
python tools/train.py configs/tednet/tednet_cityscapes-1024x1024.py --resume --work-dir work_dirs/tednet_cityscapes
```

Multi-GPU example:

```bash
bash tools/dist_train.sh configs/tednet/tednet_cityscapes-1024x1024.py 2
```

## Evaluation and speed measurement

Evaluate a trained checkpoint with the included OpenMMLab test runner:

```bash
python tools/test.py configs/tednet/tednet_cityscapes-1024x1024.py work_dirs/tednet_cityscapes/best_mIoU.pth
```

For FPS reporting, use batch size 1, the same image resolution as the result
table, and a warm-up period before timing. Report the GPU model, CUDA version,
PyTorch version, input resolution, and whether batch normalization or other
operator fusion is applied.

## Reproducibility notes

- Training hardware: two NVIDIA L20 GPUs.
- Evaluation and FPS hardware: one NVIDIA RTX 3090 GPU.
- Software environment: Python `3.10.0`, PyTorch `2.7.0+cu128`,
  TorchVision `0.22.0+cu128`, CUDA `12.8`, cuDNN `9.7.1`, MMCV `2.2.0`,
  MMEngine `0.10.7`, MMSegmentation `1.2.2`, NumPy `2.2.6`, and OpenCV
  `4.12.0`.
- Default random seed: `304` in the training configs.
- Optimizer: SGD with momentum `0.9`; dataset-specific learning rates and weight
  decay values are defined in each config file.
- Cityscapes config: 160k iterations, crop size `1024x1024`, total batch size `12` (`6` per GPU).
- CamVid config: 80k iterations, crop size `960x720`, total batch size `12` (`6` per GPU).
- ADE20K config: 160k iterations, crop size `512x512`, total batch size `8` (`4` per GPU).
- COCO-Stuff10K config: 80k iterations, crop size `640x640`, total batch size `12` (`6` per GPU).
- During inference, TEDNet keeps only the main segmentation output; auxiliary,
  boundary, and detail heads are training-only supervision branches.

## Citation

```bibtex
@article{wei2026tednet,
  title = {Towards Boundary-Aware Real-Time Semantic Segmentation with Transformer and Detail Enhancement},
  author = {Wei, Jingjie and Yuan, Qingni and Qu, Pengju and Jia, Wei},
  journal = {The Visual Computer},
  year = {2026},
  note = {Manuscript submitted},
  doi = {10.5281/zenodo.20065873}
}
```

## Acknowledgement

This implementation builds on PyTorch and the OpenMMLab ecosystem. Please also
cite the corresponding OpenMMLab projects when using this code.
