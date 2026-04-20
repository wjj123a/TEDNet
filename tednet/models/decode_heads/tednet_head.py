from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


class DetailHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 mid_channels: int = 64,
                 norm_cfg: OptConfigType = dict(
                     type="BN", requires_grad=True)):
        super().__init__()
        self.bn1 = build_norm_layer(norm_cfg, in_channels)[1]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn2 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class DetailLoss(nn.Module):

    def __init__(self, eps: float = 1.0, loss_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                         dtype=torch.float32).view(1, 1, 3, 3))

    def generate_detail_gt(self, gt: Tensor) -> Tensor:
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)

        gt_float = gt.float()
        detail = F.conv2d(gt_float, self.laplacian_kernel, padding=1)
        detail = (detail.abs() > 0).float()
        return detail

    def dice_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = torch.sigmoid(pred)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        pred_sum = (pred_flat * pred_flat).sum(1)
        target_sum = (target_flat * target_flat).sum(1)

        dice = (2 * intersection + self.eps) / (pred_sum + target_sum +
                                                self.eps)
        return (1 - dice).mean()

    def bce_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            pred, target, reduction="mean")

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        detail_gt = self.generate_detail_gt(gt)

        if pred.shape[-2:] != detail_gt.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=detail_gt.shape[-2:],
                mode="bilinear",
                align_corners=False)

        loss_dice = self.dice_loss(pred, detail_gt)
        loss_bce = self.bce_loss(pred, detail_gt)
        return self.loss_weight * (loss_dice + loss_bce)


@MODELS.register_module(name="TEDNet_Head", force=True)
class TEDNet_Head(BaseDecodeHead):

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 aux_in_channels: int = None,
                 boundary_in_channels: int = None,
                 detail_channels: int = 64,
                 use_aux_loss: bool = True,
                 use_boundary_loss: bool = True,
                 use_detail_loss: bool = True,
                 aux_loss_weight: float = 1.0,
                 detail_loss_weight: float = 1.0,
                 boundary_loss_weight: float = 0.2,
                 norm_cfg: OptConfigType = dict(type="BN"),
                 act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.aux_in_channels = aux_in_channels or in_channels // 2
        self.boundary_in_channels = boundary_in_channels or in_channels // 2
        self.use_aux_loss = use_aux_loss
        self.use_boundary_loss = use_boundary_loss
        self.use_detail_loss = use_detail_loss
        self.aux_loss_weight = aux_loss_weight
        self.detail_loss_weight = detail_loss_weight
        self.boundary_loss_weight = boundary_loss_weight

        self.head = self._make_base_head(self.in_channels, self.channels)
        if use_aux_loss:
            self.aux_head = self._make_base_head(self.aux_in_channels,
                                                 self.channels)
            self.aux_cls_seg = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=1)

        if use_boundary_loss:
            self.boundary_head = self._make_base_head(
                self.boundary_in_channels, self.channels)
            self.boundary_cls_seg = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=1)

        if use_detail_loss:
            self.detail_heads = nn.ModuleList([
                DetailHead(self.aux_in_channels, detail_channels, norm_cfg),
                DetailHead(self.aux_in_channels, detail_channels, norm_cfg),
                DetailHead(self.in_channels, detail_channels, norm_cfg),
            ])
            self.detail_loss = DetailLoss(loss_weight=detail_loss_weight)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(
        self, inputs: Union[Tensor, Tuple[Tensor]]
    ) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            temp_spatial, out, temp_detail, detail_features = inputs

            x_main = self.head(out)
            x_main = self.cls_seg(x_main)
            outputs = dict(main=x_main)

            if self.use_aux_loss:
                x_aux = self.aux_head(temp_spatial)
                outputs["aux"] = self.aux_cls_seg(x_aux)

            if self.use_boundary_loss:
                x_boundary = self.boundary_head(temp_detail)
                outputs["boundary"] = self.boundary_cls_seg(x_boundary)

            if self.use_detail_loss:
                detail_preds = []
                for detail_feat, detail_head in zip(detail_features,
                                                    self.detail_heads):
                    detail_preds.append(detail_head(detail_feat))
                outputs["detail_preds"] = detail_preds

            return outputs

        x_main = self.head(inputs)
        x_main = self.cls_seg(x_main)
        return x_main

    def _make_base_head(self, in_channels: int,
                        channels: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=("norm", "act", "conv")),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]
        return nn.Sequential(*layers)

    def _decode_loss(self, index: int):
        if isinstance(self.loss_decode, nn.ModuleList):
            if index < len(self.loss_decode):
                return self.loss_decode[index]
            return self.loss_decode[0]
        if isinstance(self.loss_decode, (list, tuple)):
            if index < len(self.loss_decode):
                return self.loss_decode[index]
            return self.loss_decode[0]
        return self.loss_decode

    def _parse_logits(self, seg_logits):
        if isinstance(seg_logits, dict):
            return (seg_logits["main"], seg_logits.get("aux"),
                    seg_logits.get("boundary"),
                    seg_logits.get("detail_preds"))

        if self.use_detail_loss and len(seg_logits) == 4:
            main_logit, aux_logit, boundary_logit, detail_preds = seg_logits
        else:
            main_logit, aux_logit, boundary_logit = seg_logits[:3]
            detail_preds = None
        return main_logit, aux_logit, boundary_logit, detail_preds

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        seg_label = self._stack_batch_gt(batch_data_samples)

        main_logit, aux_logit, boundary_logit, detail_preds = \
            self._parse_logits(seg_logits)

        main_logit = resize(
            main_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners)
        if aux_logit is not None:
            aux_logit = resize(
                aux_logit,
                size=seg_label.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners)
        if boundary_logit is not None:
            boundary_logit = resize(
                boundary_logit,
                size=seg_label.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)

        loss["loss_main"] = self._decode_loss(0)(
            main_logit, seg_label, ignore_index=self.ignore_index)
        if self.use_aux_loss and aux_logit is not None:
            loss["loss_aux"] = self._decode_loss(1)(
                aux_logit, seg_label,
                ignore_index=self.ignore_index) * self.aux_loss_weight

        if self.use_boundary_loss and boundary_logit is not None:
            loss["loss_boundary"] = self._decode_loss(2)(
                boundary_logit,
                seg_label,
                ignore_index=self.ignore_index) * self.boundary_loss_weight

        if self.use_detail_loss and detail_preds:
            detail_loss_total = 0
            for i, detail_pred in enumerate(detail_preds):
                scale = 2**i if i < 2 else 2
                if scale > 1:
                    scaled_label = F.interpolate(
                        seg_label.unsqueeze(1).float(),
                        scale_factor=1.0 / scale,
                        mode="nearest").squeeze(1).long()
                else:
                    scaled_label = seg_label
                detail_loss_total += self.detail_loss(detail_pred,
                                                      scaled_label)
            loss["loss_detail"] = detail_loss_total / len(detail_preds)

        loss["acc_seg"] = accuracy(
            main_logit, seg_label, ignore_index=self.ignore_index)
        return loss
