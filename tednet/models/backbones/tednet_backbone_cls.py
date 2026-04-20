import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType


class ImprovedSelfAttention(nn.Module):

    def __init__(self,
                 in_channels: int,
                 qk_channels: int = 128,
                 v_channels: int = 256,
                 num_heads: int = 8,
                 norm_cfg: OptConfigType = dict(
                     type="BN", requires_grad=True)):
        super().__init__()
        self.in_channels = in_channels
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.num_heads = num_heads
        self.scale = (qk_channels // num_heads) ** -0.5

        self.q_conv = nn.Conv2d(in_channels, qk_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, qk_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, v_channels, kernel_size=1)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                v_channels,
                kernel_size=3,
                padding=1,
                groups=min(in_channels, v_channels)),
            nn.Hardswish(inplace=True))

        self.out_conv = nn.Conv2d(v_channels, in_channels, kernel_size=1)
        self.out_bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.hardswish = nn.Hardswish(inplace=True)

    def forward(self, x):
        batch_size, _, height, width = x.shape

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = q.view(batch_size, self.num_heads,
                   self.qk_channels // self.num_heads, height * width)
        q = q.transpose(-2, -1)
        k = k.view(batch_size, self.num_heads,
                   self.qk_channels // self.num_heads, height * width)
        k = k.transpose(-2, -1)
        v = v.view(batch_size, self.num_heads,
                   self.v_channels // self.num_heads, height * width)
        v = v.transpose(-2, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(batch_size,
                                                      self.v_channels, height,
                                                      width)

        dw_out = self.dw_conv(x)
        out = self.hardswish(out + dw_out)
        out = self.out_conv(out)
        out = self.out_bn(out)
        return out


class SAFFN(nn.Module):

    def __init__(self,
                 in_channels: int,
                 expansion_ratio: int = 2,
                 norm_cfg: OptConfigType = dict(
                     type="BN", requires_grad=True),
                 act_cfg: OptConfigType = dict(type="ReLU", inplace=True)):
        super().__init__()
        hidden_channels = in_channels * expansion_ratio

        self.in_conv = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.dw_conv_3x3 = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels)
        self.dw_conv_5x5 = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels)

        self.out_conv = ConvModule(
            in_channels * 2,
            in_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        hidden = self.in_conv(x)
        out_3x3 = self.dw_conv_3x3(hidden)
        out_5x5 = self.dw_conv_5x5(hidden)
        out = torch.cat([out_3x3, out_5x5], dim=1)
        return self.out_conv(out)


class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 qk_channels: int = 128,
                 v_channels: int = 256,
                 num_heads: int = 8,
                 ffn_expansion: int = 2,
                 norm_cfg: OptConfigType = dict(
                     type="BN", requires_grad=True),
                 act_cfg: OptConfigType = dict(type="ReLU", inplace=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
        self.attn = ImprovedSelfAttention(
            in_channels=in_channels,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            norm_cfg=norm_cfg)
        self.norm2 = build_norm_layer(norm_cfg, in_channels)[1]
        self.ffn = SAFFN(
            in_channels=in_channels,
            expansion_ratio=ffn_expansion,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DiffModule(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: OptConfigType = dict(type="BN")):
        super().__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x_context, target_size):
        x_up = F.interpolate(
            x_context,
            size=target_size,
            mode="bilinear",
            align_corners=False)
        return self.conv(x_up)


@MODELS.register_module(name="TEDNet_Backbone_Cls", force=True)
class TEDNet_Backbone_Cls(BaseModule):

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_heads: int = 8,
                 qk_channels: int = 128,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(
                     type="BN", requires_grad=True),
                 act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        self.context_branch_layers = nn.ModuleList()
        for i in range(3):
            self.context_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2**(i + 1),
                    planes=channels * 8 if i > 0 else channels * 4,
                    num_blocks=2 if i < 2 else 1,
                    stride=2))

        transformer_channels = channels * 16
        self.transformer_encoder = TransformerEncoderBlock(
            in_channels=transformer_channels,
            qk_channels=qk_channels,
            v_channels=qk_channels * 2,
            num_heads=num_heads,
            ffn_expansion=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))

        self.spatial_branch_layers = nn.ModuleList()
        for i in range(3):
            self.spatial_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2,
                    planes=channels * 2,
                    num_blocks=2 if i < 2 else 1))

        self.spp = DAPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5)

        self.diff_1 = DiffModule(channels * 4, channels * 2, norm_cfg=norm_cfg)
        self.diff_2 = DiffModule(channels * 8, channels * 2, norm_cfg=norm_cfg)

        self.detail_branch_layers = nn.ModuleList([
            self._make_layer(
                BasicBlock, channels * 2, channels * 2, num_blocks=1),
            self._make_layer(
                BasicBlock, channels * 2, channels * 2, num_blocks=1),
        ])

        self.detail_proj = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def _make_stem_layer(self, in_channels, channels, num_blocks):
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.extend([
            self._make_layer(BasicBlock, channels, channels, num_blocks),
            nn.ReLU(),
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2),
            nn.ReLU(),
        ])
        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None
                    if i == num_blocks - 1 else self.act_cfg))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x_c = self.context_branch_layers[0](x)
        x_s = self.spatial_branch_layers[0](x)
        out_size = x_s.shape[-2:]

        comp_c = self.compression_1(self.relu(x_c))
        x_c = x_c + self.down_1(self.relu(x_s))
        x_s = x_s + resize(
            comp_c,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners)

        diff_c = self.diff_1(self.relu(x_c), out_size)
        x_d = self.detail_branch_layers[0](diff_c)

        x_c = self.context_branch_layers[1](self.relu(x_c))
        x_s = self.spatial_branch_layers[1](self.relu(x_s))

        comp_c = self.compression_2(self.relu(x_c))
        x_c = x_c + self.down_2(self.relu(x_s))
        x_s = x_s + resize(
            comp_c,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners)

        diff_c = self.diff_2(self.relu(x_c), out_size)
        x_d = x_d + diff_c
        x_d = self.detail_branch_layers[1](self.relu(x_d))

        x_s = self.spatial_branch_layers[2](self.relu(x_s))
        x_c = self.context_branch_layers[2](self.relu(x_c))

        x_c = self.transformer_encoder(x_c)
        x_c = self.spp(x_c)
        x_c = resize(
            x_c,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners)

        x_d = self.detail_proj(self.relu(x_d))
        return x_s + x_c + x_d
