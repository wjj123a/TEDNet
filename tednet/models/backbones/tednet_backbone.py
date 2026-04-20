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


class ZeroAttention(nn.Module):

    def forward(self, x):
        return torch.zeros_like(x)


class SEAttention(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ECAAttention(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CBAMAttention(nn.Module):

    def __init__(self,
                 in_channels: int,
                 reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1))
        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attn = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_attn = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * self.sigmoid(avg_attn + max_attn)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        spatial_attn = self.sigmoid(
            self.spatial(torch.cat([avg_map, max_map], dim=1)))
        return x * spatial_attn


def build_attention(attention_type: str,
                    in_channels: int,
                    qk_channels: int,
                    v_channels: int,
                    num_heads: int,
                    norm_cfg: OptConfigType):
    attention_type = attention_type.lower()
    if attention_type == "improved":
        return ImprovedSelfAttention(
            in_channels=in_channels,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            norm_cfg=norm_cfg)
    if attention_type == "none":
        return ZeroAttention()
    if attention_type == "cbam":
        return CBAMAttention(in_channels)
    if attention_type == "se":
        return SEAttention(in_channels)
    if attention_type == "eca":
        return ECAAttention(in_channels)
    raise ValueError(f"Unsupported attention_type: {attention_type}")


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
                 attention_type: str = "improved",
                 norm_cfg: OptConfigType = dict(
                     type="BN", requires_grad=True),
                 act_cfg: OptConfigType = dict(type="ReLU", inplace=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
        self.attn = build_attention(
            attention_type=attention_type,
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


@MODELS.register_module(name="TEDNet_Backbone", force=True)
class TEDNet_Backbone(BaseModule):

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_heads: int = 8,
                 qk_channels: int = 128,
                 use_transformer_encoder: bool = True,
                 attention_type: str = "improved",
                 detail_mode: str = "diff",
                 diff_count: int = 2,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(
                     type="BN", requires_grad=True),
                 act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        detail_mode = detail_mode.lower()
        if detail_mode not in {"diff", "spatial", "none"}:
            raise ValueError(f"Unsupported detail_mode: {detail_mode}")
        if diff_count < 0 or diff_count > 3:
            raise ValueError("diff_count must be in the range [0, 3].")

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.use_transformer_encoder = use_transformer_encoder
        self.attention_type = attention_type
        self.detail_mode = detail_mode
        self.diff_count = diff_count
        self.use_detail_path = detail_mode != "none" and not (
            detail_mode == "diff" and diff_count == 0)

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
        if use_transformer_encoder:
            self.transformer_encoder = TransformerEncoderBlock(
                in_channels=transformer_channels,
                qk_channels=qk_channels,
                v_channels=qk_channels * 2,
                num_heads=num_heads,
                ffn_expansion=2,
                attention_type=attention_type,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.transformer_encoder = nn.Identity()

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

        if self.detail_mode == "diff" and self.diff_count >= 1:
            self.diff_1 = DiffModule(
                channels * 4, channels * 2, norm_cfg=norm_cfg)
        if self.detail_mode == "diff" and self.diff_count >= 2:
            self.diff_2 = DiffModule(
                channels * 8, channels * 2, norm_cfg=norm_cfg)
        if self.detail_mode == "diff" and self.diff_count >= 3:
            self.diff_3 = DiffModule(
                transformer_channels, channels * 2, norm_cfg=norm_cfg)

        if self.use_detail_path:
            num_detail_blocks = 2 if self.detail_mode == "spatial" else max(
                self.diff_count, 1)
            self.detail_branch_layers = nn.ModuleList([
                self._make_layer(
                    BasicBlock, channels * 2, channels * 2, num_blocks=1)
                for _ in range(num_detail_blocks)
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

        detail_features = []
        x_d = None
        temp_spatial = None
        temp_detail = None

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

        if self.use_detail_path:
            if self.detail_mode == "diff":
                diff_c = self.diff_1(self.relu(x_c), out_size)
                x_d = self.detail_branch_layers[0](diff_c)
            else:
                x_d = self.detail_branch_layers[0](self.relu(x_s))

        if self.training:
            if self.use_detail_path:
                detail_features.append(x_s.clone())
            temp_spatial = x_s.clone()

        x_c = self.context_branch_layers[1](self.relu(x_c))
        x_s = self.spatial_branch_layers[1](self.relu(x_s))

        comp_c = self.compression_2(self.relu(x_c))
        x_c = x_c + self.down_2(self.relu(x_s))
        x_s = x_s + resize(
            comp_c,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners)

        if self.use_detail_path:
            if self.detail_mode == "diff" and self.diff_count >= 2:
                diff_c = self.diff_2(self.relu(x_c), out_size)
                x_d = x_d + diff_c
                x_d = self.detail_branch_layers[1](self.relu(x_d))
            elif self.detail_mode == "spatial":
                x_d = self.detail_branch_layers[1](self.relu(x_d + x_s))

        if self.training:
            if self.use_detail_path:
                detail_features.append(x_s.clone())
                temp_detail = x_d.clone()
            else:
                temp_detail = temp_spatial.clone()

        x_s = self.spatial_branch_layers[2](self.relu(x_s))
        x_c = self.context_branch_layers[2](self.relu(x_c))

        if self.training:
            if self.use_detail_path:
                detail_features.append(x_s.clone())

        if self.use_detail_path and self.detail_mode == "diff" and self.diff_count >= 3:
            diff_c = self.diff_3(self.relu(x_c), out_size)
            x_d = x_d + diff_c
            x_d = self.detail_branch_layers[2](self.relu(x_d))
            if self.training:
                temp_detail = x_d.clone()

        x_c = self.transformer_encoder(x_c)
        x_c = self.spp(x_c)
        x_c = resize(
            x_c,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners)

        out = x_s + x_c
        if self.use_detail_path:
            x_d = self.detail_proj(self.relu(x_d))
            out = out + x_d

        if self.training:
            return temp_spatial, out, temp_detail, detail_features
        return out
