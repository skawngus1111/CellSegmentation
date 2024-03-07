import math
from functools import partial
from typing import List, Optional, Callable

import torch
import torch.nn as nn

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class DWConv(nn.Module):
    """
    Depthwise Convolution

    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with Depthwise Convolution

    Args:
        in_features (int): Number of input channels.
        hidden_features (int): Number of hidden channels.
        out_features (int): Number of output channels.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop_rate (float): Dropout rate. Default: 0.
        linear (bool): If True, use nn.Linear, else nn.Conv2d. Default: False
    """
    def __init__(self,
                 in_features: int,
                 hidden_features: int=None,
                 out_features: int=None,
                 act_layer: Optional[Callable[..., nn.Module]]=nn.GELU,
                 drop_rate: float=0.,
                 linear: bool=False) -> None:
        super(MLP, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Attention(nn.Module):
    """
    Attention layer.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate.
        proj_drop (float): Dropout rate.
        sr_ratio (int): Spatial reduction ratio.
        linear (bool): If True, use nn.Linear, else nn.Conv2d.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qkv_bias: bool=False,
                 qk_scale: Optional[float]=None,
                 attn_drop: float=0.,
                 proj_drop: float=0.,
                 sr_ratio: int=1,
                 linear: bool=False) -> None:
        super(Attention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, return_qk=False):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qk: return x, q, k
        else: return x

class Block(nn.Module):
    """
    Transformer block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Hidden dimension multiplier for mlp token.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate.
        attn_drop (float): Attention dropout rate.
        drop_path (float): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        sr_ratio (int): Spatial reduction ratio. Default: 1
        linear (bool): If True, use nn.Linear, else nn.Conv2d. Default: False
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float=4.,
                 qkv_bias: bool=False,
                 qk_scale: Optional[float]=None,
                 drop_rate: float=0.,
                 attn_drop: float=0.,
                 drop_path: float=0.,
                 act_layer: Optional[Callable[..., nn.Module]]=nn.GELU,
                 norm_layer: Optional[Callable[..., nn.Module]]=nn.LayerNorm,
                 sr_ratio: int=1,
                 linear: bool=False) -> None:
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop_rate,
                              sr_ratio=sr_ratio,
                              linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop_rate=drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, return_qk=False):
        if return_qk:
            x, q, k = self.attn(self.norm1(x), H, W, return_qk)
            x = x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

            return x, q, k
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

            return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding with overlapping

    Args:
        img_size (int): Image size.  Default: 224
        patch_size (int): Patch token size. Default: 4
        stride (int): Stride of patch embedding. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 768
    """
    def __init__(self,
                 img_size: int=224,
                 patch_size: int=4,
                 stride: int=4,
                 in_chans: int=3,
                 embed_dim: int=768) -> None:
        super(OverlapPatchEmbed, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class PyramidVisionTransformerV2(nn.Module):
    """
    Wang, Wenhai, et al. "Pvt v2: Improved baselines with pyramid vision transformer." Computational Visual Media 8.3 (2022): 415-424.

    Official PyTorch impl:
        https://github.com/whai362/PVT/tree/v2

    Args:
            img_size (int): Image size. Default: 224
            patch_size (int): Patch token size. Default: 4
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            embed_dims (List[int]): Embedding dimensions. Default: [64, 128, 320, 512]
            num_heads (List[int]): Number of attention heads. Default: [1, 2, 5, 8]
            mlp_ratios (List[int]): Hidden dimension multipliers for mlp tokens. Default: [8, 8, 4, 4]
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
            drop_rate (float): Dropout rate. Default: 0.
            attn_drop_rate (float): Attention dropout rate. Default: 0.
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
            depths (List[int]): Depths of each stage. Default: [3, 4, 6, 3]
            sr_ratios (List[int]): Spatial reduction ratios of each stage. Default: [8, 4, 2, 1]
            num_stages (int): Number of stages. Default: 4
            linear (bool): If True, use nn.Linear, else nn.Conv2d. Default: False
    """
    def __init__(self,
                 img_size: int=224,
                 patch_size: int=4,
                 in_chans: int=3,
                 num_classes: int=1000,
                 embed_dims: List[int]=[64, 128, 320, 512],
                 num_heads: List[int]=[1, 2, 4, 8],
                 mlp_ratios: List[int]=[4, 4, 4, 4],
                 qkv_bias: bool=True,
                 qk_scale: float=None,
                 drop_rate: float=0.,
                 attn_drop_rate: float=0.,
                 drop_path_rate: float=0.,
                 norm_layer: Optional[Callable[..., nn.Module]]=nn.LayerNorm,
                 depths: List[int]=[3, 4, 6, 3],
                 sr_ratios: List[int]=[8, 4, 2, 1],
                 num_stages: int = 4,
                 linear: bool=False) -> None:
        super(PyramidVisionTransformerV2, self).__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(dim=embed_dims[i],
                                         num_heads=num_heads[i],
                                         mlp_ratio=mlp_ratios[i],
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop_rate=drop_rate,
                                         attn_drop=attn_drop_rate,
                                         drop_path=dpr[cur + j],
                                         norm_layer=norm_layer,
                                         sr_ratio=sr_ratios[i],
                                         linear=linear) for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x, downstream=False):
        B = x.shape[0]
        features = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1 or downstream:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            features.append(x)

        return list(reversed(features))

    def forward(self, x):
        features = self.forward_features(x)
        x = self.head(features[-1].mean(dim=1))

        return x
@register_model
def pvt_v2_b2(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model

if __name__ == '__main__':
    model = pvt_v2_b2()
    print(model)
    inp = torch.randn(2, 3, 224, 224)
    out = model(inp)
    print(out.shape)