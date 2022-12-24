
"""
MindSpore implementation of `LeViT`.
Refer to LeViT: LeViT Improving Vision Transformerswith Soft Convolutional Inductive Biases
"""
import itertools
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore.common.initializer import initializer, TruncatedNormal

from mindspore import Parameter, Tensor

from .utils import load_pretrained
from .registry import register_model

# ms.set_context(mode=ms.PYNATIVE_MODE)

# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

__all__ = [
    "LeViT",
    "LeViT_128S",
    "LeViT_128",
    "LeViT_192",
    "LeViT_256",
    "LeViT_384",
]


def _cfg(url='', **kwargs):  # need to check for
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        # 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'LeViT_128S': _cfg(url=''),
    'LeViT_128': _cfg(url=''),
    'LeViT_192': _cfg(url=''),
    'LeViT_256': _cfg(url=''),
    'LeViT_384': _cfg(url='')
}

FLOPS_COUNTER = 0


# 卷积、全连接和批归一化的各种组合以及残差层
class Conv2d_BN(nn.SequentialCell):
    def __init__(self,
                 a: int,
                 b: int,
                 ks: int = 1,
                 stride: int = 1,
                 pad: int = 0,  # pad=1
                 dilation: int = 1,
                 group: int = 1,
                 resolution: int = -10000) -> None:
        super().__init__()
        # 输入无误
        self.conv = nn.Conv2d(in_channels=a,
                              out_channels=b,
                              kernel_size=ks,
                              stride=stride,
                              padding=pad,  # padding
                              dilation=dilation,
                              group=group,
                              has_bias=False,
                              pad_mode="pad")

        self.bn = nn.BatchNorm2d(num_features=b,
                                 gamma_init="ones",
                                 beta_init="zeros",
                                 use_batch_statistics=True,
                                 momentum=0.9)  # 0.1

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1) ** 2
        FLOPS_COUNTER += a * b * output_points * (ks ** 2) // group

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class Linear_BN(nn.SequentialCell):
    def __init__(self,
                 a: int,
                 b: int,
                 resolution: int = -100000) -> None:
        super().__init__()

        self.linear = nn.Dense(a,
                               b,
                               weight_init='Uniform',
                               bias_init='Uniform',
                               has_bias=False)

        self.bn1d = nn.BatchNorm1d(num_features=b,
                                   gamma_init="ones",
                                   beta_init="zeros",
                                   # use_batch_statistics=True,
                                   momentum=0.9)  # 0.1

        global FLOPS_COUNTER
        output_points = resolution ** 2
        FLOPS_COUNTER += a * b * output_points

    def construct(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x1, x2, x3 = x.shape
        new_x = ops.reshape(x, (x1 * x2, x3))
        x = self.bn1d(new_x).reshape(x.shape)
        return x


class BN_Linear(nn.SequentialCell):
    def __init__(self,
                 a: int,
                 b: int,
                 bias: bool = True,
                 std: float = 0.02) -> None:
        super().__init__()

        self.bn1d = nn.BatchNorm1d(num_features=a,
                                   gamma_init="ones",
                                   beta_init="zeros",
                                   # use_batch_statistics=True,
                                   momentum=0.9)  # 0.1

        self.linear = nn.Dense(a,
                               b,
                               weight_init=init.TruncatedNormal(sigma=std),
                               bias_init='zeros',
                               has_bias=bias)

        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    def construct(self, x: Tensor) -> Tensor:
        x = self.bn1d(x)
        x = self.linear(x)
        return x


class Residual(nn.Cell):
    def __init__(self,
                 m: type = None,
                 drop: int = 0):
        super().__init__()
        self.m = m
        self.drop = drop

    def construct(self, x: Tensor) -> Tensor:
        if self.training and self.drop > 0:
            return x + self.m(x) * ms.Tensor.to_tensor(
                (np.random.rand(x.shape[0], 1, 1) > self.drop) / (1 - self.drop))  # 可能修改
        else:
            y = self.m(x)
            x = x + y
            return x


def b16(n, activation=nn.HSwish, resolution=224):  # CNN 分块嵌入（Patch Embedding）  ops.HSwish
    return nn.SequentialCell(
        Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Subsample(nn.Cell):  # 下采样
    def __init__(self,
                 stride: int,
                 resolution: int):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class Attention(nn.Cell):  # 注意力（Attention）
    def __init__(self,
                 dim: int,
                 key_dim: int,
                 num_heads: int = 8,
                 attn_ratio: int = 4,
                 activation: type = None,
                 resolution: int = 14) -> None:

        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = nn.SequentialCell(activation(), Linear_BN(self.dh, dim, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))  # 迭代两个不同大小的列表来获取新列表
        self.N = len(points)
        self.softmax = nn.Softmax(axis=-1)

        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = ms.Parameter(
            Tensor(np.zeros([num_heads, len(attention_offsets)], np.float32)))

        attention_bias_idxs = ms.Tensor(idxs, dtype=ms.int64).view(self.N, self.N)
        self.attention_bias_idxs = ms.Parameter(attention_bias_idxs, requires_grad=False)

        self.ab = self.attention_biases[:, self.attention_bias_idxs]

        global FLOPS_COUNTER  # 可能去除
        # queries * keys
        FLOPS_COUNTER += num_heads * (resolution ** 4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution ** 4)
        # attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution ** 4)

    def construct(self,
                  x: Tensor) -> Tensor:  # x (B,N,C)
        B, N, C = x.shape
        atte = self.qkv(x).view(B, N, self.num_heads, -1)
        # atte_np = atte.asnumpy()
        qkv =         ms.numpy.split(atte,
                                    [self.key_dim,
                                     self.d],
                                    axis=3)
        q=qkv[0]
        k=qkv[1]
        v=qkv[2]
        # q = Tensor(q)
        # k = Tensor(k)
        # v = Tensor(v)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = (
                (ops.matmul(q, ops.transpose(k, (-4, -3, -1, -2)))) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        attn = self.softmax(attn)

        x = ops.transpose((ops.matmul(attn, v)), (0, 2, 1, 3))

        x = x.reshape(B, N, self.dh)

        x = self.proj(x)

        return x


# AttentionSubsample：使用注意力机制的下采样层
class AttentionSubsample(nn.Cell):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 key_dim: int,
                 num_heads: int = 8,
                 attn_ratio: int = 2,
                 activation: type = None,
                 stride: int = 2,
                 resolution: int = 14,
                 resolution_: int = 7) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = nn.SequentialCell(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = nn.SequentialCell(activation(), Linear_BN(self.dh, out_dim, resolution=resolution_))
        #  self.proj = Linear_BN(self.dh, out_dim, resolution=resolution)
        self.softmax = nn.Softmax(axis=-1)
        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(range(resolution_), range(resolution_)))

        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = Parameter(
            Tensor(np.zeros([num_heads, len(attention_offsets)], np.float32)))

        attention_bias_idxs = (ms.Tensor(idxs, dtype=ms.int64)).view((N_, N))
        self.attention_bias_idxs = ms.Parameter(attention_bias_idxs, requires_grad=False)

        self.ab = self.attention_biases[:, self.attention_bias_idxs]

        global FLOPS_COUNTER
        # queries * keys
        FLOPS_COUNTER += num_heads * (resolution ** 2) * (resolution_ ** 2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution ** 2) * (resolution_ ** 2)
        # attention * v
        FLOPS_COUNTER += num_heads * (resolution ** 2) * (resolution_ ** 2) * self.d

    def construct(self,
                  x: Tensor) -> Tensor:

        B, N, C = x.shape
        atte = self.kv(x).view(B, N, self.num_heads, -1)
        # atte_np = atte.asnumpy()

        kv= ms.numpy.split(atte, [self.key_dim], axis=3)
        k=kv[0]
        v=kv[1]
        # k = Tensor(k)
        # v = Tensor(v)
        v = ops.transpose(v, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))

        q = self.q(x).view(B, self.resolution_2, self.num_heads, self.key_dim)
        q = ops.transpose(q, (0, 2, 1, 3))

        attn = (
                ops.matmul(q, ops.transpose(k, (-4, -3, -1, -2))) * self.scale +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        attn = self.softmax(attn)

        x = ops.transpose((ops.matmul(attn, v)), (0, 2, 1, 3))
        x = x.reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: list = [128, 256, 384],
                 key_dim: list = [16, 16, 16],
                 depth: list = [2, 3, 4],
                 num_heads: list = [4, 6, 8],
                 attn_ratio: list = [2, 2, 2],
                 mlp_ratio: list = [2, 2, 2],
                 hybrid_backbone: type = b16(128, activation=nn.HSwish),
                 down_ops: list = [['Subsample', 16, 128 // 16, 4, 2, 2], ['Subsample', 16, 256 // 16, 4, 2, 2]],
                 attention_activation: type = nn.HSwish,
                 mlp_activation: type = nn.HSwish,
                 distillation: bool = True,
                 drop_path: int = 0):
        super().__init__()

        global FLOPS_COUNTER
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        self.blocks = []

        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(nn.SequentialCell(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed,  # bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))

            if do[0] == 'Subsample':
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(nn.SequentialCell(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1],  # bn_weight_init=0,
                                resolution=resolution),
                        ), drop_path))
        self.blocks = nn.SequentialCell(*self.blocks)

        # Classifier head
        if num_classes > 0:
            self.head = BN_Linear(embed_dim[-1], num_classes)
            if distillation:
                self.head_dist = BN_Linear(embed_dim[-1], num_classes)

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0
        # 　self._initialize_weights()

        # def _initialize_weights(self) -> None:
        #     for _, cell in self.cells_and_names():
        #         if isinstance(cell, nn.Dense):
        #             cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.data.shape))
        #             if cell.bias is not None:
        #                 cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape))
        #         elif isinstance(cell, nn.LayerNorm):
        #             cell.gamma.set_data(init.initializer(init.Constant(1.0), cell.gamma.shape))
        #             cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape))

    def construct(self, x: Tensor) -> Tensor:

        x = self.patch_embed(x)  # 问题
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = ops.transpose(x, (0, 2, 1))
        x = self.blocks(x)  # 问题?
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)  # 问题？？
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


@register_model
def LeViT_128S(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_128S']
    model = LeViT(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=[128, 256, 384],
                  num_heads=[4, 6, 8],
                  key_dim=[16, 16, 16],
                  depth=[2, 3, 4],
                  down_ops=[
                      ['Subsample', 16, 128 // 16, 4, 2, 2],
                      ['Subsample', 16, 256 // 16, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(128),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_128']
    model = LeViT(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=[128, 256, 384],
                  num_heads=[4, 8, 12],
                  key_dim=[16, 16, 16],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 16, 128 // 16, 4, 2, 2],
                      ['Subsample', 16, 256 // 16, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(128),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_192']
    model = LeViT(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=[192, 288, 384],
                  num_heads=[3, 5, 6],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 192 // 32, 4, 2, 2],
                      ['Subsample', 32, 288 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(192),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_256(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_256']
    model = LeViT(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=[256, 384, 512],
                  num_heads=[4, 6, 8],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 256 // 32, 4, 2, 2],
                      ['Subsample', 32, 384 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(256),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_384']
    model = LeViT(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=[384, 512, 768],
                  num_heads=[6, 9, 12],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 384 // 32, 4, 2, 2],
                      ['Subsample', 32, 512 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(384),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


# if __name__ == '__main__':
#     import numpy as np
#     import mindspore
#     from mindspore import Tensor
#     from mindspore import context
#
#     context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
#
#     model = LeViT_128S()
#     print(model)
#     dummy_input = Tensor(np.random.rand(4, 3, 224, 224), dtype=mindspore.float32)
#     y = model(dummy_input)
#     print(y.shape)
