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

# from .utils import load_pretrained
from registry import register_model

# ms.set_context(mode=ms.PYNATIVE_MODE)

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
        'first_conv': '', 'classifier': '',
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
        # print('Conv2d_BN.x.1.shape:', x.shape)
        return x


'''
Conv2d_BN.x.1.shape: (2, 16, 112, 112)
Conv2d_BN.x.1.shape: (2, 32, 56, 56)
Conv2d_BN.x.1.shape: (2, 64, 28, 28)
Conv2d_BN.x.1.shape: (2, 128, 14, 14)
'''


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
        # print(x.shape)
        # l, bn = self.cells()
        # print('l.type:', type(l))  # Dense 对应上
        # print('Linear_BN.x.start.shape:', x.shape)  # 2,196,128   shape right
        x = self.linear(x)
        # print('Linear.x.shape:', x.shape)  # 2,196,256 shape right
        # x = ops.flatten(x)  # flatten出错
        x1, x2, x3 = x.shape
        new_x = ops.reshape(x, (x1 * x2, x3))
        # print('Linear_BN.x.2.shape:', x.shape)  # (2, 196, 256)
        x = self.bn1d(new_x).reshape(x.shape)
        # print('Linear_BN.x.3.shape:', x.shape)  # 2,196,256  right
        print('Linear_BN.x.4.shape:', x.shape)  # (392, 256)
        return x
        # return bn(x.flatten(0, 1)).reshape_as(x)  # 源码 2,196,256


'''
        self.insert_child_to_cell('c', nn.Dense(a,
                                                b,
                                                weight_init='Uniform',
                                                bias_init='Uniform',
                                                has_bias=False))
        bn = nn.BatchNorm1d(num_features=b,
                            gamma_init="ones",
                            beta_init="zeros",
                            momentum=0.9)  # 0.1

        self.insert_child_to_cell('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution ** 2
        FLOPS_COUNTER += a * b * output_points

    def fuse(self):
        # print('A')
        l, bn = self.cells()
        w = bn.gamma / (bn.moving_mean + bn.eps) ** 0.5
        w = l.gamma * w[:, None]
        b = bn.beta - bn.moving_mean * bn.gamma / \
            (bn.moving_variance + bn.eps) ** 0.5

        m = nn.Dense(w.size(1), w.size(0), bias_init='Uniform', weight_init='Uniform')  # check
        m.gamma.data.copy_(w)
        m.beta.data.copy_(b)
        return m
'''


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
        # print('BN_Linear.x.1.shape:', x.shape)
        return x


'''
        self.insert_child_to_cell('bn', nn.BatchNorm1d(num_features=b, momentum=0.9))
        l = nn.Dense(a,
                     b,
                     weight_init=init.TruncatedNormal(sigma=std),
                     bias_init='zeros',
                     has_bias=bias)

        # l = init.initializer(init.TruncatedNormal(sigma=std), nn.Dense(a, b, has_bias=bias), ms.float32)
        # 　tensor1 = initializer(TruncatedNormal(sigma=std), [1, 2, 3], ms.float32)
        # l: ms.Tensor = init.initializer(init.TruncatedNormal(sigma=std), nn.Dense(a, b, has_bias=bias), ms.float32)
        # trunc_normal_(l.weight, std=std)
        # TruncatedNormal
        # if bias:
        #    torch.nn.init.constant_(l.bias, 0)

        self.insert_child_to_cell('l', l)

        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    def fuse(self):
        bn, l = self.cells()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Dense(w.size(1), w.size(0), bias_init='Uniform', weight_init='Uniform')
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
'''


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
        #   return x + self.m(x) * ops.UniformReal(x.size(0), 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        #   return x + self.m(x) * paddle.to_tensor((np.random.rand(x.shape[0], 1, 1) > self.drop) / (1 - self.drop))
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
        #################
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        # self.q = Linear_BN(dim, h, resolution=resolution)
        # self.k = Linear_BN(dim, h, resolution=resolution)
        # self.v = Linear_BN(dim, h, resolution=resolution)
        #################
        self.proj = nn.SequentialCell(activation(), Linear_BN(self.dh, dim, resolution=resolution))
        # self.hswish = activation()
        # self.lbn = Linear_BN(self.dh, dim, resolution=resolution)
        # print('resolution:', resolution)
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

        # zeros = ops.Zeros()
        # print(len(attention_offsets))
        # print(num_heads)
        self.attention_biases = ms.Parameter(
            Tensor(np.zeros([num_heads, len(attention_offsets)], np.float32)))

        # self.ab = self.attention_biases[:, self.attention_bias_idxs]
        # print('type of idxs',type(idxs))
        # idxs = ms.Tensor(idxs)
        # attention_bias_idxs = ms.LongTensor(idxs).view(self.N, self.N)
        attention_bias_idxs = ms.Tensor(idxs, dtype=ms.int64).view(self.N, self.N)
        self.attention_bias_idxs = ms.Parameter(attention_bias_idxs, requires_grad=False)

        # ops.scalar_to_tensor(input_x, dtype=ms.int64)？？  cast_inputs(inputs, dst_type)??
        # paddle.to_tensor(idxs, dtype='int64')
        #  self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))
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
        print('Attention.construct.start')
        B, N, C = x.shape
        print('Attention.construct.x.shape:', x.shape)  # 2,196,128  right   # (2, 50176, 3)??

        atte = self.qkv(x).view(B, N, self.num_heads, -1)
        print('atte.shape:', atte.shape)
        atte_np = atte.asnumpy()
        [q, k, v, n] = np.split(atte_np,
                                [self.key_dim,
                                 self.key_dim + self.key_dim,
                                 self.key_dim + self.key_dim + self.d],
                                axis=3)
        q = Tensor(q)
        k = Tensor(k)
        v = Tensor(v)
        '''
        qkv = self.qkv(x)
        print('self.num_heads:', self.num_heads)  # 4
        # q, k, v = .split([self.key_dim, self.key_dim, self.d], axis=3)
        fir, v = qkv.view(B, N, self.num_heads, -1).split(3, 2)
        print('fir.shape:', fir.shape)
        print('v.shape:', v.shape)
        q, k = fir.split(3, 2)
        # print('q.type', type(q))
        print('q1.shape:', q.shape)
        print('k1.shape:', k.shape)
        print('v1.shape:', v.shape)
        '''
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # print('q1.shape:', q.shape)  # q1.shape: torch.Size([2, 4, 196, 16])   right
        # print('k1.shape:', k.shape)  # k1.shape: torch.Size([2, 4, 196, 16])   right
        # print('v1.shape:', v.shape)  # v1.shape: torch.Size([2, 4, 196, 32])   right
        # print('ops.transpose(k, (-4, -3, -1, -2)', ops.transpose(k, (-4, -3, -1, -2)).shape)  # (2, 4, 16, 196) right
        # print('q1.type:', type(q))
        # 　print('ops.transpose().type', type(ops.transpose(k, (-4, -3, -1, -2))))
        attn = (
                (ops.matmul(q, ops.transpose(k, (-4, -3, -1, -2)))) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        # paddle.index_select(self.attention_biases, self.attention_bias_idxs, axis=1).reshape((self.num_heads, self.N, self.N))
        # self.N = len(points)

        # print('attn.type:', type(attn))  # Tensor
        # print('attn.shape:', attn.shape)  # (2, 4, 196, 196)right

        attn = self.softmax(attn)
        # attn = attn.softmax(dim=-1)
        x = ops.transpose((ops.matmul(attn, v)), (0, 2, 1, 3))

        x = x.reshape(B, N, self.dh)

        x = self.proj(x)
        # x = self.hswish(x)
        # x = self.lbn(x)
        print('Attention.construct.end.x.shape:', x.shape)  # (2, 196, 128) right
        return x


'''
    def train(self,
              mode: bool = True) -> None:
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
'''


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
        # self.k = Linear_BN(in_dim, h, resolution=resolution)
        # self.v = Linear_BN(in_dim, h, resolution=resolution)

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
        # self.attention_biases = Parameter(
        #    ms.zeros(num_heads, len(attention_offsets)))
        # self.register_buffer('attention_bias_idxs',
        #                     ms.LongTensor(idxs).view(self.N, self.N))
        self.attention_biases = Parameter(
            Tensor(np.zeros([num_heads, len(attention_offsets)], np.float32)))
        # print('N:', N)    # 196 right 49
        # print('N_:', N_)  # 49 right 16
        # print('idxs.type:', type(idxs))
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
        print('AttentionSubsample.x.shape:', x.shape)  # (2, 196, 128)  right
        print('AttentionSubsample.self.num_heads:', self.num_heads)  # 8  right
        print('AttentionSubsample.self.key_dim:', self.key_dim)  # 16
        print('AttentionSubsample.self.d:', self.d)  # 64
        B, N, C = x.shape

        # k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.d], dim=3)源码
        print('AttentionSubsample.self.kv(x).view(B, N, self.num_heads, -1):',
              (self.kv(x).view(B, N, self.num_heads, -1)).shape)
        # k, v= self.kv(x).view(B, N, self.num_heads, -1).split(axis=3, output_num=2)  # 问题??
        # q, k = fir.split(3, 2)
        # k, g = fir.split(3, 2)
        # print('k0.shape:', k.shape)  # 196,8,80
        atte = self.kv(x).view(B, N, self.num_heads, -1)
        print('atte.shape:', atte.shape)  # 2,196,8,80
        atte_np = atte.asnumpy()

        [k, v, n] = np.split(atte_np, [self.key_dim, self.key_dim + self.d], axis=3)
        k = Tensor(k)
        v = Tensor(v)
        # k, v = np.split((self.kv(x).view(B, N, self.num_heads, -1)).asnumpy, [self.key_dim, self.d], axis=3)
        # np.split(array, [1, 3, 5], axis=0) 与 torch.split(tensor, [1, 2, 2, 1], dim=0) 效果相同
        # k = np.split(k, [self.key_dim, self.d], axis=3)
        # v = np.split(v, [self.key_dim, self.d], axis=3)
        # k = self.k(x).view(B, N, self.num_heads, -1)
        # v = self.v(x).view(B, N, self.num_heads, -1)
        # k = ops.tuple_to_array(k)
        # v = ops.tuple_to_array(v)
        print('k1.shape:', k.shape)  # 源码[2, 196, 8, 16]   right (2, 196, 8, 40)??
        print('v1.shape:', k.shape)  # 源码[2, 196, 8, 16]   right (2, 196, 8, 40)??
        # k = k.permute(0, 2, 1, 3)  # BHNC
        # v = v.permute(0, 2, 1, 3)  # BHNC
        v = ops.transpose(v, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        print('k2.shape:', k.shape)  # (2, 8, 196, 16)  right
        print('v2.shape:', k.shape)  # (2, 8, 196, 16)  right
        # q = self.q(x).view(B, self.resolution_2, self.num_heads,
        #                    self.key_dim).permute(0, 2, 1, 3)

        q = self.q(x).view(B, self.resolution_2, self.num_heads, self.key_dim)
        q = ops.transpose(q, (0, 2, 1, 3))

        print('q.shape:', q.shape)  # q.shape: torch.Size([2, 8, 49, 16])  right
        print('ops.transpose(k, (-4, -3, -1, -2).shape:',
              ops.transpose(k, (-4, -3, -1, -2)).shape)  # torch.Size([2, 8, 16, 196])
        # print('self.attention_bias_idxs:', self.attention_bias_idxs)
        print('B:', (self.attention_biases[:, self.attention_bias_idxs]
                     if self.training else self.ab).shape)  # 源码 8,49,196  8,196,196 问题

        # temp = ops.matmul(q, ops.transpose(k, (-4, -3, -1, -2)))

        attn = (
                ops.matmul(q, ops.transpose(k, (-4, -3, -1, -2))) * self.scale +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        attn = self.softmax(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        # x = ops.transpose((attn @ v), (0, 2, 1, 3))
        x = ops.transpose((ops.matmul(attn, v)), (0, 2, 1, 3))
        x = x.reshape(B, -1, self.dh)
        x = self.proj(x)
        print('AttentionSubsample.construct.end.x.shape:', x.shape)
        return x


'''
    def train(self,
              mode: bool = True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
'''


class LeViT(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 # in_channels: int = 3,
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

        # print('hybrid_backbone', hybrid_backbone)
        global FLOPS_COUNTER
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        self.blocks = []
        down_ops.append([''])
        # print('img_size // patch_size:', img_size, patch_size)
        resolution = img_size // patch_size
        # print('resolution.type:', type(resolution))
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
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
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

    def construct(self, x: Tensor) -> Tensor:
        print('x0.shape:', x.shape)  # 源码 2,3,224,224
        x = self.patch_embed(x)  # 问题
        # print('x1.shape:', x.shape)  # 源码 2,128,14,14 right
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        # print('x1.5.shape:', x.shape)  # 源码 2,128,196 right
        # x = x.flatten(2)  # 改flatten flatten有问题,无法在特定维度进行展开
        x = ops.transpose(x, (0, 2, 1))
        print('x2.shape:', x.shape)  # 源码 2,196,128 right

        x = self.blocks(x)  # 问题?
        print('x3.shape:', x.shape)
        x = x.mean(1)
        print('x4.shape:', x.shape)

        if self.distillation:
            x = self.head(x), self.head_dist(x)  # 问题？？
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        # 上面问题？？

        print('x_end.shape:', x.shape)
        return x


'''
    # @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}
'''

'''
if __name__ == '__main__':
    inputs = ms.Tensor(np.ones([4, 3, 224, 224]), ms.float32)  # 4,3,244,244
    print('inputs.shape:', inputs.shape)
    levit = LeViT(embed_dim=[128, 256, 384],
                  num_heads=[4, 6, 8],
                  key_dim=[16, 16, 16],
                  depth=[2, 3, 4],
                  down_ops=[['Subsample', 16, 128 // 16, 4, 2, 2],
                            ['Subsample', 16, 256 // 16, 4, 2, 2], ],
                  hybrid_backbone=b16(128)
                  )
    outputs = levit(inputs)
    print(outputs)
    print('outputs.shape:', outputs.shape)
'''


@register_model
def LeViT_128S(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_128S']
    model = LeViT(embed_dim=[128, 256, 384],
                  num_heads=[4, 6, 8],
                  key_dim=[16, 16, 16],
                  depth=[2, 3, 4],
                  down_ops=[
                      ['Subsample', 16, 128 // 16, 4, 2, 2],
                      ['Subsample', 16, 256 // 16, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(128),
                  **kwargs)

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_128']
    model = LeViT(embed_dim=[128, 256, 384],
                  num_heads=[4, 8, 12],
                  key_dim=[16, 16, 16],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 16, 128 // 16, 4, 2, 2],
                      ['Subsample', 16, 256 // 16, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(128),
                  **kwargs)

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_192']
    model = LeViT(embed_dim=[192, 288, 384],
                  num_heads=[3, 5, 6],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 192 // 32, 4, 2, 2],
                      ['Subsample', 32, 288 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(192),
                  **kwargs)

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_256(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_256']
    model = LeViT(embed_dim=[256, 384, 512],
                  num_heads=[4, 6, 8],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 256 // 32, 4, 2, 2],
                      ['Subsample', 32, 384 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(256),
                  **kwargs)

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    default_cfg = default_cfgs['LeViT_384']
    model = LeViT(embed_dim=[384, 512, 768],
                  num_heads=[6, 9, 12],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 384 // 32, 4, 2, 2],
                      ['Subsample', 32, 512 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(384),
                  **kwargs)

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


if __name__ == '__main__':
    import numpy as np
    import mindspore
    from mindspore import Tensor
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")  # context.GRAPH_MODE
    # context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    model = LeViT_128S()
    print(model)
    dummy_input = Tensor(np.random.rand(4, 3, 224, 224), dtype=mindspore.float32)
    y = model(dummy_input)
    print(y.shape)
