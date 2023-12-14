"""
https://github.com/lucidrains/make-a-video-pytorch
"""

import math
import functools
from operator import mul

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from .modules_conv import avg_pool_nd, zero_module, normalization, conv_nd

# helper functions

def exists(val): 
    return val is not None

def default(val, d):
    return val if exists(val) else d

def mul_reduce(tup):
    # 使用 functools.reduce 对元组中的元素进行逐元素乘法并减少结果
    return functools.reduce(mul, tup)

def divisible_by(numer, denom):
    # 检查一个数字是否可以被另一个数字整除，返回bool类型
    return (numer % denom) == 0

mlist = nn.ModuleList

# for time conditioning
# 生成输入序列的正弦位置嵌入，帮助模型更好地理解序列中的位置信息。
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        # 初始化 SinusoidalPosEmb 模块，接收维度 dim 和正弦函数参数 theta
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(self, x):
        # 正向传播逻辑
        dtype, device = x.dtype, x.device
        # 检查输入 x 的数据类型是否为浮点型
        assert dtype == torch.float, 'input to sinusoidal pos emb must be a float type'

        # 计算正弦位置嵌入的一半维度
        half_dim = self.dim // 2
        # 计算正弦位置嵌入的参数 emb
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        # 生成正弦位置嵌入，并与输入 x 进行相应的元素乘法
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        # 将正弦和余弦的位置嵌入连接在一起，并按最后一个维度连接
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


# 通道层归一化，用于在通道维度上对输入进行归一化，并通过可学习参数 g 进行缩放
class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        # 初始化通道层归一化模块，接收维度 dim
        super().__init__()
        # 定义可学习参数 g，形状为 (dim, 1, 1, 1)，用于缩放
        self.g = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        # 正向传播逻辑
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        # 计算输入 x 沿着通道维度的方差和均值
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        # 标准化输入 x
        x = (x - mean) * var.clamp(min=eps).rsqrt()
        dtype = self.g.dtype
        # 对标准化后的 x 进行缩放，并转换为与 g 相同的数据类型
        return x.to(dtype) * self.g


def shift_token(t):
    # 将输入张量 t 沿着第二个维度分块为两个张量 t 和 t_shift （第二维度通常是时间维度或特征维度）
    t, t_shift = t.chunk(2, dim=1)
    # 使用 F.pad 对 t_shift 进行零填充，将最后一列移动到第一列
    t_shift = F.pad(t_shift, (0, 0, 0, 0, 1, -1), value=0.)
    # 沿着第二个维度连接 t 和 t_shift，并返回结果张量
    return torch.cat((t, t_shift), dim=1)



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 初始化可学习参数 g
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 根据数据类型选择适当的 epsilon（eps）
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        # 计算输入张量 x 在第二个维度上的方差（variance）和均值（mean）
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        # Layer Normalization 的计算公式
        normalized_x = (x - mean) * var.clamp(min=eps).rsqrt() * self.g
        return normalized_x



# feedforward
class GEGLU(nn.Module):
    def forward(self, x):
        # 将输入张量 x 转换为 float 类型
        x = x.float()
        # 将 x 沿着第二个维度分割成两部分，得到 x 和 gate
        x, gate = x.chunk(2, dim=1)
        # 使用 GELU 激活函数对 gate 部分进行处理，并将其与 x 相乘
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        # 计算内部维度
        inner_dim = int(dim * mult * 2 / 3)

        # 输入投影部分，包括 3D 卷积和 GEGLU 激活函数
        self.proj_in = nn.Sequential(
            nn.Conv3d(dim, inner_dim * 2, 1, bias=False),
            GEGLU()
        )

        # 输出投影部分，包括通道层归一化和 3D 卷积
        self.proj_out = nn.Sequential(
            ChanLayerNorm(inner_dim),
            nn.Conv3d(inner_dim, dim, 1, bias=False)
        )

    def forward(self, x, enable_time=True):
        # 对输入进行投影
        x = self.proj_in(x)

        # 如果启用时间维度，对输入进行 shift_token 操作
        if enable_time:
            x = shift_token(x)

        # 对投影后的结果进行输出投影
        return self.proj_out(x)



# feedforwa
# best relative positional encoding
class ContinuousPositionBias(nn.Module):
    """来自 https://arxiv.org/abs/2111.09883"""

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims=1,
        layers=2,
        log_dist=True,
        cache_rel_pos=False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        # 创建神经网络层
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, *dimensions):
        device = self.device

        # 如果没有缓存相对位置或不缓存相对位置，则重新计算
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing='ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            # 如果使用对数距离，则进行对数变换
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent=False)

        rel_pos = self.rel_pos.to(self.dtype)

        # 通过神经网络层处理相对位置
        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')


# helper classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)

        # 用于计算查询、键、值的线性层
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # 初始化输出层的权重为零，以实现跳跃连接
        nn.init.zeros_(self.to_out.weight.data)

        # 位置嵌入
        self.pos_embeds = nn.Parameter(torch.randn([1, 30, dim]))
        self.frame_rate_embeds = nn.Parameter(torch.randn([1, 30, dim]))

    def forward(
        self,
        x,
        context=None,
        rel_pos_bias=None,
        framerate=None,
    ):
        # 添加位置嵌入，这里使用了预定义的位置和帧速率嵌入
        if framerate is not None:
            x = x + self.pos_embeds[:, :x.shape[1]].repeat(x.shape[0], 1, 1)
            x = x + self.frame_rate_embeds[:, framerate-1:framerate].repeat(x.shape[0], x.shape[1], 1)

        # 如果没有提供上下文，将上下文设为输入本身
        if context is None:
            context = x

        # 归一化输入和上下文
        x = self.norm(x)
        context = self.norm(context)

        # 计算查询、键、值
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)

        # 重排形状以适应多头自注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # 缩放查询
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 如果提供了相对位置偏置，则添加到相似度矩阵中
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # 计算注意力权重
        attn = sim.softmax(dim=-1)

        # 计算加权和
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 重排形状，将多头注意力结果连接在一起
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 通过线性层处理最终输出
        return self.to_out(out)


# main contribution - pseudo 3d conv

class PseudoConv3d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        kernel_size=3,
        *,
        temporal_kernel_size=None,
        **kwargs
    ):
        super().__init__()
        # 如果未提供dim_out，则默认与输入维度相同
        dim_out = default(dim_out, dim)
        # 如果未提供temporal_kernel_size，则默认为kernel_size
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        # 空间卷积层
        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)
        # 如果kernel_size大于1，则添加时间卷积层
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size=temporal_kernel_size, padding=temporal_kernel_size // 2) if kernel_size > 1 else None

        # 如果存在时间卷积层，初始化为单位矩阵
        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data)
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(
        self,
        x,
        enable_time=True
    ):
        b, c, *_, h, w = x.shape

        # 判断是否为视频数据（5D张量）
        is_video = x.ndim == 5
        enable_time &= is_video

        # 如果是视频，重排形状以适应卷积操作
        if is_video:
            x = rearrange(x, 'b c t h w -> (b t) c h w')

        # 进行空间卷积
        x = self.spatial_conv(x)

        # 如果是视频，将形状还原
        if is_video:
            x = rearrange(x, '(b t) c h w -> b c t h w', b=b)

        # 如果不启用时间卷积或者没有时间卷积层，则直接返回结果
        if not enable_time or not exists(self.temporal_conv):
            return x

        # 对输入进行形状重排以适应时间卷积
        x = rearrange(x, 'b c t h w -> (b h w) c t')

        # 进行时间卷积
        x = self.temporal_conv(x)

        # 将形状还原
        x = rearrange(x, '(b h w) c t -> b c t h w', h=h, w=w)

        return x


def frame_shift(x, shift_num=8):
    # 获取输入张量的帧数
    num_frame = x.shape[2]
    # 将输入张量在通道维度上分块成shift_num块
    x = list(x.chunk(shift_num, 1))

    # 循环遍历每一块
    for i in range(shift_num):
        if i > 0:
            # 对于非第一块，创建一个位移后的新块，并拼接在一起
            shifted = torch.cat([torch.zeros_like(x[i][:, :, :i]), x[i][:, :, :-i]], 2)
        else:
            # 对于第一块，直接使用原始块
            shifted = x[i]
        # 更新块列表
        x[i] = shifted

    # 将所有块在通道维度上拼接在一起，形成位移后的张量
    return torch.cat(x, 1)


class ResBlockFrameShift(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        # 定义输出层
        self.out_layers = nn.Sequential(
            normalization(self.channels),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
            ),
        )

        # 定义skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        num_frames = x.shape[2]
        # 将输入张量在通道维度上拆分成(b*t, c, h, w)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        # 对输入张量应用输出层
        h = self.out_layers(x)

        # 将输出张量在通道维度上拼接在一起，并进行帧位移操作
        h = rearrange(h, '(b t) c h w -> b c t h w', t=num_frames)
        h = frame_shift(h)
        # 将输出张量再次在通道维度上拆分成(b*t, c, h, w)
        h = rearrange(h, 'b c t h w -> (b t) c h w')

        # 计算skip connection，并将其与输出张量相加
        out = self.skip_connection(x) + h
        # 将最终的输出张量再次在通道维度上拆分成(b*t, c, h, w)
        out = rearrange(out, '(b t) c h w -> b c t h w', t=num_frames)
        return out


class ResBlockVideo(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # 定义输入层
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # 定义输出层
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # 定义skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        num_frames = x.shape[2]
        # 将输入张量在通道维度上拆分成(b*t, c, h, w)
        x = rearrange(x, 'b c t h w -> (b t) c h w ')

        # 复制输入张量作为初始值
        h = x
        # 对初始值应用输入层
        h = self.in_layers(h)
        # 对输入层的输出应用输出层
        h = self.out_layers(h)

        # 计算skip connection，并将其与输出张量相加
        out = self.skip_connection(x) + h
        # 将最终的输出张量再次在通道维度上拆分成(b*t, c, h, w)
        out = rearrange(out, '(b t) c h w -> b c t h w', t=num_frames)
        return out


class Downsample3D(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, stride=None, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 1
        # 根据use_conv的值选择是使用卷积操作还是平均池化操作
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            # 如果不使用卷积，则使用平均池化操作
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        # 检查输入张量的通道数是否与指定的channels一致
        assert x.shape[1] == self.channels
        # 应用选择的操作（卷积或平均池化）
        return self.op(x)


class SpatioTemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dim_head=64,
            heads=8,
            use_resnet=False,
            use_frame_shift=True,
            use_context_att=False,
            use_temp_att=True,
            use_context=False,
    ):
        super().__init__()
        self.use_resnet = use_resnet
        self.use_frame_shift = use_frame_shift
        self.use_context_att = use_context_att
        self.use_temp_att = use_temp_att

        # 如果使用 ResNet 结构
        if use_resnet:
            self.resblock = ResBlockVideo(dim, dropout=0, dims=2)
        # 如果使用帧位移（frame shift）
        if use_frame_shift:
            self.frameshiftblock = ResBlockFrameShift(dim, dropout=0, dims=2)
        # 如果使用上下文注意力（context attention）
        if use_context_att:
            self.downsample_x0 = Downsample3D(4, True, 2, out_channels=dim)
            self.temporal_attn_x0 = Attention(dim=dim, dim_head=dim_head, heads=heads)

        # 如果使用时间注意力（temporal attention）
        if use_temp_att:
            self.temporal_attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
            self.temporal_rel_pos_bias = ContinuousPositionBias(dim=dim // 2, heads=heads, num_dims=1)

            self.ff = FeedForward(dim=dim, mult=4)

    def forward(
            self,
            x,
            x_0=None,
            enable_time=True,
            framerate=4,
            is_video=False,
    ):

        x_ndim = x.ndim
        is_video = x_ndim == 5 or is_video
        enable_time &= is_video

        if enable_time:
            img_size = x.shape[-1]
            if self.use_temp_att:
                if x_ndim == 5:
                    b, c, *_, h, w = x.shape
                    x = rearrange(x, 'b c t h w -> (b h w) t c')
                time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1])

            if self.use_context_att and x_0 is not None:
                x_0_img_size = x_0.shape[-1]
                kernel_size = x_0_img_size // img_size
                # 使用平均池化对 x_0 进行下采样
                x_0 = F.avg_pool2d(x_0, [kernel_size, kernel_size], stride=None, padding=0, ceil_mode=False,
                                   count_include_pad=True, divisor_override=None)
                x_0 = self.downsample_x0(x_0).unsqueeze(2)
                if x_ndim == 5:
                    x_0 = rearrange(x_0, 'b c t h w -> (b h w) t c')
                # 应用时间注意力到 x 上
                x = self.temporal_attn_x0(x, context=x_0, rel_pos_bias=time_rel_pos_bias, framerate=framerate) + x

            if self.use_temp_att:
                # 应用时间注意力到 x 上
                x = self.temporal_attn(x, rel_pos_bias=time_rel_pos_bias, framerate=framerate) + x
                if x_ndim == 5:
                    x = rearrange(x, '(b h w) t c -> b c t h w', w=w, h=h)
                # 应用 FeedForward 模块
                x = self.ff(x, enable_time=enable_time) + x

            if self.use_frame_shift:
                # 应用帧位移
                x = self.frameshiftblock(x)

            if self.use_resnet:
                # 应用 ResNet 结构
                x = self.resblock(x)

        return x
