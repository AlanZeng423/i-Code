from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .modules_conv import checkpoint

# 返回一个bool类型的函数值，val存在就true
def exists(val):
    return val is not None

# arr为输入数组，返回数组中所有唯一的元素（返回一个列表）
def uniq(arr):
    return{el: True for el in arr}.keys()

# val存在就返回val，不存在就调用d或者返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 返回了与输入张量 t 具有相同数据类型的最小负无穷大值。
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# 对输入张量初始化
def init_(tensor):
    dim = tensor.shape[-1] # 获取张量最后一个维度大小，用于线性层输入特征维度
    std = 1 / math.sqrt(dim) # 计算标准差
    tensor.uniform_(-std, std) # 范围内随机选择初始值，避免梯度爆炸或梯度消失
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out): # 将输入维度dim_in映射到输出维度dim_out的两倍
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x): # 模块向前传播
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate) # 增强神经网络的表达能力

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult) # 输入维度dim乘倍增因子mult，用于计算内部维度
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim) # 创建一个包含线性变换和 GELU（或GEGLU）激活函数的序列。这个序列会作为前馈神经网络的输入映射部分。

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x) # 输入x经过整个前馈神经网络的序列

# 综合来说，这个类实现了一个前馈神经网络，
# 包含一个映射部分（可以是线性变换 + GELU 或线性变换 + GEGLU）、
# 一个 dropout 层，以及最终的线性变换。
# 这种结构可以用于在神经网络中引入非线性映射和一些非线性激活函数，以提高模型的表达能力。


# 训练神经网络时，将模型参数初始化为0或者重新初始化为0，加快学习数据
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# 这个函数返回一个 Group Normalization 层，可以用于神经网络中的归一化操作
# 加速训练，提高模型性能
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# 自定义的带有线性注意力机制的pytorch模块
# 该模块通过 1x1 卷积层实现了查询、键和值的线性变换，然后使用线性注意力机制计算输出。
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        # 通过 1x1 卷积层实现查询（query）、键（key）和值（value）的线性变换
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 输出层，将变换后的值映射回原始维度
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 通过卷积层进行查询、键和值的线性变换
        qkv = self.to_qkv(x)

        # 重新排列张量的维度，以适应线性注意力的计算
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)

        # 使用 softmax 函数计算键的权重
        k = k.softmax(dim=-1)

        # 计算注意力上下文
        context = torch.einsum('bhdn,bhen->bhde', k, v)

        # 计算最终的输出
        out = torch.einsum('bhde,bhdn->bhen', context, q)

        # 重新排列张量的维度，以得到最终输出
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)

        # 通过 1x1 卷积层映射回原始维度
        return self.to_out(out)

# 自定义空间自注意力模块
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 归一化层，用于规范输入
        self.norm = Normalize(in_channels)

        # 查询（query）、键（key）、值（value）变换的卷积层
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # 通过卷积层映射输出
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x

        # 归一化输入
        h_ = self.norm(h_)

        # 查询、键、值变换
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 计算注意力权重
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 对值应用注意力权重
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)

        # 通过卷积层映射输出并加上原始输入
        h_ = self.proj_out(h_)
        return x + h_

# 交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        # 如果没有提供上下文维度，将其默认设置为查询维度
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        # 查询、键、值的线性变换
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 输出层，包括线性变换和可选的dropout
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        # 对查询、上下文进行线性变换
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 对查询、键、值进行形状变换以适应多头注意力
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # 计算注意力相似性
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 如果提供了掩码，则将掩码应用于相似性矩阵
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # 注意力权重计算
        attn = sim.softmax(dim=-1)

        # 使用注意力权重对值进行加权求和
        out = einsum('b i j, b j d -> b i d', attn, v)

        # 重塑输出形状并通过输出层
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()

        # 第一个注意力模块，是自注意力（self-attention）
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)

        # 前馈网络模块
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        # 第二个注意力模块，如果没有提供上下文，则是自注意力
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # 是否使用checkpointing
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # 如果启用checkpointing，则使用checkpoint函数以减少内存占用
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # 第一个自注意力模块，然后进行层归一化，并将结果与输入相加
        x = self.attn1(self.norm1(x)) + x

        # 第二个注意力模块，如果提供了上下文，则是交叉注意力，然后进行层归一化，并将结果与输入相加
        x = self.attn2(self.norm2(x), context=context) + x

        # 前馈网络，然后进行层归一化，并将结果与输入相加
        x = self.ff(self.norm3(x)) + x

        return x


# 这个模块首先通过卷积将输入映射到内部表示的维度，然后将其形状调整为b, t, d
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()

        # 输入通道数
        self.in_channels = in_channels

        # 归一化层
        self.norm = Normalize(in_channels)

        # 输入投影，将输入通道映射到内部表示的维度
        inner_dim = n_heads * d_head
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # Transformer块的列表，通过基本Transformer块的多次堆叠形成
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        # 输出投影，将内部表示映射回输入通道数
        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # 注意：如果没有提供上下文，交叉注意力默认为自注意力
        b, c, h, w = x.shape

        # 保存输入张量，用于最后将其与变换后的张量相加
        x_in = x

        # 归一化输入
        x = self.norm(x)

        # 投影输入，然后将其形状调整为 b, t, d
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # 通过多个Transformer块进行处理
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # 将形状调整回 b, c, h, w，并通过输出投影映射回输入通道数
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)

        # 将变换后的张量与输入相加，以便保留原始信息
        return x + x_in
