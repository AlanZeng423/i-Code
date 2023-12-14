import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    # 前向传播函数，通常用于梯度检查点
    def forward(ctx, run_function, length, *args):
        # 保存传递给 forward 函数的前向传播函数
        ctx.run_function = run_function
        # 将前 length 个输入张量保存到上下文中
        ctx.input_tensors = list(args[:length])
        # 将剩余的参数保存到上下文中
        ctx.input_params = list(args[length:])

        # 在 torch.no_grad() 上下文中执行前向传播函数，不记录梯度信息
        with torch.no_grad():
            # 调用保存的前向传播函数，计算输出张量
            output_tensors = ctx.run_function(*ctx.input_tensors)

        # 返回前向传播函数的输出张量作为这个自定义前向传播函数的输出
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # 将输入张量从计算图中分离，并设置 requires_grad 为 True，以便计算梯度
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

        # 在 torch.enable_grad() 上下文中执行梯度计算的过程
        with torch.enable_grad():
            # 修复一个 bug，其中 run_function 中的第一个操作会原地修改 Tensor 存储，而对于 detach() 的 Tensors，这是不允许的
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # 调用保存的前向传播函数，计算输出张量
            output_tensors = ctx.run_function(*shallow_copies)

        # 使用 torch.autograd.grad 计算输入张量和参数相对于输出梯度的梯度
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )

        # 删除保存在上下文中的临时变量，以释放内存
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors

        # 返回一个元组，其中包含两个 None，对应于未使用的参数和 run_function 的梯度，以及计算得到的输入梯度
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    # 如果 flag 为 True，使用梯度检查点
    if flag:
        # 将输入参数和参数转换为元组，并调用 CheckpointFunction.apply 函数
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    # 如果 flag 为 False，直接调用原始函数
    else:
        return func(*inputs)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: if True, directly repeat the input along the specified dimension.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # 如果不是直接重复，生成正弦余弦的时间步嵌入
    if not repeat_only:
        half = dim // 2
        # 计算频率
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        # 计算嵌入的角度
        args = timesteps[:, None].float() * freqs[None]
        # 使用余弦和正弦函数生成嵌入
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果维度是奇数，添加全零列
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        # 直接重复输入，维度为 dim
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 遍历模块的参数，将其全部置零
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    # 遍历模块的参数，将其乘以给定的尺度
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    # 在除了批次维度以外的所有维度上取平均
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # 使用 GroupNorm32 创建标准的规范化层
    return GroupNorm32(32, channels)

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        # 使用 Sigmoid Linear Unit（SiLU）激活函数
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        # 调用父类的 forward 方法
        return super().forward(x)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    # 根据维度创建对应维度的卷积模块
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    # 创建线性模块
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    # 根据维度创建对应维度的平均池化模块
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class HybridConditioner(nn.Module):
    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        # 根据配置文件实例化两个条件器
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        # 调用两个条件器的 forward 方法
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}

def noise_like(x, repeat=False):
    # 生成与输入形状相同的随机噪声
    noise = torch.randn_like(x)
    if repeat:
        bs = x.shape[0]
        # 如果需要，将噪声在批次维度上进行重复
        noise = noise[0:1].repeat(bs, *((1,) * (len(x.shape) - 1)))
    return noise
