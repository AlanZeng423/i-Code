import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import itertools

########
# unit #
########

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def str2value(v):
    v = v.strip()  # 使用strip()方法去除字符串v的前后空白字符
    try:
        return int(v)
    except:                     #异常处理
        pass
    try:
        return float(v)
    except:
        pass
    if v in ('True', 'true'):   # 检查字符串v是否为“True”或“true”
        return True
    elif v in ('False', 'false'):
        return False
    else:
        return v

@singleton
class get_unit(object):
    def __init__(self):
        self.unit = {}
        self.register('none', None)

        # general convolution
        # 注册了一些常见的神经网络层
        self.register('conv'  , nn.Conv2d)
        self.register('bn'    , nn.BatchNorm2d)
        self.register('relu'  , nn.ReLU)
        self.register('relu6' , nn.ReLU6)
        self.register('lrelu' , nn.LeakyReLU)
        self.register('dropout'  , nn.Dropout)
        self.register('dropout2d', nn.Dropout2d)
        self.register('sine',     Sine)
        self.register('relusine', ReLUSine)

    def register(self, 
                 name, 
                 unitf,):

        self.unit[name] = unitf

    def __call__(self, name):
        if name is None:
            return None
        i = name.find('(')            # 查找name字符串中第一个左括号(的位置
        i = len(name) if i==-1 else i # 如果没找到左括号（即i为-1），则将name的长度赋值给i，否则保持i不变
        t = name[:i]                  # 取name从开始到i位置的子字符串
        f = self.unit[t]              # 取出
        args = name[i:].strip('()')   # 取name从位置i到最后的子字符串，并去除两侧可能的括号
        if len(args) == 0:
            args = {}
            return f
        else:
            args = args.split('=')   # 使用等号=分割字符串，得到一个列表
            # 使用逗号,进一步分割字符串，并对每个元素进行处理，得到一个嵌套列表
            args = [[','.join(i.split(',')[:-1]), i.split(',')[-1]] for i in args]
            # 使用itertools库中的chain和from_iterable函数将嵌套列表展平为一个列表
            args = list(itertools.chain.from_iterable(args))
            # 去除每个元素两侧的空白字符，并过滤掉长度为0的元素
            args = [i.strip() for i in args if len(i)>0]
            kwargs = {}
            for k, v in zip(args[::2], args[1::2]): # args[1::2]：每隔一个元素取一个
                if v[0]=='(' and v[-1]==')':
                    # 首先去除字符串两端的括号，然后使用逗号,分割字符串，将每个分割得到的子字符串转换为适当的类型（通过调用str2value函数）
                    # 最后将这些值组成一个元组，并赋值给字典kwargs的相应键
                    kwargs[k] = tuple([str2value(i) for i in v.strip('()').split(',')])
                elif v[0]=='[' and v[-1]==']':
                    kwargs[k] = [str2value(i) for i in v.strip('[]').split(',')]
                else:
                    kwargs[k] = str2value(v)
            # 使用functools库中的partial函数创建一个新的函数对象
            # 该对象带有之前解析的参数和值（通过**kwargs的方式传递给新函数）
            return functools.partial(f, **kwargs)

def register(name):
    def wrapper(class_):
        get_unit().register(name, class_)
        return class_
    return wrapper

# 创建并调用不同频率和增益的正弦波
class Sine(object):
    def __init__(self, freq, gain=1):
        self.freq = freq    # 频率
        self.gain = gain    # 增益
        self.repr = 'sine(freq={}, gain={})'.format(freq, gain)

    def __call__(self, x, gain=1):
        act_gain = self.gain * gain
        return torch.sin(self.freq * x) * act_gain

    def __repr__(self,):
        return self.repr

# 简单的神经网络模块，它结合了正弦函数和 ReLU (Rectified Linear Unit) 激活函数
class ReLUSine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        a = torch.sin(30 * input)
        b = nn.ReLU(inplace=False)(input)
        return a+b

@register('lrelu_agc')
# class lrelu_agc(nn.Module):
class lrelu_agc(object):
    """
    The lrelu layer with alpha, gain and clamp
    """
    def __init__(self, alpha=0.1, gain=1, clamp=None):
        # super().__init__()
        self.alpha = alpha # Leaky ReLU的负斜率，当输入为负数时，输出的值将是输入值的alpha倍。默认0.1.
        if gain == 'sqrt_2':
            self.gain = np.sqrt(2)
        else:
            self.gain = gain
        # 可选的参数，用于在Leaky ReLU激活函数后对输出进行clamp操作，即限制输出的最小值和最大值。默认值为None
        self.clamp = clamp  
        self.repr = 'lrelu_agc(alpha={}, gain={}, clamp={})'.format(
            alpha, gain, clamp)

    # def forward(self, x, gain=1):
    def __call__(self, x, gain=1):
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True) # Leaky ReLU激活函数
        act_gain = self.gain * gain    
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain    # 应用增益
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)   # clamp操作
        return x

    def __repr__(self,):
        return self.repr

####################
# spatial encoding #
####################

@register('se')
# 提供了一个空间嵌入和空间编码的功能
class SpatialEncoding(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 sigma = 6,
                 cat_input=True,
                 require_grad=False,):
    # in_dim: 输入的维度。out_dim: 输出的维度。sigma: 用于生成高斯核的sigma值，默认为6。
    # cat_input: 是否将输入与嵌入相连接，默认为True。require_grad: 是否需要计算梯度，默认为False。

        super().__init__()
        assert out_dim % (2*in_dim) == 0, "dimension must be dividable"  # 确保输出维度是输入维度的两倍的倍数。

        n = out_dim // 2 // in_dim   # //:整除
        # 生成一个从0到sigma的等差数列，共有n个数。然后，对每个数取2的幂，得到一个数列
        m = 2**np.linspace(0, sigma, n) 
        # 将原始的m数组与(in_dim-1)个与m形状相同但所有元素为0的数组堆叠起来。
        # 堆叠的方向是最后一个轴（由axis=-1指定），所以结果仍然是一个二维数组
        m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
        # 首先使用列表推导式对m进行循环滚动。对于每个i从0到(in_dim-1)，都沿着最后一个轴滚动i个位置。
        # 然后，将这些滚动后的数组沿着第一个轴（即axis=0）连接起来。结果是一个形状改变的二维数组
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        # 将numpy数组m转换为PyTorch的张量，并存储在类的属性self.emb中。这个张量表示了空间编码的嵌入
        self.emb = torch.FloatTensor(m)
        # 如果require_grad为True，则将self.emb设置为一个可训练的参数
        # 在训练神经网络时，该嵌入将被更新以最小化损失函数
        if require_grad: 
            self.emb = nn.Parameter(self.emb, requires_grad=True)    
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x, format='[n x c]'):
        # x: 输入数据  format: 输入数据的格式
        """
        Args:
            x: [n x m1],
                m1 usually is 2
        Outputs:
            y: [n x m2]         
                m2 dimention number
        """
        # 如果 format 是 '[bs x c x 2D]'，则首先对输入数据进行维度调整，然后将其重塑为 [-1 x c]
        # 其中bs代表批量大小（batch size），c代表通道数（channel count），2D可能代表某种二维形状或尺寸
        if format == '[bs x c x 2D]':
            xshape = x.shape   # 获取输入张量x的形状
            '''
            对张量x进行维度重排。
            permute(0, 2, 3, 1)函数用于改变张量的轴(axis),这里是将张量的维度从原来的顺序重排为0, 2, 3, 1的顺序
            '''
            x = x.permute(0, 2, 3, 1).contiguous()
            '''
            将重排后的张量x重塑为新的形状。view函数用于改变张量的大小而不改变其内容。
            -1表示该维度的大小是自动计算的,以保持元素总数不变。
            x.size(-1)获取张量最后一个维度的大小
            '''
            x = x.view(-1, x.size(-1))
        elif format == '[n x c]':
            pass
        else:
            raise ValueError

        if not self.require_grad: # 检查类的require_grad属性是否为False。如果为False，表示不需要计算该模型的梯度
            self.emb = self.emb.to(x.device)  # 将嵌入向量（self.emb）移动到与输入数据相同的设备上(.to(device))
        y = torch.mm(x, self.emb.T) # 执行矩阵乘法,它将输入数据x与嵌入向量的转置（self.emb.T）相乘
        if self.cat_input:
            # 如果cat_input为True，这行代码将输入数据、y的正弦值和余弦值连接在一起，结果存储在变量z中
            z = torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            z = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)  # 将y的正弦值和余弦值连接在一起

        if format == '[bs x c x 2D]':
            z = z.view(xshape[0], xshape[2], xshape[3], -1)  # 如果format是[bs x c x 2D]，重塑变量z的形状
            z = z.permute(0, 3, 1, 2).contiguous()  # 对重塑后的张量z进行维度重排，并确保它是连续的
        return z

    def extra_repr(self):
        outstr = 'SpatialEncoding (in={}, out={}, sigma={}, cat_input={}, require_grad={})'.format(
            self.in_dim, self.out_dim, self.sigma, self.cat_input, self.require_grad)
        return outstr

@register('rffe')
class RFFEncoding(SpatialEncoding):
    """
    Random Fourier Features
    """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 sigma = 6,
                 cat_input=True,
                 require_grad=False,):
        '''
        * `in_dim`: 输入的维度。  
        * `out_dim`: 输出的维度。  
        * `sigma`: 默认值为6,可能用于高斯分布的标准差。  
        * `cat_input`: 默认为True,可能表示是否将输入与某些特征连接。  
        * `require_grad`: 默认为False,表示是否需要计算梯度。
        '''

        super().__init__(in_dim, out_dim, sigma, cat_input, require_grad)
        n = out_dim // 2
        # 使用NumPy库生成一个大小为(n, in_dim)的矩阵，其元素服从均值为0、标准差为sigma的正态分布
        m = np.random.normal(0, sigma, size=(n, in_dim))
        # 将NumPy矩阵m转换为PyTorch的浮点张量
        self.emb = torch.FloatTensor(m)
        # 如果require_grad为True，这行代码将类的属性self.emb转换为一个PyTorch参数，这意味着在计算梯度时，它将被考虑在内
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)    

    def extra_repr(self):
        outstr = 'RFFEncoding (in={}, out={}, sigma={}, cat_input={}, require_grad={})'.format(
            self.in_dim, self.out_dim, self.sigma, self.cat_input, self.require_grad)
        return outstr

##########
# helper #
##########

'''
这个函数 freeze 用于将神经网络模型的各个层设置为不进行训练
也就是说，它会锁定模型的参数，使得在后续的训练过程中，这些参数不会被更新
'''
def freeze(net):
    for m in net.modules():  # 遍历神经网络中的所有模块
    # 检查模块是否是批量归一化层（BatchNorm2d）或同步批量归一化层（SyncBatchNorm）
        if isinstance(m, (
                nn.BatchNorm2d, 
                nn.SyncBatchNorm,)):
            # inplace_abn not supported
            '''
            如果模块是批量归一化层或同步批量归一化层,将其设置为评估模式。
            在评估模式下,这些层的行为与训练模式不同。
            具体来说,训练时,这些层会使用mini-batch的统计数据来更新内部参数
            而在评估模式下,它们会使用在训练期间计算并存储的固定统计数据
            '''
            m.eval()
    for pi in net.parameters():  # 遍历神经网络的所有参数
        pi.requires_grad = False
    return net

def common_init(m):
    if isinstance(m, (
            nn.Conv2d, 
            nn.ConvTranspose2d,)):
        '''
        如果参数 m 是一个二维卷积层 (nn.Conv2d) 或一个二维卷积转置层 (nn.ConvTranspose2d)
        那么它会使用 nn.init.kaiming_normal_ 方法来初始化权重。
        mode='fan_out' 表示权重初始化的方式是根据 fan-out 模式
        而 nonlinearity='relu' 表示这个初始化方式特别适用于有 ReLU 非线性激活函数的网络。
        如果该层有偏置 (bias),则将其初始化为0。
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        '''
        如果参数 m 是一个二维批归一化层 (nn.BatchNorm2d) 或一个同步批归一化层 (nn.SyncBatchNorm)
        那么它会将权重和偏置都初始化为1和0
        这是批归一化的默认初始化方式。
        '''
    elif isinstance(m, (
            nn.BatchNorm2d, 
            nn.SyncBatchNorm,)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        pass

def init_module(module):
    """
    Args:
        module: [nn.module] list or nn.module
            a list of module to be initialized.
    """
    # 检查 module 是否是列表或元组。如果是，它将其转化为列表。否则，它创建一个包含 module 的列表
    if isinstance(module, (list, tuple)):
        module = list(module)
    else:
        module = [module]

    for mi in module:
        for mii in mi.modules():  # 子模块
            common_init(mii)

# 计算神经网络模型 net 中所有参数的总数量
def get_total_param(net):
    if getattr(net, 'parameters', None) is None:
        return 0
    return sum(p.numel() for p in net.parameters())  # numel() 方法计算 net 中所有参数的数量

# 计算神经网络模型 net 中所有参数的参数值之和
def get_total_param_sum(net):
    if getattr(net, 'parameters', None) is None:
        return 0 
    with torch.no_grad():  # 确保 parameters 属性不在计算梯度的上下文中
    '''
    #对于 net 中的每个参数 p,它首先将参数移到 CPU 上并断开与梯度的连接，然后将其转化为 NumPy 数组
    接下来，它计算这些参数在 CPU 上的和，并将这个和返回
    '''
        s = sum(p.cpu().detach().numpy().sum().item() for p in net.parameters())
    return s 
