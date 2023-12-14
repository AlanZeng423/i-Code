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
    v = v.strip()  # ʹ��strip()����ȥ���ַ���v��ǰ��հ��ַ�
    try:
        return int(v)
    except:                     #�쳣����
        pass
    try:
        return float(v)
    except:
        pass
    if v in ('True', 'true'):   # ����ַ���v�Ƿ�Ϊ��True����true��
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
        # ע����һЩ�������������
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
        i = name.find('(')            # ����name�ַ����е�һ��������(��λ��
        i = len(name) if i==-1 else i # ���û�ҵ������ţ���iΪ-1������name�ĳ��ȸ�ֵ��i�����򱣳�i����
        t = name[:i]                  # ȡname�ӿ�ʼ��iλ�õ����ַ���
        f = self.unit[t]              # ȡ��
        args = name[i:].strip('()')   # ȡname��λ��i���������ַ�������ȥ��������ܵ�����
        if len(args) == 0:
            args = {}
            return f
        else:
            args = args.split('=')   # ʹ�õȺ�=�ָ��ַ������õ�һ���б�
            # ʹ�ö���,��һ���ָ��ַ���������ÿ��Ԫ�ؽ��д����õ�һ��Ƕ���б�
            args = [[','.join(i.split(',')[:-1]), i.split(',')[-1]] for i in args]
            # ʹ��itertools���е�chain��from_iterable������Ƕ���б�չƽΪһ���б�
            args = list(itertools.chain.from_iterable(args))
            # ȥ��ÿ��Ԫ������Ŀհ��ַ��������˵�����Ϊ0��Ԫ��
            args = [i.strip() for i in args if len(i)>0]
            kwargs = {}
            for k, v in zip(args[::2], args[1::2]): # args[1::2]��ÿ��һ��Ԫ��ȡһ��
                if v[0]=='(' and v[-1]==')':
                    # ����ȥ���ַ������˵����ţ�Ȼ��ʹ�ö���,�ָ��ַ�������ÿ���ָ�õ������ַ���ת��Ϊ�ʵ������ͣ�ͨ������str2value������
                    # �����Щֵ���һ��Ԫ�飬����ֵ���ֵ�kwargs����Ӧ��
                    kwargs[k] = tuple([str2value(i) for i in v.strip('()').split(',')])
                elif v[0]=='[' and v[-1]==']':
                    kwargs[k] = [str2value(i) for i in v.strip('[]').split(',')]
                else:
                    kwargs[k] = str2value(v)
            # ʹ��functools���е�partial��������һ���µĺ�������
            # �ö������֮ǰ�����Ĳ�����ֵ��ͨ��**kwargs�ķ�ʽ���ݸ��º�����
            return functools.partial(f, **kwargs)

def register(name):
    def wrapper(class_):
        get_unit().register(name, class_)
        return class_
    return wrapper

# ���������ò�ͬƵ�ʺ���������Ҳ�
class Sine(object):
    def __init__(self, freq, gain=1):
        self.freq = freq    # Ƶ��
        self.gain = gain    # ����
        self.repr = 'sine(freq={}, gain={})'.format(freq, gain)

    def __call__(self, x, gain=1):
        act_gain = self.gain * gain
        return torch.sin(self.freq * x) * act_gain

    def __repr__(self,):
        return self.repr

# �򵥵�������ģ�飬����������Һ����� ReLU (Rectified Linear Unit) �����
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
        self.alpha = alpha # Leaky ReLU�ĸ�б�ʣ�������Ϊ����ʱ�������ֵ��������ֵ��alpha����Ĭ��0.1.
        if gain == 'sqrt_2':
            self.gain = np.sqrt(2)
        else:
            self.gain = gain
        # ��ѡ�Ĳ�����������Leaky ReLU���������������clamp�������������������Сֵ�����ֵ��Ĭ��ֵΪNone
        self.clamp = clamp  
        self.repr = 'lrelu_agc(alpha={}, gain={}, clamp={})'.format(
            alpha, gain, clamp)

    # def forward(self, x, gain=1):
    def __call__(self, x, gain=1):
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True) # Leaky ReLU�����
        act_gain = self.gain * gain    
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain    # Ӧ������
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)   # clamp����
        return x

    def __repr__(self,):
        return self.repr

####################
# spatial encoding #
####################

@register('se')
# �ṩ��һ���ռ�Ƕ��Ϳռ����Ĺ���
class SpatialEncoding(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 sigma = 6,
                 cat_input=True,
                 require_grad=False,):
    # in_dim: �����ά�ȡ�out_dim: �����ά�ȡ�sigma: �������ɸ�˹�˵�sigmaֵ��Ĭ��Ϊ6��
    # cat_input: �Ƿ�������Ƕ�������ӣ�Ĭ��ΪTrue��require_grad: �Ƿ���Ҫ�����ݶȣ�Ĭ��ΪFalse��

        super().__init__()
        assert out_dim % (2*in_dim) == 0, "dimension must be dividable"  # ȷ�����ά��������ά�ȵ������ı�����

        n = out_dim // 2 // in_dim   # //:����
        # ����һ����0��sigma�ĵȲ����У�����n������Ȼ�󣬶�ÿ����ȡ2���ݣ��õ�һ������
        m = 2**np.linspace(0, sigma, n) 
        # ��ԭʼ��m������(in_dim-1)����m��״��ͬ������Ԫ��Ϊ0������ѵ�������
        # �ѵ��ķ��������һ���ᣨ��axis=-1ָ���������Խ����Ȼ��һ����ά����
        m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
        # ����ʹ���б��Ƶ�ʽ��m����ѭ������������ÿ��i��0��(in_dim-1)�����������һ�������i��λ�á�
        # Ȼ�󣬽���Щ��������������ŵ�һ���ᣨ��axis=0�����������������һ����״�ı�Ķ�ά����
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        # ��numpy����mת��ΪPyTorch�����������洢���������self.emb�С����������ʾ�˿ռ�����Ƕ��
        self.emb = torch.FloatTensor(m)
        # ���require_gradΪTrue����self.emb����Ϊһ����ѵ���Ĳ���
        # ��ѵ��������ʱ����Ƕ�뽫����������С����ʧ����
        if require_grad: 
            self.emb = nn.Parameter(self.emb, requires_grad=True)    
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x, format='[n x c]'):
        # x: ��������  format: �������ݵĸ�ʽ
        """
        Args:
            x: [n x m1],
                m1 usually is 2
        Outputs:
            y: [n x m2]         
                m2 dimention number
        """
        # ��� format �� '[bs x c x 2D]'�������ȶ��������ݽ���ά�ȵ�����Ȼ��������Ϊ [-1 x c]
        # ����bs����������С��batch size����c����ͨ������channel count����2D���ܴ���ĳ�ֶ�ά��״��ߴ�
        if format == '[bs x c x 2D]':
            xshape = x.shape   # ��ȡ��������x����״
            '''
            ������x����ά�����š�
            permute(0, 2, 3, 1)�������ڸı���������(axis),�����ǽ�������ά�ȴ�ԭ����˳������Ϊ0, 2, 3, 1��˳��
            '''
            x = x.permute(0, 2, 3, 1).contiguous()
            '''
            �����ź������x����Ϊ�µ���״��view�������ڸı������Ĵ�С�����ı������ݡ�
            -1��ʾ��ά�ȵĴ�С���Զ������,�Ա���Ԫ���������䡣
            x.size(-1)��ȡ�������һ��ά�ȵĴ�С
            '''
            x = x.view(-1, x.size(-1))
        elif format == '[n x c]':
            pass
        else:
            raise ValueError

        if not self.require_grad: # ������require_grad�����Ƿ�ΪFalse�����ΪFalse����ʾ����Ҫ�����ģ�͵��ݶ�
            self.emb = self.emb.to(x.device)  # ��Ƕ��������self.emb���ƶ���������������ͬ���豸��(.to(device))
        y = torch.mm(x, self.emb.T) # ִ�о���˷�,������������x��Ƕ��������ת�ã�self.emb.T�����
        if self.cat_input:
            # ���cat_inputΪTrue�����д��뽫�������ݡ�y������ֵ������ֵ������һ�𣬽���洢�ڱ���z��
            z = torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            z = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)  # ��y������ֵ������ֵ������һ��

        if format == '[bs x c x 2D]':
            z = z.view(xshape[0], xshape[2], xshape[3], -1)  # ���format��[bs x c x 2D]�����ܱ���z����״
            z = z.permute(0, 3, 1, 2).contiguous()  # �����ܺ������z����ά�����ţ���ȷ������������
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
        * `in_dim`: �����ά�ȡ�  
        * `out_dim`: �����ά�ȡ�  
        * `sigma`: Ĭ��ֵΪ6,�������ڸ�˹�ֲ��ı�׼�  
        * `cat_input`: Ĭ��ΪTrue,���ܱ�ʾ�Ƿ�������ĳЩ�������ӡ�  
        * `require_grad`: Ĭ��ΪFalse,��ʾ�Ƿ���Ҫ�����ݶȡ�
        '''

        super().__init__(in_dim, out_dim, sigma, cat_input, require_grad)
        n = out_dim // 2
        # ʹ��NumPy������һ����СΪ(n, in_dim)�ľ�����Ԫ�ط��Ӿ�ֵΪ0����׼��Ϊsigma����̬�ֲ�
        m = np.random.normal(0, sigma, size=(n, in_dim))
        # ��NumPy����mת��ΪPyTorch�ĸ�������
        self.emb = torch.FloatTensor(m)
        # ���require_gradΪTrue�����д��뽫�������self.embת��Ϊһ��PyTorch����������ζ���ڼ����ݶ�ʱ����������������
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
������� freeze ���ڽ�������ģ�͵ĸ���������Ϊ������ѵ��
Ҳ����˵����������ģ�͵Ĳ�����ʹ���ں�����ѵ�������У���Щ�������ᱻ����
'''
def freeze(net):
    for m in net.modules():  # �����������е�����ģ��
    # ���ģ���Ƿ���������һ���㣨BatchNorm2d����ͬ��������һ���㣨SyncBatchNorm��
        if isinstance(m, (
                nn.BatchNorm2d, 
                nn.SyncBatchNorm,)):
            # inplace_abn not supported
            '''
            ���ģ����������һ�����ͬ��������һ����,��������Ϊ����ģʽ��
            ������ģʽ��,��Щ�����Ϊ��ѵ��ģʽ��ͬ��
            ������˵,ѵ��ʱ,��Щ���ʹ��mini-batch��ͳ�������������ڲ�����
            ��������ģʽ��,���ǻ�ʹ����ѵ���ڼ���㲢�洢�Ĺ̶�ͳ������
            '''
            m.eval()
    for pi in net.parameters():  # ��������������в���
        pi.requires_grad = False
    return net

def common_init(m):
    if isinstance(m, (
            nn.Conv2d, 
            nn.ConvTranspose2d,)):
        '''
        ������� m ��һ����ά����� (nn.Conv2d) ��һ����ά���ת�ò� (nn.ConvTranspose2d)
        ��ô����ʹ�� nn.init.kaiming_normal_ ��������ʼ��Ȩ�ء�
        mode='fan_out' ��ʾȨ�س�ʼ���ķ�ʽ�Ǹ��� fan-out ģʽ
        �� nonlinearity='relu' ��ʾ�����ʼ����ʽ�ر��������� ReLU �����Լ���������硣
        ����ò���ƫ�� (bias),�����ʼ��Ϊ0��
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        '''
        ������� m ��һ����ά����һ���� (nn.BatchNorm2d) ��һ��ͬ������һ���� (nn.SyncBatchNorm)
        ��ô���ὫȨ�غ�ƫ�ö���ʼ��Ϊ1��0
        ��������һ����Ĭ�ϳ�ʼ����ʽ��
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
    # ��� module �Ƿ����б��Ԫ�顣����ǣ�������ת��Ϊ�б�����������һ������ module ���б�
    if isinstance(module, (list, tuple)):
        module = list(module)
    else:
        module = [module]

    for mi in module:
        for mii in mi.modules():  # ��ģ��
            common_init(mii)

# ����������ģ�� net �����в�����������
def get_total_param(net):
    if getattr(net, 'parameters', None) is None:
        return 0
    return sum(p.numel() for p in net.parameters())  # numel() �������� net �����в���������

# ����������ģ�� net �����в����Ĳ���ֵ֮��
def get_total_param_sum(net):
    if getattr(net, 'parameters', None) is None:
        return 0 
    with torch.no_grad():  # ȷ�� parameters ���Բ��ڼ����ݶȵ���������
    '''
    #���� net �е�ÿ������ p,�����Ƚ������Ƶ� CPU �ϲ��Ͽ����ݶȵ����ӣ�Ȼ����ת��Ϊ NumPy ����
    ����������������Щ������ CPU �ϵĺͣ���������ͷ���
    '''
        s = sum(p.cpu().detach().numpy().sum().item() for p in net.parameters())
    return s 
