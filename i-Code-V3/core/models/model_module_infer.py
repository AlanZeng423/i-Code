# 处理文件和目录的模块
import os

# torch 和 torch.nn 用于构建和训练神经网络。
import torch
import torch.nn as nn # nn表示神经网络
import torch.nn.functional as F # F表示函数, 提供了一系列用于构建神经网络的函数
'''
`torch.nn` 提供面向对象的网络层类，包含可学习参数，而 `torch.nn.functional` 包含相同操作的无状态函数，需要手动提供参数。
'''
import torchvision.transforms as tvtrans
'''
torchvision.transforms 是 PyTorch 中 torchvision 库的一个模块，主要用于图像预处理和增强。
这个模块提供了许多常用的图像变换操作，使得它们可以轻松地应用于图像数据，特别是用于训练深度学习模型时。
'''

from einops import rearrange
'''
- einops(Einstein Operations)是一个强大的库，用于提供清晰、可读和简洁的方式来处理和转换多维数组，特别是在深度学习和数据分析中。
- rearrange 函数是 einops 的核心功能之一，它允许你以非常直观和灵活的方式重新排列多维数组（如 NumPy 数组、PyTorch 张量、TensorFlow 张量等）的维度。
这种操作在深度学习中非常常见，比如在处理图像、时间序列或其他复杂数据结构时。
- 使用 rearrange, 你可以通过一个简单的字符串来描述如何转换数组的形状。
例如，你可以轻松地交换维度、改变形状、增加或减少维度等。这比传统的方法（如使用 reshape、transpose 等）更为直观和灵活。
'''

import pytorch_lightning as pl
'''
PyTorch Lightning 是一个在 PyTorch 之上的开源库，旨在简化深度学习模型的开发和训练过程，同时保持 PyTorch 的灵活性和强大功能。
'''

from . import get_model
from ..cfg_helper import model_cfg_bank
from ..common.utils import regularize_image, regularize_video, remove_duplicate_word

import warnings
warnings.filterwarnings("ignore")


class model_module(pl.LightningModule): # model_module 类继承自 pl.LightningModule 类
    def __init__(self, data_dir='pretrained', pth=["CoDi_encoders.pth"], fp16=False): # 初始化函数
        super().__init__()
        
        cfgm = model_cfg_bank()('codi') # 首先从配置库中获取名为 'codi' 的模型配置
        net = get_model()(cfgm) # 并使用这个配置来实例化相应的模型
        
        if fp16: # 如果 fp16 为 True, 则将模型转换为半精度
            net = net.half()
        for path in pth: # 遍历 pth 中的每个路径
            net.load_state_dict(torch.load(os.path.join(data_dir, path), map_location='cpu'), strict=False) 
            # 从指定文件中加载模型的参数状态字典，并将这些参数状态应用到名为 net 的PyTorch模型中
        print('Load pretrained weight from {}'.format(pth))

        self.net = net
        
        from core.models.ddim.ddim_vd import DDIMSampler_VD
        self.sampler = DDIMSampler_VD(net)
        '''
        创建一个 DDIMSampler_VD 的实例，并将名为 net 的 PyTorch 模型作为参数传递给 DDIMSampler_VD 的构造函数,
        然后将这个 DDIMSampler_VD 实例赋值给当前对象的 sampler 属性
        '''

    def decode(self, z, xtype): # 定义 decode(解码器) 
        net = self.net
        z = z.cuda() # 将 z 移动到 GPU 上
        if xtype == 'image': # 如果 xtype 为 'image'
            # 1. 使用 autokl_decode 方法解码 z
            x = net.autokl_decode(z) 
            '''    [in ./codi.py] 
            @torch.no_grad()
            def autokl_decode(self, z):
                z = 1. / self.vision_scale_factor * z
                return self.autokl.decode(z)
            '''
            # 2.数据后处理：解码后的数据被标准化到0到1的范围内（torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)）
            # 这是因为模型通常输出的数据范围是 [-1, 1]，而图像数据通常需要在 [0, 1] 范围内。
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)

            # 3. 转换为PIL图像：使用 tvtrans.ToPILImage() 将张量转换为PIL图像格式，以便进一步处理或显示
            x = [tvtrans.ToPILImage()(xi) for xi in x]
            return x
        
        elif xtype == 'video': # 如果 xtype 为 'video'
            '''
            shape:
            - b:批次大小(Batch Size)
            - c:通道数(Channels)
            - f:时间帧数(Frames)
            - h:高度(Height)
            - w:宽度(Width)
            '''
            num_frames = z.shape[2] # 获取张量 z 的第三个维度的大小，即时间帧数
            
            z = rearrange(z, 'b c f h w -> (b f) c h w') 
            # 将原始的 z 张量从形状 (b, c, f, h, w) 重排列为 (b * f, c, h, w) 的形状
            # 重排列操作的目的是将时间帧 f 与批次维度 b 结合，以便后续的解码操作
            x = net.autokl_decode(z) # 解码
            x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
            # 将解码后的 x 张量从形状 (b * f, c, h, w) 重排列回原始形状 (b, f, c, h, w)，并使用之前提取的时间帧数 num_frames 进行指定。
            # 这个操作的目的是将时间帧维度重新分离回来，以获得每个批次中的时间序列数据
            
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            video_list = []
            for video in x:
                video_list.append([tvtrans.ToPILImage()(xi) for xi in video])
            return video_list

        elif xtype == 'text':
            prompt_temperature = 1.0 # 温度参数
            prompt_merge_same_adj_word = True # 合并相邻的重复单词
            x = net.optimus_decode(z, temperature=prompt_temperature)
            '''
            @torch.no_grad()
            def optimus_decode(self, z, temperature=1.0):
                z = 1.0 / self.text_scale_factor * z
                return self.optimus.decode(z, temperature)
            '''
            if prompt_merge_same_adj_word:
                xnew = []
                for xi in x:
                    xi_split = xi.split()
                    xinew = []
                    for idxi, wi in enumerate(xi_split):
                        if idxi!=0 and wi==xi_split[idxi-1]:
                            continue
                        xinew.append(wi)
                    xnew.append(remove_duplicate_word(' '.join(xinew)))
                x = xnew
            return x
        
        elif xtype == 'audio':
            x = net.audioldm_decode(z)
            '''
            @torch.no_grad()
            def audioldm_decode(self, z):
                if (torch.max(torch.abs(z)) > 1e2):
                    z = torch.clip(z, min=-10, max=10)
                z = 1.0 / self.audio_scale_factor * z
                return self.audioldm.decode(z)
            '''
            x = net.mel_spectrogram_to_waveform(x)
            ''' 将梅尔频谱图(Mel spectrogram)转换为波形信号(waveform)
            @torch.no_grad()
            def mel_spectrogram_to_waveform(self, mel):
                # Mel: [bs, 1, t-steps, fbins]
                if len(mel.size()) == 4:
                    mel = mel.squeeze(1)
                mel = mel.permute(0, 2, 1)
                waveform = self.audioldm.vocoder(mel)
                waveform = waveform.cpu().detach().numpy()
                return waveform
            '''
            return x

    def inference(self, xtype=[], condition=[], condition_types=[], n_samples=1, mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1}, image_size=256, ddim_steps=50, scale=7.5, num_frames=8):
        '''
        输入:
        - xtype: 一个字符串列表，指定所需的输出格式。支持的值包括 "image"、"video"、"text" 和 "audio"。
        - condition: 一个条件输入列表，对应于按相同顺序的每个 xtype。这些可以是图像、视频、音频剪辑或文本字符串。
        - condition_types: 一个字符串列表，指定每个条件输入的类型，对应于 condition 中的顺序。可能的值与 xtype 中的值匹配。
        - n_samples: 一个整数，指定为每个 xtype 生成样本的数量。
        - mix_weight: 一个字典，指定在生成过程中混合时分配给每个模态的权重。键是 "video"、"audio"、"text" 和 "image"，值是对应的权重。默认为所有模态的权重相等。
        - image_size: 一个整数，指定生成的图像的目标图像大小。
        - ddim_steps: 一个整数，指定用于生成的扩散步骤数。
        - scale: 一个浮点数，指定无条件引导缩放因子。默认为 1.0。
        - num_frames: 一个整数，指定生成的视频的帧数。仅与 xtype="video" 相关。
        '''
        net = self.net
        sampler = self.sampler
        ddim_eta = 0.0

        conditioning = []
        assert len(set(condition_types)) == len(condition_types), "we don't support condition with same modalities yet."
        assert len(condition) == len(condition_types)
        
        for i, condition_type in enumerate(condition_types): 
            # enumerate() 函数用于将一个可迭代的对象转换为一个枚举对象
            if condition_type == 'image':
                ctemp1 = regularize_image(condition[i]).cuda() # in core/common/utils.py, regularize_image() 函数用于将图像转换为张量
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1) 
                # 这行代码将 ctemp1 张量重复 n_samples 次, 这是为了生成 n_samples 个不同的条件
                # ctemp1[None] 或 ctemp1.unsqueeze(0)：这两种方式都是用来在 ctemp1 的最前面增加一个新的维度。
                # repeat():
                #   假设有一个张量 A，其形状是 [d1, d2, ..., dn]，你调用 A.repeat(x1, x2, ..., xn)。
                #   这里，xi 表示第 i 维上元素应该重复的次数。结果张量的每个维度的大小将是原始大小乘以重复次数。 
                #   例子:A = [[1, 2, 3],
                #            [4, 5, 6]]
                #   调用A.repeat(2, 3)结果为:
                #   [[1, 2, 3, 1, 2, 3, 1, 2, 3],
                #   [4, 5, 6, 4, 5, 6, 4, 5, 6],
                #   [1, 2, 3, 1, 2, 3, 1, 2, 3],
                #   [4, 5, 6, 4, 5, 6, 4, 5, 6]]

                cim = net.clip_encode_vision(ctemp1).cuda()
                '''
                @torch.no_grad()
                def clip_encode_vision(self, vision, encode_type='encode_vision'):
                    swap_type = self.clip.encode_type
                    self.clip.encode_type = encode_type
                    embedding = self.clip.encode(vision)
                    self.clip.encode_type = swap_type
                    return embedding
                '''
                uim = None # 初始化了一个名为 uim 的变量，并将其设置为 None。uim 似乎被用于存储无条件编码，即不依赖于任何具体输入的编码。
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp1).cuda() # 创建一个与 ctemp1 张量具有相同形状的张量，并将其设置为 0
                    uim = net.clip_encode_vision(dummy).cuda() # 将 dummy 张量传递给 clip_encode_vision() 函数，得到 uim 张量
                conditioning.append(torch.cat([uim, cim]))
                
            elif condition_type == 'video':
                ctemp1 = regularize_video(condition[i]).cuda()
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1, 1)
                cim = net.clip_encode_vision(ctemp1).cuda()
                uim = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp1).cuda()
                    uim = net.clip_encode_vision(dummy).cuda()
                conditioning.append(torch.cat([uim, cim]))
                
            elif condition_type == 'audio':
                ctemp = condition[i][None].repeat(n_samples, 1, 1)
                cad = net.clap_encode_audio(ctemp)
                uad = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp)
                    uad = net.clap_encode_audio(dummy)  
                conditioning.append(torch.cat([uad, cad]))
                
            elif condition_type == 'text':
                ctx = net.clip_encode_text(n_samples * [condition[i]]).cuda()
                utx = None
                if scale != 1.0:
                    utx = net.clip_encode_text(n_samples * [""]).cuda()
                conditioning.append(torch.cat([utx, ctx]))
        
        # 为不同类型的输出(例如图像、视频、文本和音频)创建张量形状(shape)
        shapes = []
        for xtype_i in xtype:
            if xtype_i == 'image':
                h, w = [image_size, image_size]
                shape = [n_samples, 4, h//8, w//8] # // 符号表示整数除法
            elif xtype_i == 'video':
                h, w = [image_size, image_size]
                shape = [n_samples, 4, num_frames, h//8, w//8]
            elif xtype_i == 'text':
                n = 768
                shape = [n_samples, n]
            elif xtype_i == 'audio':
                h, w = [256, 16]
                shape = [n_samples, 8, h, w]
            else:
                raise
            shapes.append(shape)
        
        # 采样过程, 生成样本
        z, _ = sampler.sample(
            steps=ddim_steps,
            shape=shapes,
            condition=conditioning,
            unconditional_guidance_scale=scale,
            xtype=xtype, 
            condition_types=condition_types,
            eta=ddim_eta,
            verbose=False,
            mix_weight=mix_weight)

        # 解码过程, 将采样得到的 z 解码为最终的输出
        out_all = []
        for i, xtype_i in enumerate(xtype):
            z[i] = z[i].cuda()
            x_i = self.decode(z[i], xtype_i)
            out_all.append(x_i)
        return out_all
